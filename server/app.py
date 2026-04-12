"""
server/app.py

WebSocket server that bridges Arduino serial data to a browser-based
training data collection UI.

Usage:
    python -m server
    python -m server --port /dev/cu.usbmodem2101 --http-port 8765
"""

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

from .bridge import SerialBridge
from .dataset import append_sample, make_sample, get_stats, load_samples, delete_sample
from .session import write_session_csv

STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "samples.jsonl"


def build_app(bridge, dataset_path):
    app = web.Application()

    async def on_startup(_app):
        bridge.loop = asyncio.get_running_loop()

    app.on_startup.append(on_startup)

    async def index(_request):
        return web.FileResponse(STATIC_DIR / "index.html")

    async def static_handler(request):
        name = request.match_info["filename"]
        path = STATIC_DIR / name
        if path.exists() and path.is_file():
            return web.FileResponse(path)
        raise web.HTTPNotFound()

    async def api_stats(_request):
        return web.json_response(get_stats(str(dataset_path)))

    async def api_samples(request):
        samples = load_samples(str(dataset_path))
        samples.reverse()
        offset = int(request.query.get("offset", 0))
        limit = int(request.query.get("limit", 50))
        page = samples[offset:offset + limit]
        lightweight = [{
            "id": s["id"],
            "word": s["word"],
            "duration_s": s.get("duration_s", 0),
            "num_samples": s.get("num_samples", 0),
            "created_at": s.get("created_at", ""),
        } for s in page]
        return web.json_response({"samples": lightweight, "total": len(samples)})

    async def api_download(_request):
        path = Path(dataset_path)
        if not path.exists():
            raise web.HTTPNotFound(text="No dataset file yet")
        return web.FileResponse(
            path,
            headers={"Content-Disposition": 'attachment; filename="samples.jsonl"'},
        )

    async def api_upload_audio(request):
        sample_id = request.match_info["id"]
        audio_dir = dataset_path.parent / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{sample_id}.wav"
        body = await request.read()
        with open(audio_path, "wb") as f:
            f.write(body)
        return web.json_response({"saved": True, "path": str(audio_path)})

    async def api_delete_sample(request):
        sample_id = request.match_info["id"]
        if delete_sample(str(dataset_path), sample_id):
            return web.json_response({"deleted": True})
        raise web.HTTPNotFound()

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        q = bridge.subscribe()
        pending_samples = {}

        sessions_dir = dataset_path.parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        auto_id = str(uuid.uuid4())[:8]
        auto_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_jsonl = str(sessions_dir / f"session_{auto_ts}_{auto_id}.jsonl")

        await ws.send_str(json.dumps({
            "type": "status",
            "connected": bridge.connected,
            "port": bridge.port or "",
            "sample_rate_hz": bridge.sample_rate,
        }))

        stats = get_stats(str(dataset_path))
        await ws.send_str(json.dumps({"type": "stats", **stats}))

        async def pump():
            try:
                while True:
                    msg = await q.get()
                    if ws.closed:
                        break
                    await ws.send_str(msg)
            except Exception:
                pass

        pump_task = asyncio.create_task(pump())

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    cmd = data.get("type")

                    if cmd == "start_recording":
                        bridge.start_recording()
                        bridge.add_session_event({"type": "word_start", "t": time.time()})
                        await ws.send_str(json.dumps({"type": "recording_started"}))

                    elif cmd == "stop_recording":
                        stop_t = time.time()
                        buf, start, wall_clock = bridge.stop_recording()
                        sample_id = str(uuid.uuid4())
                        bridge.add_session_event({
                            "type": "word_stop", "t": stop_t, "sample_id": sample_id,
                        })
                        samples_list = [[x, y, z] for x, y, z, _t in buf]
                        timestamps = [round(t - start, 4) for _x, _y, _z, t in buf]
                        duration = timestamps[-1] if timestamps else 0.0
                        pending_samples[sample_id] = (samples_list, timestamps, wall_clock)
                        await ws.send_str(json.dumps({
                            "type": "recording_stopped",
                            "sample_id": sample_id,
                            "samples": samples_list,
                            "timestamps": timestamps,
                            "duration_s": round(duration, 3),
                            "num_samples": len(samples_list),
                            "recorded_at": wall_clock,
                        }))

                    elif cmd == "save_sample":
                        sid = data["sample_id"]
                        word = data["word"].strip()
                        line = data.get("line", 1)
                        with bridge._lock:
                            for evt in reversed(bridge.session_events):
                                if evt["type"] == "word_stop" and "word" not in evt:
                                    evt["word"] = word
                                    evt["line"] = line
                                    evt["sample_id"] = sid
                                    break
                        if sid in pending_samples and word:
                            samples_list, timestamps, wall_clock = pending_samples.pop(sid)
                            audio_dir = dataset_path.parent / "audio"
                            audio_path = audio_dir / f"{sid}.wav"
                            audio_rel = f"audio/{sid}.wav" if audio_path.exists() else None
                            sample = make_sample(word, samples_list, timestamps,
                                                 recorded_at=wall_clock, audio_file=audio_rel)
                            sample["id"] = sid
                            if "line" in data:
                                sample["line"] = data["line"]
                            target = session_jsonl or str(dataset_path)
                            append_sample(target, sample)
                            stats = get_stats(target)
                            await ws.send_str(json.dumps({
                                "type": "sample_saved", "sample_id": sid, **stats,
                            }))

                    elif cmd == "mark_newline":
                        bridge.add_session_event({
                            "type": "newline", "t": time.time(),
                            "to_line": data.get("line", 0),
                        })

                    elif cmd == "discard_sample":
                        pending_samples.pop(data.get("sample_id", ""), None)
                        await ws.send_str(json.dumps({"type": "sample_discarded"}))

                    elif cmd == "start_session":
                        bridge.start_session()
                        wall_clock = bridge.session_wall_clock
                        sid = str(uuid.uuid4())[:8]
                        ts_str = wall_clock[:19].replace(":", "").replace("-", "").replace("T", "_")
                        sessions_dir.mkdir(parents=True, exist_ok=True)
                        base = f"session_{ts_str}_{sid}"
                        session_jsonl = str(sessions_dir / f"{base}.jsonl")
                        await ws.send_str(json.dumps({
                            "type": "session_started", "session_file": f"{base}.jsonl",
                        }))

                    elif cmd == "stop_session":
                        bridge.stop_session()
                        buf_len = len(bridge.session_buffer)
                        start = bridge.session_start
                        duration = (bridge.session_buffer[-1][3] - start) if bridge.session_buffer else 0.0
                        await ws.send_str(json.dumps({
                            "type": "session_stopped",
                            "num_samples": buf_len,
                            "duration_s": round(duration, 2),
                        }))

                    elif cmd == "finalize_session":
                        word_order = data.get("words", [])
                        buf, start, wall_clock, events = bridge.finalize_session()
                        csv_path = Path(session_jsonl).with_suffix(".csv") if session_jsonl else None
                        if csv_path and buf:
                            write_session_csv(csv_path, buf, start, events, word_order)
                        session_name = Path(session_jsonl).stem if session_jsonl else ""
                        session_jsonl = None
                        await ws.send_str(json.dumps({
                            "type": "session_finalized", "file": session_name,
                        }))

                    elif cmd == "get_stats":
                        target = session_jsonl or str(dataset_path)
                        stats = get_stats(target)
                        await ws.send_str(json.dumps({"type": "stats", **stats}))

                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            pump_task.cancel()
            bridge.unsubscribe(q)

        return ws

    app.router.add_get("/", index)
    app.router.add_get("/static/{filename}", static_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/samples", api_samples)
    app.router.add_get("/api/download", api_download)
    app.router.add_post("/api/samples/{id}/audio", api_upload_audio)
    app.router.add_delete("/api/samples/{id}", api_delete_sample)

    return app


def main():
    parser = argparse.ArgumentParser(description="ThePenn Training Data Collector")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--http-port", type=int, default=8765,
                        help="HTTP/WebSocket port (default: 8765)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                        help="Path to JSONL dataset file")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    bridge = SerialBridge(args.port)
    app = build_app(bridge, dataset_path)

    print(f"Starting ThePenn Training Server...")
    print(f"  Serial: {args.port or 'auto-detect'}")
    print(f"  Dataset: {dataset_path}")
    print(f"  UI: http://localhost:{args.http_port}")
    print()

    try:
        web.run_app(app, host="localhost", port=args.http_port, print=None)
    finally:
        bridge.stop()
