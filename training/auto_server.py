"""
training/auto_server.py

Auto-segmenting data collector. Uses the trained SegmentationTCN to detect
word boundaries in real time — no manual button pressing needed.

Usage:
    python3 -m training.auto_server
    python3 -m training.auto_server --words "the quick brown fox"

Flow:
    1. Enter a list of words to write
    2. Click Start — begin writing the first word
    3. The model watches the accel stream and detects when you stop writing
    4. UI flashes "NEXT" and advances to the next word
    5. Samples for each word are saved automatically
"""

import argparse
import asyncio
import csv as csv_mod
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serial_utils import find_arduino_port, open_serial, parse_line
from training.model import SegmentationTCN
from training.dataset import append_sample, make_sample

WHISPER_MODEL_NAME = "mlx-community/whisper-large-v3-mlx"


async def transcribe_audio(wav_path: str) -> str:
    """Transcribe a WAV file using mlx-whisper in a subprocess."""
    script = (
        "import mlx_whisper, json, sys; "
        "r = mlx_whisper.transcribe(sys.argv[1], "
        f"path_or_hf_repo='{WHISPER_MODEL_NAME}', "
        "language='en', word_timestamps=False); "
        "t = r.get('text','').strip().lower().strip('.,!?;:\"\\'()-'); "
        "print(json.dumps(t))"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, wav_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"[transcribe] subprocess error: {stderr.decode()[-300:]}")
        return ""
    try:
        return json.loads(stdout.decode().strip())
    except Exception:
        return stdout.decode().strip().strip('"')

STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "segmenter.pt"

# Detection parameters
BUFFER_SIZE = 256
WRITING_THRESHOLD_HI = 0.55   # P(writing) must rise above this to enter "writing"
WRITING_THRESHOLD_LO = 0.40   # P(writing) must drop below this to leave "writing"
MIN_WORD_SAMPLES = 25         # ~0.5s minimum to count as a real word
MIN_GAP_SAMPLES = 8           # ~0.16s minimum gap before declaring word end


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AutoSegmenter:
    """Runs the TCN on a rolling buffer and detects writing/gap transitions."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.state = "idle"  # idle -> writing -> gap -> writing -> ...
        self.writing_count = 0
        self.gap_count = 0

        # Current word's samples
        self.word_buffer = []
        self.word_start_time = 0.0

    def feed(self, x, y, z, t):
        """Feed one sample. Returns event dict or None.

        Events:
            {"event": "word_start", "t": ...}
            {"event": "word_end", "t": ..., "samples": [...], "duration_s": ...}
        """
        self.buffer.append((x, y, z))

        if len(self.buffer) < 30:
            return None

        # Run inference
        arr = np.array(list(self.buffer), dtype=np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            logits = self.model(inp)
            prob = torch.sigmoid(logits[0, -1]).item()

        if self.state == "idle":
            if prob > WRITING_THRESHOLD_HI:
                self.writing_count += 1
                if self.writing_count >= 3:
                    self.state = "writing"
                    self.gap_count = 0
                    self.word_buffer = []
                    self.word_start_time = t
                    return {"event": "word_start", "t": t, "prob": prob}
            else:
                self.writing_count = 0

        elif self.state == "writing":
            self.word_buffer.append([x, y, z, t])

            if prob < WRITING_THRESHOLD_LO:
                self.gap_count += 1
            else:
                self.gap_count = 0

            if self.gap_count >= MIN_GAP_SAMPLES and len(self.word_buffer) >= MIN_WORD_SAMPLES:
                self.state = "idle"
                self.writing_count = 0
                self.gap_count = 0
                # Trim the trailing gap samples from the word
                trim = min(MIN_GAP_SAMPLES, len(self.word_buffer))
                word_samples = self.word_buffer[:-trim]
                samples_xyz = [[s[0], s[1], s[2]] for s in word_samples]
                timestamps = [s[3] - self.word_start_time for s in word_samples]
                duration = timestamps[-1] if timestamps else 0.0
                self.word_buffer = []
                return {
                    "event": "word_end",
                    "t": t,
                    "prob": prob,
                    "sample_id": str(uuid.uuid4()),
                    "samples": samples_xyz,
                    "timestamps": timestamps,
                    "duration_s": round(duration, 3),
                    "num_samples": len(samples_xyz),
                }

        return {"event": "prob", "prob": prob}


class SerialReader:
    """Reads serial in a background thread, pushes to subscribers + segmenter."""

    def __init__(self, port, loop, segmenter):
        self.port = port
        self.loop = loop
        self.segmenter = segmenter
        self.connected = False
        self.running = True
        self.sample_rate = 0.0
        self.subscribers = set()
        self.ring = deque(maxlen=200)
        self.session_buffer = []
        self.session_active = False
        self.session_start = 0.0
        self.recording_word = False

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        ser = None
        rate_n = 0
        rate_t = time.time()

        while self.running:
            if ser is None:
                try:
                    port = self.port or find_arduino_port()
                    if not port:
                        self._broadcast_status(False)
                        time.sleep(2)
                        continue
                    ser = open_serial(port)
                    self.port = port
                    self._broadcast_status(True)
                    rate_n = 0
                    rate_t = time.time()
                except Exception:
                    self._broadcast_status(False)
                    time.sleep(2)
                    continue

            try:
                raw = ser.readline()
                if not raw:
                    continue
                parsed = parse_line(raw)
                if parsed is None:
                    continue

                x, y, z = parsed
                t = time.time()

                rate_n += 1
                if t - rate_t >= 2.0:
                    self.sample_rate = round(rate_n / (t - rate_t), 1)
                    rate_n = 0
                    rate_t = t

                self.ring.append((x, y, z, t))

                # Session CSV
                if self.session_active:
                    self.session_buffer.append((x, y, z, t, 1 if self.recording_word else 0))

                # Run segmenter
                event = self.segmenter.feed(x, y, z, t)

                # Push accel + detection events
                msg = json.dumps({"type": "accel", "x": x, "y": y, "z": z, "t": t})
                self._push(msg)

                if event and event["event"] in ("word_start", "word_end"):
                    self._push(json.dumps({"type": event["event"], **event}))
                    if event["event"] == "word_start":
                        self.recording_word = True
                    elif event["event"] == "word_end":
                        self.recording_word = False

                if event and "prob" in event:
                    self._push(json.dumps({"type": "prob", "p": round(event["prob"], 3)}))

            except Exception:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                self._broadcast_status(False)
                time.sleep(1)

    def _broadcast_status(self, val):
        self.connected = val
        msg = json.dumps({
            "type": "status", "connected": val,
            "port": self.port or "", "sample_rate_hz": self.sample_rate,
        })
        self._push(msg)

    def _push(self, msg):
        if self.loop:
            for q in list(self.subscribers):
                try:
                    self.loop.call_soon_threadsafe(q.put_nowait, msg)
                except (asyncio.QueueFull, RuntimeError):
                    pass

    def subscribe(self):
        q = asyncio.Queue(maxsize=200)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q):
        self.subscribers.discard(q)

    def stop(self):
        self.running = False


def build_app(reader):
    app = web.Application()

    async def on_startup(app):
        reader.loop = asyncio.get_running_loop()
    app.on_startup.append(on_startup)

    async def index(request):
        return web.FileResponse(STATIC_DIR / "auto.html")

    async def static_handler(request):
        name = request.match_info["filename"]
        path = STATIC_DIR / name
        if path.exists():
            return web.FileResponse(path)
        raise web.HTTPNotFound()

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        q = reader.subscribe()

        await ws.send_str(json.dumps({
            "type": "status", "connected": reader.connected,
            "port": reader.port or "", "sample_rate_hz": reader.sample_rate,
        }))

        sessions_dir = DATA_DIR / "sessions" / "auto"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        auto_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        auto_id = str(uuid.uuid4())[:8]
        session_jsonl = sessions_dir / f"session_{auto_ts}_{auto_id}.jsonl"

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

                    if cmd == "start_session":
                        reader.session_active = True
                        reader.session_buffer = []
                        reader.session_start = time.time()
                        reader.segmenter.state = "idle"
                        reader.segmenter.writing_count = 0
                        reader.segmenter.gap_count = 0
                        reader.segmenter.buffer.clear()
                        await ws.send_str(json.dumps({"type": "session_started"}))

                    elif cmd == "stop_session":
                        reader.session_active = False
                        await ws.send_str(json.dumps({"type": "session_stopped"}))

                    elif cmd == "save_word":
                        word = data["word"]
                        sid = data.get("sample_id", str(uuid.uuid4()))
                        samples_xyz = data["samples"]
                        timestamps = data["timestamps"]
                        wall_clock = datetime.now(timezone.utc).isoformat()
                        audio_path = DATA_DIR / "audio" / f"{sid}.wav"
                        audio_rel = f"audio/{sid}.wav" if audio_path.exists() else None
                        sample = make_sample(word, samples_xyz, timestamps,
                                             recorded_at=wall_clock,
                                             audio_file=audio_rel)
                        sample["id"] = sid
                        append_sample(str(session_jsonl), sample)
                        print(f"[save] '{word}' -> {Path(session_jsonl).name}")
                        await ws.send_str(json.dumps({
                            "type": "word_saved", "word": word,
                            "sample_id": sid,
                            "file": Path(session_jsonl).name,
                        }))

                    elif cmd == "transcribe":
                        sample_id = data["sample_id"]
                        audio_path = DATA_DIR / "audio" / f"{sample_id}.wav"
                        if audio_path.exists():
                            print(f"[transcribe] starting: {sample_id}")
                            try:
                                text = await transcribe_audio(str(audio_path))
                            except Exception as e:
                                print(f"[transcribe] ERROR: {e}")
                                text = ""
                            print(f"[transcribe] result: '{text}'")
                            await ws.send_str(json.dumps({
                                "type": "transcription",
                                "sample_id": sample_id,
                                "text": text,
                            }))
                        else:
                            print(f"[transcribe] file not found: {audio_path}")
                            await ws.send_str(json.dumps({
                                "type": "transcription",
                                "sample_id": sample_id,
                                "text": "",
                            }))

                    elif cmd == "save_session_csv":
                        word_events = data.get("words", [])
                        buf = list(reader.session_buffer)
                        start = reader.session_start
                        if buf:
                            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            sid = str(uuid.uuid4())[:8]
                            csv_path = sessions_dir / f"session_{ts}_{sid}.csv"
                            jsonl_path = csv_path.with_suffix(".jsonl")

                            # Build word time ranges from events
                            word_ranges = []
                            for we in word_events:
                                word_ranges.append((
                                    we["t_start"] , we["t_end"],
                                    we["word"],
                                ))

                            with open(csv_path, "w", newline="") as f:
                                wr = csv_mod.writer(f)
                                wr.writerow(["elapsed_s", "x_g", "y_g", "z_g",
                                             "writing", "word", "newline"])
                                for x, y, z, t, writing_flag in buf:
                                    elapsed = t - start
                                    word_label = ""
                                    for t_s, t_e, w in word_ranges:
                                        if t_s <= t <= t_e:
                                            word_label = w
                                            break
                                    wr.writerow([f"{elapsed:.4f}", f"{x:.4f}",
                                                 f"{y:.4f}", f"{z:.4f}",
                                                 writing_flag, word_label, 0])

                            # Also save each word as JSONL sample
                            for we in word_events:
                                sid = we.get("sample_id", str(uuid.uuid4()))
                                audio_path = DATA_DIR / "audio" / f"{sid}.wav"
                                audio_rel = f"audio/{sid}.wav" if audio_path.exists() else None
                                sample = make_sample(
                                    we["word"], we["samples"], we["timestamps"],
                                    recorded_at=datetime.now(timezone.utc).isoformat(),
                                    audio_file=audio_rel,
                                )
                                sample["id"] = sid
                                append_sample(str(jsonl_path), sample)

                            await ws.send_str(json.dumps({
                                "type": "session_saved",
                                "csv": csv_path.name,
                                "jsonl": jsonl_path.name,
                                "words": len(word_events),
                            }))

                    elif cmd == "send_to_ara":
                        words = data.get("words", [])
                        subject = data.get("subject", "general").strip()
                        action = data.get("action", "save").strip()
                        text = " ".join(w.strip() for w in words if w.strip())

                        if not text:
                            await ws.send_str(json.dumps({
                                "type": "ara_response",
                                "text": "No text to send.",
                            }))
                        else:
                            await ws.send_str(json.dumps({
                                "type": "ara_status",
                                "text": "Sending to ThePenn study agent...",
                            }))

                            ara_app_path = str(Path(__file__).resolve().parent.parent / "ara_app.py")
                            if action == "save":
                                msg_text = f"New notes for {subject}: {text}"
                            elif action == "flashcards":
                                msg_text = f"Generate flashcards for my {subject} notes"
                            elif action == "quiz":
                                msg_text = f"Quiz me on my {subject} notes"
                            elif action == "summary":
                                msg_text = f"Summarize my {subject} notes"
                            else:
                                msg_text = text

                            try:
                                ara_env = os.environ.copy()
                                ara_env.setdefault("ARA_API_KEY", "")
                                ara_env.setdefault("ARA_RUNTIME_KEY", "")
                                proc = await asyncio.create_subprocess_exec(
                                    sys.executable, "-m", "ara_sdk",
                                    "run", ara_app_path,
                                    "--agent", "study",
                                    "--message", msg_text,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                    cwd=str(Path(__file__).resolve().parent.parent),
                                    env=ara_env,
                                )
                                stdout, stderr = await proc.communicate()
                                response = stdout.decode().strip()
                                if proc.returncode != 0:
                                    err = stderr.decode().strip()[-500:]
                                    print(f"[ara] error: {err}")
                                    response = f"Ara agent error: {err}"
                                else:
                                    print(f"[ara] response: {response[:200]}")
                            except Exception as e:
                                response = f"Failed to reach Ara agent: {e}"
                                print(f"[ara] exception: {e}")

                            await ws.send_str(json.dumps({
                                "type": "ara_response",
                                "text": response,
                                "action": action,
                            }))

                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            pump_task.cancel()
            reader.unsubscribe(q)
        return ws

    async def api_upload_audio(request):
        sample_id = request.match_info["id"]
        audio_dir = DATA_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        body = await request.read()
        audio_path = audio_dir / f"{sample_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(body)
        return web.json_response({"saved": True, "path": str(audio_path)})

    async def api_transcribe(request):
        sample_id = request.match_info["id"]
        audio_path = DATA_DIR / "audio" / f"{sample_id}.wav"
        if not audio_path.exists():
            print(f"[transcribe] file not found: {audio_path}")
            raise web.HTTPNotFound(text="Audio file not found")
        print(f"[transcribe] starting: {sample_id} ({audio_path.stat().st_size} bytes)")
        try:
            text = await transcribe_audio(str(audio_path))
            print(f"[transcribe] result: '{text}'")
            return web.json_response({"sample_id": sample_id, "text": text})
        except Exception as e:
            print(f"[transcribe] ERROR: {e}")
            import traceback; traceback.print_exc()
            return web.json_response({"sample_id": sample_id, "text": "", "error": str(e)})

    async def notes_page(request):
        return web.FileResponse(STATIC_DIR / "notes.html")

    async def api_ara(request):
        """HTTP endpoint for Ara agent — used by the notes UI."""
        data = await request.json()
        action = data.get("action", "save")
        subject = data.get("subject", "general").strip()
        text = data.get("text", "").strip()

        if not text:
            return web.json_response({"text": "No text provided.", "error": True})

        ara_app_path = str(Path(__file__).resolve().parent.parent / "ara_app.py")

        if action == "save":
            msg_text = f"New notes for {subject}: {text}"
        elif action == "flashcards":
            msg_text = f"Generate flashcards for my {subject} notes"
        elif action == "quiz":
            msg_text = f"Quiz me on my {subject} notes"
        elif action == "summary":
            msg_text = f"Summarize my {subject} notes"
        else:
            msg_text = text

        try:
            ara_env = os.environ.copy()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ara_sdk",
                "run", ara_app_path,
                "--agent", "study",
                "--message", msg_text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parent.parent),
                env=ara_env,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode().strip()[-500:]
                print(f"[ara] error: {err}")
                return web.json_response({"text": f"Ara error: {err}", "error": True})

            result = json.loads(stdout.decode())
            output = result.get("result", {}).get("output_text", "No response from agent.")
            print(f"[ara] {action}: {output[:100]}...")
            return web.json_response({"text": output, "action": action})
        except Exception as e:
            print(f"[ara] exception: {e}")
            return web.json_response({"text": f"Failed to reach Ara: {e}", "error": True})

    app.router.add_get("/", index)
    app.router.add_get("/notes", notes_page)
    app.router.add_get("/static/{filename}", static_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/api/samples/{id}/audio", api_upload_audio)
    app.router.add_get("/api/samples/{id}/transcribe", api_transcribe)
    app.router.add_post("/api/ara", api_ara)
    return app


def main():
    parser = argparse.ArgumentParser(description="Ghost Writer Auto-Segmenting Collector")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--http-port", type=int, default=8766)
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}\nRun: python3 -m training.train_segmenter")

    device = get_device()
    print(f"Loading model from {args.model} on {device}...")
    model = SegmentationTCN(in_channels=3, hidden=64, kernel_size=3,
                            num_blocks=5, dropout=0.15).to(device)
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))
    model.eval()

    segmenter = AutoSegmenter(model, device)
    reader = SerialReader(args.port, None, segmenter)
    app = build_app(reader)

    print(f"Auto-Segmenting Collector")
    print(f"  Serial: {args.port or 'auto-detect'}")
    print(f"  Model: {args.model}")
    print(f"  UI: http://localhost:{args.http_port}")
    print()

    try:
        web.run_app(app, host="localhost", port=args.http_port, print=None)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
