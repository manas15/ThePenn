"""
pennference.py

End-to-end inference pipeline with web dashboard.

Architecture:
  - SerialReader thread: reads accelerometer, feeds AutoSegmenter, pushes
    accel data + segmenter events over WebSocket (never blocks)
  - Classification: runs in a thread pool when word_end fires, result
    pushed back to the browser
  - Browser: shows live accel chart, P(writing) bar, pipeline log,
    captured words queue, and recognized output sentence

Usage:
    python3 pennference.py
    python3 pennference.py --confidence 0.6
    python3 pennference.py --port /dev/cu.usbmodem2101
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

from serial_utils import find_arduino_port, open_serial, parse_line
from training.model import SegmentationTCN, WordClassifier
from training.data_pipeline import trim_idle, compute_features

STATIC_DIR = Path(__file__).resolve().parent / "training" / "static"
SEGMENTER_PATH = Path(__file__).resolve().parent / "segmenter.pt"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "word_classifier_best.pt"

DEFAULT_CONFIDENCE = 0.5

# Segmentation parameters
BUFFER_SIZE = 256
WRITING_THRESHOLD_HI = 0.55
WRITING_THRESHOLD_LO = 0.40
MIN_WORD_SAMPLES = 25
MIN_GAP_SAMPLES = 8


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# AutoSegmenter (same as auto_server.py)
# ---------------------------------------------------------------------------

class AutoSegmenter:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.state = "idle"
        self.writing_count = 0
        self.gap_count = 0
        self.word_buffer = []
        self.word_start_time = 0.0

    def feed(self, x, y, z, t):
        self.buffer.append((x, y, z))
        if len(self.buffer) < 30:
            return None

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
                    "samples": samples_xyz,
                    "timestamps": timestamps,
                    "duration_s": round(duration, 3),
                    "num_samples": len(samples_xyz),
                }

        return {"event": "prob", "prob": prob}


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------

class SerialReader:
    def __init__(self, port, loop, segmenter):
        self.port = port
        self.loop = loop
        self.segmenter = segmenter
        self.connected = False
        self.running = True
        self.sample_rate = 0.0
        self.subscribers = set()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        ser = None
        rate_n = 0
        rate_t = time.time()
        prev_raw = None

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

                # Push raw accel to browser
                self._push(json.dumps({"type": "accel", "x": x, "y": y, "z": z, "t": t}))

                # Run segmenter
                event = self.segmenter.feed(x, y, z, t)

                if event and event["event"] in ("word_start", "word_end"):
                    self._push(json.dumps({"type": event["event"], **event}))
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


# ---------------------------------------------------------------------------
# Classification (runs in thread pool)
# ---------------------------------------------------------------------------

def classify_word_sync(cls_model, idx_to_word, samples_xyz, device):
    trimmed = trim_idle(samples_xyz)
    features = compute_features(trimmed)
    features_t = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([features_t.size(1)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = cls_model(features_t, lengths)
        probs = torch.softmax(logits, dim=1)[0]

    confidence, pred_idx = probs.max(dim=0)
    word = idx_to_word[pred_idx.item()]

    top_k = torch.topk(probs, min(3, len(probs)))
    top3 = [{"word": idx_to_word[i.item()], "prob": round(p.item(), 4)}
            for i, p in zip(top_k.indices, top_k.values)]

    return word, round(confidence.item(), 4), top3


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------

def build_app(reader, cls_model, idx_to_word, device, confidence):
    app = web.Application()

    async def on_startup(app):
        reader.loop = asyncio.get_running_loop()
    app.on_startup.append(on_startup)

    async def index(request):
        return web.FileResponse(STATIC_DIR / "pennference.html")

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

        # Send initial state
        await ws.send_str(json.dumps({
            "type": "status",
            "connected": reader.connected,
            "port": reader.port or "",
            "sample_rate_hz": reader.sample_rate,
        }))
        await ws.send_str(json.dumps({
            "type": "config",
            "confidence": confidence,
            "vocabulary": sorted(idx_to_word.values()),
        }))

        loop = asyncio.get_running_loop()

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

                    if data.get("type") == "classify":
                        samples = data["samples"]
                        word_idx = data.get("word_idx", 0)

                        # Run classification in thread pool
                        word, conf, top3 = await loop.run_in_executor(
                            None, classify_word_sync,
                            cls_model, idx_to_word, samples, device)

                        await ws.send_str(json.dumps({
                            "type": "classification",
                            "word_idx": word_idx,
                            "word": word,
                            "confidence": conf,
                            "top3": top3,
                            "accepted": conf >= confidence,
                        }))

                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            pump_task.cancel()
            reader.unsubscribe(q)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/static/{filename}", static_handler)
    app.router.add_get("/ws", ws_handler)
    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ThePenn inference dashboard")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--http-port", type=int, default=8767)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--segmenter", type=str, default=str(SEGMENTER_PATH))
    parser.add_argument("--classifier", type=str, default=str(CLASSIFIER_PATH))
    args = parser.parse_args()

    if not Path(args.segmenter).exists():
        raise SystemExit(f"Segmenter not found: {args.segmenter}\nRun: python3 -m training.train_segmenter")
    if not Path(args.classifier).exists():
        raise SystemExit(f"Classifier not found: {args.classifier}\nRun: python3 -m training.train")

    device = get_device()

    # Load segmenter
    print(f"Loading segmenter from {args.segmenter}...")
    seg_model = SegmentationTCN(
        in_channels=3, hidden=64, kernel_size=3,
        num_blocks=5, dropout=0.15).to(device)
    seg_model.load_state_dict(
        torch.load(args.segmenter, weights_only=True, map_location=device))
    seg_model.eval()

    # Load classifier
    print(f"Loading classifier from {args.classifier}...")
    ckpt = torch.load(args.classifier, weights_only=False, map_location=device)
    idx_to_word = ckpt["idx_to_word"]
    num_words = ckpt["num_words"]
    cls_model = WordClassifier(num_features=10, num_words=num_words).to(device)
    cls_model.load_state_dict(ckpt["model_state_dict"])
    cls_model.eval()

    vocab = sorted(idx_to_word.values())
    print(f"Device: {device}")
    print(f"Vocabulary ({num_words}): {', '.join(vocab)}")
    print(f"Confidence: {args.confidence:.0%}")
    print()

    segmenter = AutoSegmenter(seg_model, device)
    reader = SerialReader(args.port, None, segmenter)
    app = build_app(reader, cls_model, idx_to_word, device, args.confidence)

    print(f"Pennference Dashboard")
    print(f"  Serial: {args.port or 'auto-detect'}")
    print(f"  UI: http://localhost:{args.http_port}")
    print()

    try:
        web.run_app(app, host="localhost", port=args.http_port, print=None)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
