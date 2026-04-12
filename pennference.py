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
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import aiohttp
import numpy as np
import torch
from aiohttp import web

from serial_utils import find_arduino_port, open_serial, parse_line
from training.model import SegmentationTCN, WordClassifier
from training.data_pipeline import trim_idle, compute_features

# Load .env file
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

STATIC_DIR = Path(__file__).resolve().parent / "training" / "static"
SEGMENTER_PATH = Path(__file__).resolve().parent / "segmenter.pt"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "word_classifier_best.pt"
DATA_DIR = Path(__file__).resolve().parent / "training_data"
MODELS_DIR = Path(__file__).resolve().parent / "models"

DEFAULT_CONFIDENCE = 0.5
DEFAULT_NOTES_URL = "http://localhost:8767/api/word"
DEFAULT_RERANK_THRESHOLD = 0.7

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
# LLM re-ranking (Anthropic API)
# ---------------------------------------------------------------------------

async def rerank_with_llm(anthropic_client, recent_words, top3):
    """Ask Claude to pick the most likely word given context.

    Returns the chosen word (must be one of the top3 candidates) or None.
    """
    if not anthropic_client:
        return None

    context = " ".join(recent_words) if recent_words else "(start of sentence)"
    candidates_str = ", ".join(
        f"{c['word']} ({c['prob']:.0%})" for c in top3)

    try:
        resp = await asyncio.wait_for(
            anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                system="You are a word predictor for handwritten sentence recognition. "
                       "Given the recent words and classifier candidates, pick the word that "
                       "makes the sentence grammatically correct and coherent. "
                       "Prioritize grammatical sense over classifier confidence. "
                       "Reply with ONLY the word, nothing else.",
                messages=[{
                    "role": "user",
                    "content": f"Recent words: {context}\n"
                               f"Candidates: {candidates_str}\n"
                               f"Which word comes next?",
                }],
            ),
            timeout=5.0,
        )
        pick = resp.content[0].text.strip().lower()
        valid_words = {c["word"] for c in top3}
        if pick in valid_words:
            print(f"[rerank] LLM picked '{pick}' from {candidates_str}")
            return pick
        print(f"[rerank] LLM returned '{pick}' — not in candidates, ignoring")
        return None
    except asyncio.TimeoutError:
        print("[rerank] LLM timed out")
        return None
    except Exception as e:
        print(f"[rerank] LLM error: {e}")
        return None


# ---------------------------------------------------------------------------
# Training: ElevenLabs STT + retrain
# ---------------------------------------------------------------------------

def transcribe_audio_sync(elevenlabs_client, wav_path, vocabulary):
    """Transcribe a WAV file using ElevenLabs STT. Returns the text or ""."""
    try:
        with open(wav_path, "rb") as f:
            resp = elevenlabs_client.speech_to_text.convert(
                model_id="scribe_v2",
                file=f,
                language_code="en",
                tag_audio_events=False,
            )
        text = resp.text.strip().lower().strip(".,!?;:\"'()-")
        if text:
            text = text.split()[0]
        return text
    except Exception as e:
        print(f"[stt] ElevenLabs error: {e}")
        return ""


def retrain_classifier_sync(data_dir, save_dir, device):
    """Retrain the WordClassifier from scratch on all available data.

    Returns (model, idx_to_word, word_to_idx, num_words, val_acc) or raises.
    """
    from training.data_pipeline import WordDataset, collate_word
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    dataset = WordDataset(str(data_dir), augment_data=True)
    num_words = dataset.num_words
    if len(dataset) < 5:
        raise ValueError(f"Only {len(dataset)} samples — need at least 5")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=collate_word)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            collate_fn=collate_word)

    model = WordClassifier(num_features=10, num_words=num_words).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs = 50

    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels, lengths in train_loader:
            features, labels, lengths = (
                features.to(device), labels.to(device), lengths.to(device))
            loss = criterion(model(features, lengths), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features, labels, lengths = (
                    features.to(device), labels.to(device), lengths.to(device))
                val_correct += (model(features, lengths).argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"[retrain] epoch {epoch}/{epochs}  val_acc={val_acc:.3f}")

    # Save checkpoint
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "word_classifier_best.pt"
    if best_state:
        model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_to_idx": dataset.word_to_idx,
        "idx_to_word": dataset.idx_to_word,
        "num_words": num_words,
        "val_acc": best_val_acc,
    }, save_path)

    # Per-word sample counts
    word_counts = {}
    for s in dataset.samples:
        w = s["word"].lower()
        word_counts[w] = word_counts.get(w, 0) + 1

    print(f"[retrain] done — {num_words} words, {len(dataset)} samples, val_acc={best_val_acc:.3f}")
    return {
        "model": model,
        "idx_to_word": dataset.idx_to_word,
        "word_to_idx": dataset.word_to_idx,
        "num_words": num_words,
        "val_acc": best_val_acc,
        "total_samples": len(dataset),
        "word_counts": word_counts,
    }


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------

async def forward_word(session, notes_url, word, confidence):
    """POST a confirmed word to the notes server. Fire-and-forget."""
    try:
        async with session.post(notes_url, json={
            "word": word,
            "confidence": confidence,
        }) as resp:
            if resp.status == 200:
                print(f"[forward] '{word}' -> {notes_url}")
            else:
                print(f"[forward] '{word}' got status {resp.status}")
    except Exception as e:
        print(f"[forward] '{word}' failed: {e}")


def build_app(reader, cls_model, idx_to_word, device, confidence, notes_url,
              anthropic_client=None, rerank_threshold=0.7,
              elevenlabs_client=None):
    app = web.Application()
    app["recent_words"] = deque(maxlen=5)
    # Mutable state dict avoids aiohttp deprecation warning for post-startup changes
    app["state"] = {
        "cls_model": cls_model,
        "idx_to_word": idx_to_word,
        "forward_enabled": True,
    }

    async def on_startup(app):
        reader.loop = asyncio.get_running_loop()
        app["http_session"] = aiohttp.ClientSession()
    app.on_startup.append(on_startup)

    async def on_cleanup(app):
        await app["http_session"].close()
    app.on_cleanup.append(on_cleanup)

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
            "vocabulary": sorted(request.app["state"]["idx_to_word"].values()),
            "training_available": elevenlabs_client is not None,
            "forward_enabled": request.app["state"]["forward_enabled"],
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

                    if data.get("type") == "toggle_forward":
                        request.app["state"]["forward_enabled"] = data.get("enabled", True)
                        state = request.app["state"]["forward_enabled"]
                        print(f"[forward] {'enabled' if state else 'disabled'}")
                        await ws.send_str(json.dumps({
                            "type": "forward_state",
                            "enabled": state,
                        }))

                    elif data.get("type") == "classify":
                        samples = data["samples"]
                        word_idx = data.get("word_idx", 0)

                        # Use app-level refs so hot-swap works
                        cur_model = request.app["state"]["cls_model"]
                        cur_idx = request.app["state"]["idx_to_word"]

                        # Run classification in thread pool
                        word, conf, top3 = await loop.run_in_executor(
                            None, classify_word_sync,
                            cur_model, cur_idx, samples, device)

                        reranked = False
                        original_word = word

                        # LLM re-ranking for grammatical coherence
                        if (anthropic_client and len(top3) > 1):
                            await ws.send_str(json.dumps({
                                "type": "reranking",
                                "word_idx": word_idx,
                            }))
                            recent = list(request.app["recent_words"])
                            pick = await rerank_with_llm(
                                anthropic_client, recent, top3)
                            if pick and pick != word:
                                # Find the candidate's confidence
                                for c in top3:
                                    if c["word"] == pick:
                                        conf = c["prob"]
                                        break
                                word = pick
                                reranked = True

                        accepted = True
                        await ws.send_str(json.dumps({
                            "type": "classification",
                            "word_idx": word_idx,
                            "word": word,
                            "confidence": conf,
                            "top3": top3,
                            "accepted": accepted,
                            "reranked": reranked,
                            "original_word": original_word if reranked else None,
                        }))

                        # Track accepted words for context
                        if accepted:
                            request.app["recent_words"].append(word)

                        # Forward accepted words to notes server
                        if accepted and notes_url and request.app["state"]["forward_enabled"]:
                            asyncio.create_task(
                                forward_word(request.app["http_session"],
                                             notes_url, word, conf))

                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            pump_task.cancel()
            reader.unsubscribe(q)
        return ws

    # --- Training endpoints ---

    async def api_training_audio(request):
        """Receive WAV audio blob from browser for a training sample."""
        word_idx = request.match_info["word_idx"]
        audio_dir = DATA_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        body = await request.read()
        import uuid
        audio_id = str(uuid.uuid4())
        audio_path = audio_dir / f"{audio_id}.wav"
        audio_path.write_bytes(body)
        print(f"[training] audio saved: {audio_path.name} ({len(body)} bytes)")
        return web.json_response({
            "audio_id": audio_id,
            "path": str(audio_path),
        })

    async def api_training_transcribe(request):
        """Transcribe audio via ElevenLabs STT."""
        data = await request.json()
        audio_id = data.get("audio_id", "")
        audio_path = DATA_DIR / "audio" / f"{audio_id}.wav"
        if not audio_path.exists():
            return web.json_response({"error": "audio not found"}, status=404)
        if not elevenlabs_client:
            return web.json_response({"error": "ElevenLabs not configured"}, status=500)

        vocab = sorted(request.app["state"]["idx_to_word"].values())
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, transcribe_audio_sync,
            elevenlabs_client, str(audio_path), vocab)

        is_word = bool(text and len(text.split()) == 1)
        print(f"[training] STT: '{text}' (is_word={is_word})")
        return web.json_response({
            "text": text,
            "is_word": is_word,
            "audio_id": audio_id,
        })

    async def api_training_save(request):
        """Save reviewed training samples to JSONL."""
        from training.dataset import make_sample, append_sample
        from datetime import datetime, timezone

        data = await request.json()
        samples_list = data.get("samples", [])
        if not samples_list:
            return web.json_response({"error": "no samples"}, status=400)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sessions_dir = DATA_DIR / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = sessions_dir / f"training_{ts}.jsonl"

        saved = 0
        for s in samples_list:
            word = s.get("word", "").strip().lower()
            if not word:
                continue
            sample = make_sample(
                word, s["samples"], s["timestamps"],
                audio_file=s.get("audio_file"))
            append_sample(str(jsonl_path), sample)
            saved += 1

        print(f"[training] saved {saved} samples to {jsonl_path.name}")
        return web.json_response({
            "saved": saved,
            "file": jsonl_path.name,
        })

    async def api_training_retrain(request):
        """Retrain the classifier and hot-swap it."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, retrain_classifier_sync,
                str(DATA_DIR), str(MODELS_DIR), device)

            # Hot-swap
            request.app["state"]["cls_model"] = result["model"]
            request.app["state"]["idx_to_word"] = result["idx_to_word"]
            vocab = sorted(result["idx_to_word"].values())

            print(f"[training] hot-swapped model: {result['num_words']} words, "
                  f"{result['total_samples']} samples, val_acc={result['val_acc']:.3f}")
            return web.json_response({
                "num_words": result["num_words"],
                "val_acc": round(result["val_acc"], 3),
                "total_samples": result["total_samples"],
                "word_counts": result["word_counts"],
                "vocabulary": vocab,
            })
        except Exception as e:
            print(f"[training] retrain failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def api_training_reclassify(request):
        """Re-classify samples with the current (updated) model.
        Used after retrain to show before/after comparison on each card."""
        data = await request.json()
        items = data.get("items", [])
        cur_model = request.app["state"]["cls_model"]
        cur_idx = request.app["state"]["idx_to_word"]
        loop = asyncio.get_running_loop()
        results = []
        for item in items:
            word, conf, top3 = await loop.run_in_executor(
                None, classify_word_sync,
                cur_model, cur_idx, item["samples"], device)
            results.append({
                "word_idx": item["word_idx"],
                "word": word,
                "confidence": conf,
                "top3": top3,
            })
        return web.json_response({"results": results})

    app.router.add_get("/", index)
    app.router.add_get("/static/{filename}", static_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/api/training/audio/{word_idx}", api_training_audio)
    app.router.add_post("/api/training/transcribe", api_training_transcribe)
    app.router.add_post("/api/training/save", api_training_save)
    app.router.add_post("/api/training/retrain", api_training_retrain)
    app.router.add_post("/api/training/reclassify", api_training_reclassify)
    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ThePenn inference dashboard")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detect if omitted)")
    parser.add_argument("--http-port", type=int, default=8768,
                        help="Dashboard port (default: 8768)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--segmenter", type=str, default=str(SEGMENTER_PATH))
    parser.add_argument("--classifier", type=str, default=str(CLASSIFIER_PATH))
    parser.add_argument("--notes-url", type=str, default=DEFAULT_NOTES_URL,
                        help=f"URL to POST confirmed words (default: {DEFAULT_NOTES_URL}). "
                             "Set to empty string to disable forwarding.")
    parser.add_argument("--llm-rerank", action="store_true",
                        help="Enable LLM re-ranking for low-confidence predictions")
    parser.add_argument("--rerank-threshold", type=float, default=DEFAULT_RERANK_THRESHOLD,
                        help=f"Confidence below which LLM re-ranking kicks in (default: {DEFAULT_RERANK_THRESHOLD})")
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

    notes_url = args.notes_url.strip() or None

    # LLM re-ranking
    anthropic_client = None
    if args.llm_rerank:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise SystemExit("--llm-rerank requires ANTHROPIC_API_KEY in .env or environment")
        import anthropic
        anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        print(f"LLM re-ranking enabled (threshold: {args.rerank_threshold:.0%})")

    # ElevenLabs STT for training mode
    elevenlabs_client = None
    el_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if el_key:
        from elevenlabs import ElevenLabs
        elevenlabs_client = ElevenLabs(api_key=el_key)
        print("ElevenLabs STT enabled (training mode available)")

    segmenter = AutoSegmenter(seg_model, device)
    reader = SerialReader(args.port, None, segmenter)
    app = build_app(reader, cls_model, idx_to_word, device, args.confidence, notes_url,
                    anthropic_client=anthropic_client, rerank_threshold=args.rerank_threshold,
                    elevenlabs_client=elevenlabs_client)

    print(f"Pennference Dashboard")
    print(f"  Serial:    {args.port or 'auto-detect'}")
    print(f"  Dashboard: http://localhost:{args.http_port}")
    print(f"  Forward:   {notes_url or 'disabled'}")
    print(f"  LLM:       {'enabled' if anthropic_client else 'disabled'}")
    print(f"  Training:  {'enabled' if elevenlabs_client else 'disabled (no ELEVENLABS_API_KEY)'}")
    print()

    try:
        web.run_app(app, host="localhost", port=args.http_port, print=None)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
