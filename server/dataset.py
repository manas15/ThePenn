"""
server/dataset.py

JSONL dataset management for training samples.
Each line is a JSON object pairing accelerometer data with a word label.
"""

import glob
import json
import os
import uuid
from datetime import datetime, timezone


def make_sample(word, samples, timestamps, recorded_at=None, audio_file=None):
    """Create a sample dict ready for JSONL storage."""
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    n = len(samples)
    rate = n / duration if duration > 0 else 0.0
    s = {
        "id": str(uuid.uuid4()),
        "word": word,
        "samples": samples,
        "timestamps": timestamps,
        "sample_rate_hz": round(rate, 1),
        "duration_s": round(duration, 3),
        "num_samples": n,
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if audio_file:
        s["audio_file"] = audio_file
    return s


def append_sample(filepath, sample):
    """Append one sample as a JSONL line."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(sample) + "\n")
        f.flush()


def load_samples(filepath):
    """Read all samples from a JSONL file. Returns [] if missing."""
    if not os.path.exists(filepath):
        return []
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def delete_sample(filepath, sample_id):
    """Remove a sample by ID (rewrites the file). Returns True if found."""
    samples = load_samples(filepath)
    filtered = [s for s in samples if s["id"] != sample_id]
    if len(filtered) == len(samples):
        return False
    with open(filepath, "w") as f:
        for s in filtered:
            f.write(json.dumps(s) + "\n")
    return True


def load_all_samples(data_dir):
    """Load from legacy samples.jsonl and all session files, deduplicated by ID."""
    seen = set()
    all_samples = []

    for s in load_samples(os.path.join(data_dir, "samples.jsonl")):
        if s["id"] not in seen:
            seen.add(s["id"])
            all_samples.append(s)

    sessions_dir = os.path.join(data_dir, "sessions")
    for path in sorted(glob.glob(os.path.join(sessions_dir, "*.jsonl"))):
        for s in load_samples(path):
            if s["id"] not in seen:
                seen.add(s["id"])
                all_samples.append(s)

    return all_samples


def get_stats(filepath):
    """Per-word counts and total duration for a single JSONL file."""
    samples = load_samples(filepath)
    words = {}
    total_duration = 0.0
    for s in samples:
        w = s.get("word", "")
        words[w] = words.get(w, 0) + 1
        total_duration += s.get("duration_s", 0.0)
    return {
        "total_samples": len(samples),
        "words": words,
        "total_duration_s": round(total_duration, 1),
    }


def get_all_stats(data_dir):
    """Stats across all session files + legacy samples.jsonl."""
    samples = load_all_samples(data_dir)
    words = {}
    total_duration = 0.0
    for s in samples:
        w = s.get("word", "")
        words[w] = words.get(w, 0) + 1
        total_duration += s.get("duration_s", 0.0)
    return {
        "total_samples": len(samples),
        "words": words,
        "total_duration_s": round(total_duration, 1),
    }
