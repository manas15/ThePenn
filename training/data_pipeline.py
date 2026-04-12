"""
training/data_pipeline.py

PyTorch Dataset and feature engineering for ghost-writer.
Loads JSONL samples, computes 10-dim features, handles padding & augmentation.
Also provides SegmentationDataset for continuous session CSV streams.
"""

import glob
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import load_samples, load_all_samples


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def trim_idle(samples_xyz, margin=3, min_len=20):
    """
    Trim leading and trailing idle from raw [[x,y,z], ...] data.

    Compares energy in a sliding window against the peak writing energy
    (middle of the sample) to find where actual writing begins and ends.
    """
    arr = np.array(samples_xyz, dtype=np.float32)
    if len(arr) < min_len:
        return samples_xyz

    deltas = np.diff(arr, axis=0)
    energy = np.sqrt((deltas ** 2).sum(axis=1))

    # Smooth energy with rolling window of 5
    kernel = 5
    if len(energy) >= kernel:
        smooth = np.convolve(energy, np.ones(kernel) / kernel, mode='valid')
    else:
        return samples_xyz

    # Reference: peak energy in middle third (where writing is happening)
    mid_start = len(smooth) // 3
    mid_end = 2 * len(smooth) // 3
    if mid_start >= mid_end:
        return samples_xyz
    peak_energy = np.percentile(smooth[mid_start:mid_end], 75)

    # Threshold: 30% of peak writing energy
    threshold = peak_energy * 0.30
    if threshold < 0.05:
        return samples_xyz

    # Find first point where smoothed energy exceeds threshold (leading trim only)
    # Trailing idle is already handled by the segmenter's gap detection
    start = 0
    for i in range(len(smooth)):
        if smooth[i] > threshold:
            start = max(0, i - margin)
            break

    if start < 3:
        return samples_xyz  # nothing meaningful to trim

    if len(arr) - start < min_len:
        return samples_xyz

    return arr[start:].tolist()


def compute_features(samples_xyz):
    """
    Convert raw [[x,y,z], ...] to a (T, 10) feature matrix.

    Features per timestep:
        0-2:  raw x, y, z
        3-5:  first-order deltas  dx, dy, dz
        6:    L2 norm of first-order delta
        7-9:  second-order deltas ddx, ddy, ddz
    """
    arr = np.array(samples_xyz, dtype=np.float32)  # (T, 3)
    T = len(arr)

    # First-order deltas
    deltas = np.zeros_like(arr)
    if T > 1:
        deltas[1:] = arr[1:] - arr[:-1]

    # L2 norm of deltas
    l2 = np.sqrt((deltas ** 2).sum(axis=1, keepdims=True))  # (T, 1)

    # Second-order deltas (jerk)
    ddeltas = np.zeros_like(arr)
    if T > 2:
        ddeltas[2:] = deltas[2:] - deltas[1:-1]

    features = np.concatenate([arr, deltas, l2, ddeltas], axis=1)  # (T, 10)
    return features


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment(samples_xyz, rng=None):
    """
    Apply random augmentations to raw [[x,y,z], ...] data.
    Returns augmented copy. Does NOT modify the input.
    """
    if rng is None:
        rng = random.Random()

    arr = np.array(samples_xyz, dtype=np.float32)  # (T, 3)

    # 1. Time warping: resample at 0.8x–1.2x speed
    if rng.random() < 0.5:
        factor = rng.uniform(0.8, 1.2)
        new_len = max(3, int(len(arr) * factor))
        old_t = np.linspace(0, 1, len(arr))
        new_t = np.linspace(0, 1, new_len)
        arr = np.stack([np.interp(new_t, old_t, arr[:, i]) for i in range(3)], axis=1)

    # 2. Gaussian noise (sigma ~0.005g, sensor-level)
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 0.005, arr.shape)
        arr = arr + noise.astype(np.float32)

    # 3. Amplitude scaling (0.9–1.1x)
    if rng.random() < 0.5:
        scale = rng.uniform(0.9, 1.1)
        arr = arr * scale

    # 4. Random rotation around Z-axis (5–15°)
    if rng.random() < 0.5:
        angle = math.radians(rng.uniform(-15, 15))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = arr.copy()
        rotated[:, 0] = arr[:, 0] * cos_a - arr[:, 1] * sin_a
        rotated[:, 1] = arr[:, 0] * sin_a + arr[:, 1] * cos_a
        arr = rotated

    # 5. Time shift: trim 0–10% from start and end
    if rng.random() < 0.5 and len(arr) > 6:
        max_trim = max(1, len(arr) // 10)
        trim_start = rng.randint(0, max_trim)
        trim_end = rng.randint(0, max_trim)
        end_idx = len(arr) - trim_end
        if end_idx - trim_start >= 3:
            arr = arr[trim_start:end_idx]

    return arr.tolist()


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

def _load(path):
    """Load samples from a single JSONL file or a training_data/ directory."""
    import os
    if os.path.isdir(path):
        return load_all_samples(path)
    return load_samples(path)


class WordDataset(Dataset):
    """
    Phase 1: word-level classification dataset.

    Each item: (features_tensor, word_index)

    ``path`` can be a single .jsonl file or the training_data/ directory
    (loads all session files + legacy samples.jsonl).
    """

    def __init__(self, path, word_to_idx=None, augment_data=False, seed=42):
        raw_samples = _load(path)
        if not raw_samples:
            raise ValueError(f"No samples found in {path}")

        # Build vocabulary from data if not provided
        all_words = sorted(set(s["word"].lower() for s in raw_samples))
        if word_to_idx is None:
            self.word_to_idx = {w: i for i, w in enumerate(all_words)}
        else:
            self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.num_words = len(self.word_to_idx)

        # Filter to known words only
        self.samples = [
            s for s in raw_samples
            if s["word"].lower() in self.word_to_idx
        ]
        self.augment_data = augment_data
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        xyz = trim_idle(sample["samples"])

        if self.augment_data:
            xyz = augment(xyz, rng=self.rng)

        features = compute_features(xyz)
        features_t = torch.from_numpy(features)  # (T, 10)
        word_idx = self.word_to_idx[sample["word"].lower()]
        return features_t, word_idx


class CTCDataset(Dataset):
    """
    Phase 2: character-level CTC dataset.

    Each item: (features_tensor, target_indices, target_length)
    """

    def __init__(self, path, augment_data=False, seed=42):
        from .model import encode_text

        raw_samples = _load(path)
        if not raw_samples:
            raise ValueError(f"No samples found in {path}")

        self.samples = raw_samples
        self.augment_data = augment_data
        self.rng = random.Random(seed)
        self.encode_text = encode_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        xyz = trim_idle(sample["samples"])

        if self.augment_data:
            xyz = augment(xyz, rng=self.rng)

        features = compute_features(xyz)
        features_t = torch.from_numpy(features)  # (T, 10)
        target = torch.tensor(self.encode_text(sample["word"]), dtype=torch.long)
        return features_t, target


# ---------------------------------------------------------------------------
# Collation (padding variable-length sequences)
# ---------------------------------------------------------------------------

def collate_word(batch):
    """Collate for WordDataset: pad features, stack labels."""
    features_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, lengths


def collate_ctc(batch):
    """Collate for CTCDataset: pad features and targets separately."""
    features_list, target_list = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in target_list], dtype=torch.long)
    padded_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
    targets = torch.cat(target_list)  # CTC expects flat target tensor
    return padded_features, targets, lengths, target_lengths


# ---------------------------------------------------------------------------
# Segmentation dataset — continuous session CSVs
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Loads session CSVs and slices them into overlapping windows for
    per-sample writing/gap classification.

    Each item: (features, labels)
      features: (window_size, 3) float32 — raw x, y, z
      labels:   (window_size,)  float32 — 1.0 = writing, 0.0 = gap
    """

    def __init__(self, csv_paths, window_size=128, stride=32, augment_data=False):
        self.windows = []
        self.augment_data = augment_data

        for p in csv_paths:
            df = pd.read_csv(p)
            xyz = df[["x_g", "y_g", "z_g"]].values.astype(np.float32)
            labels = df["writing"].values.astype(np.float32)
            for start in range(0, len(df) - window_size + 1, stride):
                self.windows.append((
                    xyz[start : start + window_size],
                    labels[start : start + window_size],
                ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        xyz, labels = self.windows[idx]
        xyz = xyz.copy()

        if self.augment_data:
            # Light noise + small gain jitter
            xyz += np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
            xyz *= np.random.uniform(0.95, 1.05)

        return torch.from_numpy(xyz), torch.from_numpy(labels)


def find_session_csvs(data_dir):
    """Return sorted list of session CSV paths in a training_data directory."""
    sessions_dir = os.path.join(data_dir, "sessions")
    return sorted(glob.glob(os.path.join(sessions_dir, "session_*.csv")))
