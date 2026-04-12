"""
ml/data_pipeline.py

PyTorch datasets and collation for training.
"""

import os
import random

import torch
from torch.utils.data import Dataset

from server.dataset import load_samples, load_all_samples
from .features import compute_features, augment


def _load(path):
    """Load samples from a JSONL file or a data/ directory."""
    if os.path.isdir(path):
        return load_all_samples(path)
    return load_samples(path)


class WordDataset(Dataset):
    """
    Phase 1: word-level classification.
    Each item: (features_tensor, word_index)
    """

    def __init__(self, path, word_to_idx=None, augment_data=False, seed=42):
        raw = _load(path)
        if not raw:
            raise ValueError(f"No samples found in {path}")

        all_words = sorted(set(s["word"].lower() for s in raw))
        if word_to_idx is None:
            self.word_to_idx = {w: i for i, w in enumerate(all_words)}
        else:
            self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.num_words = len(self.word_to_idx)

        self.samples = [s for s in raw if s["word"].lower() in self.word_to_idx]
        self.augment_data = augment_data
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        xyz = sample["samples"]
        if self.augment_data:
            xyz = augment(xyz, rng=self.rng)
        features = compute_features(xyz)
        return torch.from_numpy(features), self.word_to_idx[sample["word"].lower()]


class CTCDataset(Dataset):
    """
    Phase 2: character-level CTC.
    Each item: (features_tensor, target_indices)
    """

    def __init__(self, path, augment_data=False, seed=42):
        from .model import encode_text
        raw = _load(path)
        if not raw:
            raise ValueError(f"No samples found in {path}")
        self.samples = raw
        self.augment_data = augment_data
        self.rng = random.Random(seed)
        self.encode_text = encode_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        xyz = sample["samples"]
        if self.augment_data:
            xyz = augment(xyz, rng=self.rng)
        features = compute_features(xyz)
        target = torch.tensor(self.encode_text(sample["word"]), dtype=torch.long)
        return torch.from_numpy(features), target


def collate_word(batch):
    """Pad features, stack labels."""
    features_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, lengths


def collate_ctc(batch):
    """Pad features, flatten targets for CTC loss."""
    features_list, target_list = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in target_list], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
    targets = torch.cat(target_list)
    return padded, targets, lengths, target_lengths
