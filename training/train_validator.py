"""
training/train_validator.py

Train the WordValidator — binary classifier: is this a word or not?

Positive samples: existing JSONL word samples (real words)
Negative samples: gap segments from session CSVs (writing=0 periods)

Usage:
    python3 -m training.train_validator
    python3 -m training.train_validator --epochs 30
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from .data_pipeline import (
    compute_features, trim_idle, augment, find_session_csvs
)
from .dataset import load_all_samples
from .model import WordValidator


class ValidatorDataset(Dataset):
    """
    Binary dataset: positive = real word samples, negative = gap segments.
    Each item: (features_tensor, label) where label is 1.0 (word) or 0.0 (not word).
    """

    def __init__(self, data_dir, augment_data=False, seed=42,
                 min_gap_samples=20, max_gap_samples=150):
        self.augment_data = augment_data
        self.rng = random.Random(seed)
        self.samples = []  # (samples_xyz, label)

        # Positive samples from JSONL
        positives = load_all_samples(data_dir)
        for s in positives:
            if len(s["samples"]) >= 10:
                self.samples.append((s["samples"], 1.0))

        # Negative samples from session CSV gaps
        csv_paths = find_session_csvs(data_dir)
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            xyz = df[["x_g", "y_g", "z_g"]].values
            writing = df["writing"].values

            # Extract contiguous gap segments
            in_gap = False
            gap_start = 0
            for i in range(len(writing)):
                if writing[i] == 0 and not in_gap:
                    gap_start = i
                    in_gap = True
                elif writing[i] == 1 and in_gap:
                    gap_len = i - gap_start
                    if min_gap_samples <= gap_len <= max_gap_samples:
                        segment = xyz[gap_start:i].tolist()
                        self.samples.append((segment, 0.0))
                    in_gap = False
            # Handle trailing gap
            if in_gap:
                gap_len = len(writing) - gap_start
                if min_gap_samples <= gap_len <= max_gap_samples:
                    segment = xyz[gap_start:].tolist()
                    self.samples.append((segment, 0.0))

        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = sum(1 for _, l in self.samples if l == 0.0)
        print(f"Validator dataset: {n_pos} positive, {n_neg} negative, {len(self.samples)} total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xyz, label = self.samples[idx]

        if label == 1.0:
            xyz = trim_idle(xyz)
        if self.augment_data:
            xyz = augment(xyz, rng=self.rng)

        features = compute_features(xyz)
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)


def collate_validator(batch):
    features_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
    labels = torch.stack(labels)
    return padded, labels, lengths


def train(args):
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: data directory {data_dir} not found")
        sys.exit(1)

    dataset = ValidatorDataset(str(data_dir), augment_data=True, seed=args.seed)

    if len(dataset) < 10:
        print("Error: need at least 10 samples to train")
        sys.exit(1)

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_validator, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_validator, drop_last=False)

    print(f"Train: {train_size}, Val: {val_size}")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WordValidator(num_features=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels, lengths in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(features, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_loss += loss.item() * labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                logits = model(features, lengths)
                loss = criterion(logits, labels)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
                  f"lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / "word_validator_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, save_path)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {save_dir / 'word_validator_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train word validator (is this a word?)")
    parser.add_argument("--data", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "training_data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "models"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
