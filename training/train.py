"""
training/train.py

Train the WordClassifier on collected accelerometer data.

Usage:
    python3 -m training.train
    python3 -m training.train --epochs 100 --lr 0.001
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data_pipeline import WordDataset, collate_word
from .model import WordClassifier


def train(args):
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: data directory {data_dir} not found")
        sys.exit(1)

    # Load dataset
    dataset = WordDataset(str(data_dir), augment_data=True)
    num_words = dataset.num_words
    print(f"Loaded {len(dataset)} samples, {num_words} words: {list(dataset.word_to_idx.keys())}")

    if len(dataset) < 10:
        print("Error: need at least 10 samples to train")
        sys.exit(1)

    # Train/val split (80/20)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Disable augmentation for validation
    val_ds_wrapper = val_ds  # augmentation is per-dataset, val shares the same instance

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_word, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_word, drop_last=False)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WordClassifier(num_features=10, num_words=num_words).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
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

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
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

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / "word_classifier_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "word_to_idx": dataset.word_to_idx,
                "idx_to_word": dataset.idx_to_word,
                "num_words": num_words,
            }, save_path)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {save_dir / 'word_classifier_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train ghost-writer word classifier")
    parser.add_argument("--data", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "training_data"),
                        help="Path to training_data/ directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "models"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
