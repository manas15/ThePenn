"""
training/finetune.py

Fine-tune the WordClassifier on a new user's handwriting session.

Takes a session JSONL file (from a quick calibration session), applies heavy
data augmentation to multiply the small sample set, and fine-tunes the
existing model — keeping the learned word boundary knowledge while adapting
to the new user's writing style.

Usage:
    python3 -m training.finetune --session training_data/sessions/my_session.jsonl
    python3 -m training.finetune --session training_data/sessions/my_session.jsonl --epochs 30
    python3 -m training.finetune --list-sessions

The segmenter is NOT retrained — word boundary detection generalizes well
across users. Only the classifier (writing style → word identity) is updated.
"""

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data_pipeline import compute_features, augment, trim_idle, collate_word
from .dataset import load_samples
from .model import WordClassifier


# Heavy augmentation: apply multiple augmentations per sample to multiply data
class AugmentedSessionDataset(Dataset):
    """
    Takes a small set of samples and multiplies them via aggressive augmentation.

    Each real sample generates `augment_factor` augmented copies, so 10 samples
    with factor=20 gives 200 training examples.
    """

    def __init__(self, samples, word_to_idx, augment_factor=20, seed=42):
        self.samples = samples
        self.word_to_idx = word_to_idx
        self.augment_factor = augment_factor
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        real_idx = idx // self.augment_factor
        sample = self.samples[real_idx]
        xyz = trim_idle(sample["samples"])

        # Always augment (this is the whole point)
        xyz = augment(xyz, rng=self.rng)

        features = compute_features(xyz)
        features_t = torch.from_numpy(features)
        word_idx = self.word_to_idx[sample["word"].lower()]
        return features_t, word_idx


def list_sessions(data_dir):
    """List available session JSONL files."""
    sessions = sorted(glob.glob(os.path.join(data_dir, "sessions", "*.jsonl")))
    sessions += sorted(glob.glob(os.path.join(data_dir, "sessions", "**", "*.jsonl"), recursive=True))
    seen = set()
    for s in sessions:
        if s not in seen:
            seen.add(s)
            samples = load_samples(s)
            words = {}
            for sample in samples:
                w = sample.get("word", "")
                words[w] = words.get(w, 0) + 1
            word_summary = ", ".join(f"{w}({c})" for w, c in sorted(words.items()))
            print(f"  {s}")
            print(f"    {len(samples)} samples: {word_summary}")
    if not seen:
        print("  No session files found.")


def finetune(args):
    # Load the existing model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    word_to_idx = ckpt["word_to_idx"]
    idx_to_word = ckpt["idx_to_word"]
    num_words = ckpt["num_words"]

    print(f"Base model: {model_path}")
    print(f"Vocabulary ({num_words}): {', '.join(sorted(word_to_idx.keys()))}")
    print(f"Base accuracy: {ckpt.get('val_acc', 'unknown')}")

    # Load session data
    session_path = args.session
    if not os.path.exists(session_path):
        print(f"Error: session file not found: {session_path}")
        sys.exit(1)

    session_samples = load_samples(session_path)
    if not session_samples:
        print(f"Error: no samples in {session_path}")
        sys.exit(1)

    # Filter to known vocabulary only
    valid_samples = [s for s in session_samples if s["word"].lower() in word_to_idx]
    skipped = len(session_samples) - len(valid_samples)
    if skipped:
        unknown = set(s["word"].lower() for s in session_samples) - set(word_to_idx.keys())
        print(f"Warning: skipped {skipped} samples with unknown words: {unknown}")

    if not valid_samples:
        print("Error: no samples match the vocabulary. Record words from the vocabulary.")
        sys.exit(1)

    words_in_session = {}
    for s in valid_samples:
        w = s["word"].lower()
        words_in_session[w] = words_in_session.get(w, 0) + 1

    print(f"\nSession: {session_path}")
    print(f"  {len(valid_samples)} valid samples")
    print(f"  Words: {', '.join(f'{w}({c})' for w, c in sorted(words_in_session.items()))}")
    print(f"  Augmentation factor: {args.augment_factor}x")
    print(f"  Effective training size: {len(valid_samples) * args.augment_factor}")

    # Also load original training data to mix in (prevents catastrophic forgetting)
    original_data_dir = Path(args.original_data)
    original_samples = []
    if original_data_dir.exists():
        from .dataset import load_all_samples
        original_samples = [
            s for s in load_all_samples(str(original_data_dir))
            if s["word"].lower() in word_to_idx
        ]
        print(f"  Original data: {len(original_samples)} samples (mixed in to prevent forgetting)")

    # Build datasets
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # New user data with heavy augmentation
    new_user_ds = AugmentedSessionDataset(
        valid_samples, word_to_idx,
        augment_factor=args.augment_factor,
        seed=args.seed,
    )

    # Original data with standard augmentation (if available)
    if original_samples:
        from .data_pipeline import WordDataset
        # Create a temporary JSONL to feed WordDataset... or just build inline
        original_ds = AugmentedSessionDataset(
            original_samples, word_to_idx,
            augment_factor=2,  # light augmentation on original data
            seed=args.seed + 1,
        )
        # Combine: new user data is weighted more heavily
        combined = torch.utils.data.ConcatDataset([new_user_ds, original_ds])
    else:
        combined = new_user_ds

    # Split 90/10 for train/val
    val_size = max(1, int(len(combined) * 0.1))
    train_size = len(combined) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        combined, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_word, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_word, drop_last=False)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Load model and fine-tune
    model = WordClassifier(num_features=10, num_words=num_words).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Lower learning rate for fine-tuning (don't destroy learned features)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning for {args.epochs} epochs (lr={args.lr})...\n")

    best_val_acc = 0.0
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

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Save fine-tuned model
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": best_val_acc,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "num_words": num_words,
        "finetuned_from": str(model_path),
        "finetuned_on": str(session_path),
        "finetuned_samples": len(valid_samples),
    }, save_path)

    print(f"\nDone! Best val accuracy: {best_val_acc:.3f}")
    print(f"Fine-tuned model saved to: {save_path}")
    print(f"\nTo use it with pennference:")
    print(f"  python3 pennference.py --classifier {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune WordClassifier on a new user's handwriting")
    parser.add_argument("--session", type=str, default="",
                        help="Path to session JSONL file with new user's samples")
    parser.add_argument("--model", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "models" / "word_classifier_best.pt"),
                        help="Path to base model checkpoint")
    parser.add_argument("--original-data", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "training_data"),
                        help="Path to original training data (mixed in to prevent forgetting)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "models" / "word_classifier_finetuned.pt"),
                        help="Output path for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Fine-tuning epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Learning rate (default: 0.0003, lower than full training)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augment-factor", type=int, default=20,
                        help="How many augmented copies per real sample (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-sessions", action="store_true",
                        help="List available session files and exit")
    args = parser.parse_args()

    if args.list_sessions:
        data_dir = str(Path(__file__).resolve().parent.parent / "training_data")
        print("Available sessions:")
        list_sessions(data_dir)
        return

    if not args.session:
        print("Error: --session is required. Use --list-sessions to see available files.")
        sys.exit(1)

    finetune(args)


if __name__ == "__main__":
    main()
