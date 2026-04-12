"""
training/train_ctc.py

Train the CTCRecognizer on collected accelerometer data for
character-level handwriting recognition.

Usage:
    python3 -m training.train_ctc
    python3 -m training.train_ctc --epochs 100 --lr 0.0003
"""

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

from .data_pipeline import CTCDataset, collate_ctc
from .model import CTCRecognizer, IDX_TO_CHAR, BLANK_IDX, decode_ctc


def edit_distance(a, b):
    """Levenshtein edit distance between two strings."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def indices_to_str(indices):
    """Convert a list of character indices back to a string."""
    return "".join(IDX_TO_CHAR.get(i, "") for i in indices)


def train(args):
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: data directory {data_dir} not found")
        sys.exit(1)

    # Load dataset
    dataset = CTCDataset(str(data_dir), augment_data=True)
    print(f"Loaded {len(dataset)} samples")

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_ctc, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_ctc, drop_last=False)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    # CTC loss is not implemented on MPS, so use CPU unless CUDA is available.
    # For this small dataset, CPU is fast enough and avoids device transfer overhead.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CTCRecognizer(num_features=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    # Training loop
    best_val_cer = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # Train
        model.train()
        train_loss = 0.0
        train_n = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]",
                         leave=False, unit="batch")
        for features, targets, input_lengths, target_lengths in train_bar:
            features = features.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            log_probs, output_lengths = model(features, input_lengths)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_size = features.size(0)
            train_loss += loss.item() * batch_size
            train_n += batch_size
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= train_n

        # Validate — compute CER
        model.eval()
        val_loss = 0.0
        val_n = 0
        total_edits = 0
        total_chars = 0
        examples = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [val]",
                           leave=False, unit="batch")
            for features, targets, input_lengths, target_lengths in val_bar:
                features = features.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                log_probs, output_lengths = model(features, input_lengths)
                loss = criterion(log_probs.cpu(), targets.cpu(),
                                 output_lengths.cpu(), target_lengths.cpu())

                batch_size = features.size(0)
                val_loss += loss.item() * batch_size
                val_n += batch_size

                # Decode predictions and compute CER
                pred_indices = log_probs.argmax(dim=2).transpose(0, 1)  # (batch, time)

                # Reconstruct per-sample targets from flat tensor
                offset = 0
                for i in range(batch_size):
                    tlen = target_lengths[i].item()
                    true_indices = targets[offset : offset + tlen].cpu().tolist()
                    offset += tlen

                    pred_text = decode_ctc(pred_indices[i].cpu().tolist())
                    true_text = indices_to_str(true_indices)

                    ed = edit_distance(pred_text, true_text)
                    total_edits += ed
                    total_chars += max(len(true_text), 1)

                    if len(examples) < 5:
                        examples.append((true_text, pred_text))

        val_loss /= val_n
        val_cer = total_edits / total_chars if total_chars > 0 else 1.0
        scheduler.step(val_loss)

        # Update epoch progress bar with metrics
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            cer=f"{val_cer:.3f}",
            best_cer=f"{best_val_cer:.3f}",
        )

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            tqdm.write(f"Epoch {epoch:3d}/{args.epochs}  "
                       f"train_loss={train_loss:.4f}  "
                       f"val_loss={val_loss:.4f}  val_cer={val_cer:.3f}  "
                       f"lr={lr:.6f}")
            # Show example predictions
            for true_text, pred_text in examples[:3]:
                marker = "+" if true_text == pred_text else "x"
                tqdm.write(f"    {marker} \"{true_text}\" -> \"{pred_text}\"")

        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            save_path = save_dir / "ctc_recognizer_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer": val_cer,
            }, save_path)

    print(f"\nBest val CER: {best_val_cer:.3f}")
    print(f"Model saved to {save_dir / 'ctc_recognizer_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train ghost-writer CTC recognizer")
    parser.add_argument("--data", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "training_data"),
                        help="Path to training_data/ directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "models"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
