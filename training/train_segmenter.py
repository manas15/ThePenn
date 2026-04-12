"""
Train a SegmentationTCN to detect writing vs. between-word gaps in a
continuous accelerometer stream.

Usage:
    python3 -m training.train_segmenter [--epochs 80] [--lr 1e-3]

Trains on all session CSVs except the last (longest), which is held out
for validation. Prints per-epoch loss + val metrics and saves the best
model to segmenter.pt. After training, produces segmenter_eval.png with
the val session's prediction overlay + segment-level scores.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import SegmentationTCN
from .data_pipeline import SegmentationDataset, find_session_csvs

DATA_DIR = Path(__file__).parent.parent / "training_data"
MODEL_PATH = Path(__file__).parent.parent / "segmenter.pt"
PLOT_PATH = Path(__file__).parent.parent / "segmenter_eval.png"

WINDOW = 128   # ~2.5s at 50Hz
STRIDE = 16    # ~0.3s overlap


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Segment-level evaluation
# ---------------------------------------------------------------------------

def extract_gap_spans(binary, t):
    """Return list of (start_s, end_s) for contiguous gap (==0) regions."""
    spans = []
    i = 0
    while i < len(binary):
        if binary[i] == 0:
            j = i
            while j < len(binary) and binary[j] == 0:
                j += 1
            spans.append((t[i], t[min(j - 1, len(t) - 1)]))
            i = j
        else:
            i += 1
    return spans


def match_events(pred, truth, tol=0.35):
    pred_c = [(s + e) / 2 for s, e in pred]
    truth_c = [(s + e) / 2 for s, e in truth]
    used = set()
    tp = 0
    for pc in pred_c:
        best_j, best_d = -1, tol + 1
        for j, tc in enumerate(truth_c):
            if j in used:
                continue
            d = abs(pc - tc)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= tol:
            used.add(best_j)
            tp += 1
    fp = len(pred) - tp
    fn = len(truth) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Inference on a full session (no windowing artifacts)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_full_session(model, csv_path, device):
    """Run the model over an entire session CSV. Returns (probs, labels, elapsed_s)."""
    df = pd.read_csv(csv_path)
    xyz = torch.from_numpy(
        df[["x_g", "y_g", "z_g"]].values.astype(np.float32)
    ).unsqueeze(0).to(device)  # (1, T, 3)

    model.eval()
    logits = model(xyz)  # (1, T)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    labels = df["writing"].values
    elapsed = df["elapsed_s"].values
    return probs, labels, elapsed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"device: {device}")

    csvs = find_session_csvs(str(DATA_DIR))
    if len(csvs) < 2:
        raise SystemExit(f"Need at least 2 session CSVs, found {len(csvs)}")

    # Hold out the longest session for validation
    lengths = [len(pd.read_csv(p)) for p in csvs]
    val_idx = int(np.argmax(lengths))
    val_csv = csvs[val_idx]
    train_csvs = [c for i, c in enumerate(csvs) if i != val_idx]
    print(f"train: {[Path(c).name for c in train_csvs]}")
    print(f"val:   {Path(val_csv).name}")

    train_ds = SegmentationDataset(train_csvs, WINDOW, STRIDE, augment_data=True)
    val_ds = SegmentationDataset([val_csv], WINDOW, STRIDE, augment_data=False)
    print(f"train windows: {len(train_ds)},  val windows: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Class weight: gaps are ~20% of samples, so upweight them
    gap_frac = sum(
        (1 - w[1]).sum().item() for w in train_ds.windows
    ) / sum(len(w[1]) for w in train_ds.windows)
    pos_weight = torch.tensor([gap_frac / (1 - gap_frac)]).to(device)
    print(f"gap fraction: {gap_frac:.2f}, pos_weight (for writing=1): {pos_weight.item():.2f}")
    # We want to upweight gap=0, but BCEWithLogitsLoss pos_weight scales the
    # loss for target=1 samples. So invert: we want gaps (target=0) upweighted.
    # Instead, manually weight in the loss.

    model = SegmentationTCN(in_channels=3, hidden=64, kernel_size=3,
                            num_blocks=5, dropout=0.15).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5
    )

    # Weight tensor: higher weight for gap samples
    w_gap = (1 - gap_frac) / gap_frac  # e.g. 0.8/0.2 = 4.0
    print(f"loss weight for gap samples: {w_gap:.1f}x")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_batches = 0
        for feats, labels in train_dl:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            # Per-sample weights: gap samples get w_gap, writing samples get 1.0
            weight = torch.where(labels == 0, w_gap, 1.0)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, weight=weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for feats, labels in val_dl:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                weight = torch.where(labels == 0, w_gap, 1.0)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, weight=weight
                )
                val_loss += loss.item() * feats.size(0)
                val_n += feats.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu().long())

        val_loss /= max(val_n, 1)
        scheduler.step(val_loss)

        # Per-sample accuracy on gap class
        preds_cat = torch.cat(all_preds).flatten()
        labels_cat = torch.cat(all_labels).flatten()
        gap_mask = labels_cat == 0
        gap_acc = (preds_cat[gap_mask] == 0).float().mean().item() if gap_mask.any() else 0
        wri_acc = (preds_cat[~gap_mask] == 1).float().mean().item() if (~gap_mask).any() else 0

        if epoch % 5 == 1 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  train_loss={train_loss/n_batches:.4f}  "
                  f"val_loss={val_loss:.4f}  gap_acc={gap_acc:.3f}  wri_acc={wri_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nbest val_loss: {best_val_loss:.4f}")
    print(f"saved model to {MODEL_PATH}")

    # -- Full-session eval + plot --
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    evaluate_and_plot(model, val_csv, device)


def evaluate_and_plot(model, csv_path, device):
    import matplotlib.pyplot as plt

    probs, labels, elapsed = predict_full_session(model, csv_path, device)
    pred_binary = (probs > 0.5).astype(int)

    # Segment-level F1
    dt = np.median(np.diff(elapsed))
    pred_gaps = extract_gap_spans(pred_binary, elapsed)
    true_gaps = extract_gap_spans(labels, elapsed)
    # Filter tiny predicted gaps (< 0.1s) as noise
    pred_gaps = [(s, e) for s, e in pred_gaps if e - s >= 0.1]

    tp, fp, fn = match_events(pred_gaps, true_gaps, tol=0.35)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nsegment-level eval on {Path(csv_path).name}:")
    print(f"  true gaps: {len(true_gaps)}, predicted gaps: {len(pred_gaps)}")
    print(f"  TP={tp} FP={fp} FN={fn}  P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    # Plot
    df = pd.read_csv(csv_path)

    # Extract word labels and their time spans for annotation
    word_spans = []
    i = 0
    words = df["word"].values
    writing = df["writing"].values
    while i < len(words):
        if writing[i] == 1 and isinstance(words[i], str) and words[i]:
            w = words[i]
            j = i
            while j < len(words) and writing[j] == 1 and words[j] == w:
                j += 1
            word_spans.append((w, elapsed[i], elapsed[min(j - 1, len(elapsed) - 1)]))
            i = j
        else:
            i += 1

    # Find predicted writing→gap transitions (word ends) and true ones
    pred_word_ends = []
    for k in range(1, len(pred_binary)):
        if pred_binary[k - 1] == 1 and pred_binary[k] == 0:
            pred_word_ends.append(elapsed[k])
    true_word_ends = []
    for k in range(1, len(labels)):
        if labels[k - 1] == 1 and labels[k] == 0:
            true_word_ends.append(elapsed[k])

    fig, axes = plt.subplots(3, 1, figsize=(16, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 0.7, 0.7]})

    # Row 1: raw signal with word labels
    ax = axes[0]
    for s, e in true_gaps:
        ax.axvspan(s, e, color="#ffcc66", alpha=0.3, lw=0)
    ax.plot(elapsed, df.x_g, lw=0.5, color="#d62728", label="x")
    ax.plot(elapsed, df.y_g, lw=0.5, color="#2ca02c", label="y")
    ax.plot(elapsed, df.z_g, lw=0.5, color="#1f77b4", label="z")
    for w, ws, we in word_spans:
        ax.text((ws + we) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] else 1.5,
                w, ha="center", va="bottom", fontsize=6.5, color="#333",
                fontstyle="italic")
    ax.set_ylabel("accel (g)")
    ax.set_title(f"{Path(csv_path).name}")
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    # Row 2: P(writing) with true word-end markers
    ax = axes[1]
    ax.fill_between(elapsed, 0, labels, alpha=0.25, color="#ffcc66", label="true writing")
    ax.plot(elapsed, probs, lw=0.8, color="#1f77b4", label="P(writing)")
    ax.axhline(0.5, color="#aaa", ls=":", lw=0.7)
    for t in true_word_ends:
        ax.axvline(t, color="#e6550d", ls="-", lw=0.8, alpha=0.7)
    for t in pred_word_ends:
        ax.axvline(t, color="#31a354", ls="--", lw=0.8, alpha=0.7)
    ax.plot([], [], color="#e6550d", lw=1, label="true word end")
    ax.plot([], [], color="#31a354", ls="--", lw=1, label="predicted word end")
    ax.set_ylabel("P(writing)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7, ncol=4)

    # Row 3: segment comparison
    ax = axes[2]
    for s, e in true_gaps:
        ax.axvspan(s, e, color="#ffcc66", alpha=0.3, lw=0)
    for s, e in pred_gaps:
        ax.axvspan(s, e, color="#44cc44", alpha=0.3, lw=0)
    ax.set_ylabel("segments")
    ax.set_xlabel("time (s)")
    ax.set_title(f"yellow=true gaps, green=predicted  |  P={prec:.2f} R={rec:.2f} F1={f1:.2f}",
                 fontsize=9)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    print(f"wrote {PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
