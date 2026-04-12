"""
Analyze session CSVs to detect inter-word gaps in the accelerometer stream.

Ground truth lives in each session CSV's `writing` column:
  writing == 1  -> pen is inscribing a word
  writing == 0  -> pen is in transit between words (the "gap")

Two framings are useful and we report both:

1. Per-sample classification — can we label each sample writing/gap?
   Honest ceiling: AUC ~0.72. Stroke peaks inside words look a lot like the
   ballistic lift-and-move at the edges of a gap, so sample-level separability
   is genuinely limited. The ROC here tells us where that ceiling is.

2. Segment / event detection — can we detect each gap as one event?
   This is what you actually want for streaming segmentation. A smoothed
   accel-magnitude feature with Schmitt-trigger hysteresis + a minimum-duration
   filter hits much higher segment-level F1, because short in-word spikes are
   rejected by the min-duration rule and the low threshold keeps us in "gap"
   state once we're in it.

Output:
  word_boundary_analysis.png  — feature trace with labels, ROC, and segment F1
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SESSIONS_DIR = Path(__file__).parent / "training_data" / "sessions"
OUT_PATH = Path(__file__).parent / "word_boundary_analysis.png"

# ~0.5s at 50Hz smoothing window — long enough to average stroke jitter,
# short enough to preserve ~0.3s gap events.
SMOOTH_WINDOW = 25

# Schmitt-trigger hysteresis thresholds on the smoothed |a|. Enter "gap"
# state only when the feature climbs above HI; leave only when it drops
# back below LO. The gap between HI and LO prevents chatter near the edge.
HI_THRESHOLD = 0.30
LO_THRESHOLD = 0.22

# Predicted gaps shorter than this are treated as in-word jitter and dropped.
MIN_GAP_DURATION_S = 0.15

# A predicted gap "matches" a true gap if their centers are within this
# tolerance. ~1 sample window is too tight; ~1 word is too loose.
MATCH_TOLERANCE_S = 0.30


# ---------------------------------------------------------------------------
# Loading + feature extraction
# ---------------------------------------------------------------------------

def load_session(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Subtract per-session mean to remove gravity + sensor bias.
    for c in ("x_g", "y_g", "z_g"):
        df[c] = df[c] - df[c].mean()
    df["ac_norm"] = np.sqrt(df.x_g**2 + df.y_g**2 + df.z_g**2)
    df["feature"] = (
        df.ac_norm.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Per-sample ROC
# ---------------------------------------------------------------------------

def roc(scores: np.ndarray, labels: np.ndarray, n_points: int = 200):
    """Return (fpr, tpr, thresholds) by sweeping threshold high -> low."""
    order = np.argsort(-scores)
    s = scores[order]
    l = labels[order]
    P = l.sum()
    N = len(l) - P
    tp = np.cumsum(l == 1)
    fp = np.cumsum(l == 0)
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    idx = np.linspace(0, len(s) - 1, n_points).astype(int)
    return fpr[idx], tpr[idx], s[idx]


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


# ---------------------------------------------------------------------------
# Segment detection with Schmitt-trigger hysteresis + min duration
# ---------------------------------------------------------------------------

def detect_gaps(feature: np.ndarray, t: np.ndarray,
                hi: float, lo: float, min_dur_s: float) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) for predicted between-word gaps."""
    in_gap = False
    start_i = 0
    spans = []
    for i, v in enumerate(feature):
        if not in_gap and v >= hi:
            in_gap = True
            start_i = i
        elif in_gap and v < lo:
            in_gap = False
            spans.append((start_i, i))
    if in_gap:
        spans.append((start_i, len(feature) - 1))

    out = []
    for s, e in spans:
        if t[e] - t[s] >= min_dur_s:
            out.append((t[s], t[e]))
    return out


def true_gaps(df: pd.DataFrame) -> list[tuple[float, float]]:
    flag = (df.writing == 0).values
    t = df.elapsed_s.values
    spans = []
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j < len(flag) and flag[j]:
                j += 1
            spans.append((t[i], t[min(j, len(t) - 1)]))
            i = j
        else:
            i += 1
    return spans


def match_events(pred, truth, tol):
    """Greedy 1:1 matching by span-center distance. Returns (tp, fp, fn)."""
    pred_c = [(p0 + p1) / 2 for p0, p1 in pred]
    truth_c = [(t0 + t1) / 2 for t0, t1 in truth]
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
# Main
# ---------------------------------------------------------------------------

def main():
    csvs = sorted(SESSIONS_DIR.glob("session_*.csv"))
    if not csvs:
        raise SystemExit(f"No session CSVs in {SESSIONS_DIR}")
    sessions = [(p.name, load_session(p)) for p in csvs]

    # ---- Per-sample ROC, pooled over all sessions --------------------------
    all_scores = np.concatenate([df.feature.values for _, df in sessions])
    all_labels = np.concatenate(
        [(df.writing == 0).astype(int).values for _, df in sessions]
    )
    fpr, tpr, thr = roc(all_scores, all_labels)
    auc_val = auc(fpr, tpr)
    j = tpr - fpr
    best = int(np.argmax(j))

    # ---- Segment-level detection + F1 per session --------------------------
    print(f"Segment-level detection  (hi={HI_THRESHOLD}, lo={LO_THRESHOLD}, "
          f"min_dur={MIN_GAP_DURATION_S}s, tol={MATCH_TOLERANCE_S}s)")
    seg_rows = []
    for name, df in sessions:
        pred = detect_gaps(df.feature.values, df.elapsed_s.values,
                           HI_THRESHOLD, LO_THRESHOLD, MIN_GAP_DURATION_S)
        truth = true_gaps(df)
        tp, fp, fn = match_events(pred, truth, MATCH_TOLERANCE_S)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        seg_rows.append((name, len(truth), len(pred), tp, fp, fn, prec, rec, f1))
        print(f"  {name}  truth={len(truth):3d}  pred={len(pred):3d}  "
              f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
              f"P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    # ---- Plot: longest session's trace + ROC + F1 summary ------------------
    name, df = max(sessions, key=lambda s: len(s[1]))
    pred = detect_gaps(df.feature.values, df.elapsed_s.values,
                       HI_THRESHOLD, LO_THRESHOLD, MIN_GAP_DURATION_S)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.45, wspace=0.3)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_feat = fig.add_subplot(gs[1, :], sharex=ax_raw)
    ax_roc = fig.add_subplot(gs[2, 0])
    ax_tab = fig.add_subplot(gs[2, 1])

    _shade_gaps(ax_raw, df, color="#ffcc66", alpha=0.35, label="true gap")
    ax_raw.plot(df.elapsed_s, df.ac_norm, lw=0.6, color="#333")
    ax_raw.set_ylabel("|a|  (g, mean-removed)")
    ax_raw.set_title(f"{name}  —  raw AC magnitude (true gaps shaded)")
    ax_raw.set_xlim(df.elapsed_s.min(), df.elapsed_s.max())

    _shade_gaps(ax_feat, df, color="#ffcc66", alpha=0.25)
    for s, e in pred:
        ax_feat.axvspan(s, e, color="#44cc44", alpha=0.25, lw=0)
    ax_feat.plot(df.elapsed_s, df.feature, lw=0.9, color="#1f77b4")
    ax_feat.axhline(HI_THRESHOLD, color="red", ls="--", lw=1, label=f"hi={HI_THRESHOLD}")
    ax_feat.axhline(LO_THRESHOLD, color="red", ls=":", lw=1, label=f"lo={LO_THRESHOLD}")
    ax_feat.set_ylabel(f"smoothed |a|  (w={SMOOTH_WINDOW})")
    ax_feat.set_xlabel("time (s)")
    ax_feat.set_title("detector feature — yellow = true gap, green = predicted gap")
    ax_feat.legend(loc="upper right", fontsize=8)

    ax_roc.plot(fpr, tpr, color="#1f77b4", lw=1.5, label=f"AUC = {auc_val:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="#aaa", ls=":", lw=1, label="chance")
    ax_roc.scatter([fpr[best]], [tpr[best]], color="red", zorder=5,
                   label=f"Youden J: TPR={tpr[best]:.2f} FPR={fpr[best]:.2f}")
    ax_roc.set_xlabel("FPR  (writing samples flagged as gap)")
    ax_roc.set_ylabel("TPR  (gaps correctly flagged)")
    ax_roc.set_title("per-sample ROC (all sessions pooled)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.set_aspect("equal")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)

    ax_tab.axis("off")
    headers = ["session", "truth", "pred", "TP", "FP", "FN", "P", "R", "F1"]
    cell_rows = []
    for r in seg_rows:
        short = r[0].replace("session_", "").replace(".csv", "")[:14]
        cell_rows.append([short, r[1], r[2], r[3], r[4], r[5],
                          f"{r[6]:.2f}", f"{r[7]:.2f}", f"{r[8]:.2f}"])
    tbl = ax_tab.table(cellText=cell_rows, colLabels=headers,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax_tab.set_title("segment-level detection per session", pad=20)

    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT_PATH}")
    print(f"pooled per-sample AUC = {auc_val:.3f}")


def _shade_gaps(ax, df: pd.DataFrame, color="#ffcc66", alpha=0.35, label=None):
    in_gap = (df.writing == 0).values
    t = df.elapsed_s.values
    first = True
    i = 0
    while i < len(in_gap):
        if in_gap[i]:
            j = i
            while j < len(in_gap) and in_gap[j]:
                j += 1
            ax.axvspan(t[i], t[min(j, len(t) - 1)],
                       color=color, alpha=alpha, lw=0,
                       label=label if first else None)
            first = False
            i = j
        else:
            i += 1


if __name__ == "__main__":
    main()
