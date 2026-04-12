"""
Exploratory visualization of session CSVs. Plots per-session raw x/y/z
traces with true gap regions shaded, plus a zoom on a single word boundary
so you can eyeball what separates "in word" from "between words".

Outputs:
  session_overview.png  — one row per session, full timelines
  gap_zoom.png          — zoomed view of one pen lift / move / plant
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SESSIONS_DIR = Path(__file__).parent / "training_data" / "sessions"
OVERVIEW_PATH = Path(__file__).parent / "session_overview.png"
ZOOM_PATH = Path(__file__).parent / "gap_zoom.png"


LP_ALPHA = 0.08  # ~2Hz cutoff at 50Hz sample rate


def load(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    dx, dy, dz = df.x_g.diff().fillna(0), df.y_g.diff().fillna(0), df.z_g.diff().fillna(0)
    df["jerk"] = np.sqrt(dx**2 + dy**2 + dz**2)

    # Separate gravity (orientation) from dynamic acceleration via exponential
    # moving average low-pass filter. alpha ~0.08 ≈ 2Hz cutoff at 50Hz.
    for axis in ("x_g", "y_g", "z_g"):
        grav = np.zeros(len(df))
        grav[0] = df[axis].iloc[0]
        vals = df[axis].values
        for j in range(1, len(df)):
            grav[j] = LP_ALPHA * vals[j] + (1 - LP_ALPHA) * grav[j - 1]
        df[f"grav_{axis[0]}"] = grav
        df[f"dyn_{axis[0]}"] = vals - grav

    # Orientation angles from gravity vector (roll / pitch)
    df["pitch"] = np.degrees(np.arctan2(df.grav_x, np.sqrt(df.grav_y**2 + df.grav_z**2)))
    df["roll"] = np.degrees(np.arctan2(df.grav_z, df.grav_y))

    # Dynamic acceleration magnitude
    df["dyn_mag"] = np.sqrt(df.dyn_x**2 + df.dyn_y**2 + df.dyn_z**2)

    return df


def gap_spans(df: pd.DataFrame):
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


def shade(ax, spans, color="#ffcc66", alpha=0.3, label=None):
    first = True
    for s, e in spans:
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0,
                   label=label if first else None)
        first = False


# ---------------------------------------------------------------------------
# (1) Overview: each session, x/y/z stacked, gaps shaded
# ---------------------------------------------------------------------------

def plot_overview(sessions):
    n = len(sessions)
    W = 9
    rows_per = 4
    fig, axes = plt.subplots(
        n * rows_per, 1, figsize=(16, 2.8 * n * rows_per),
        gridspec_kw={"height_ratios": [1.2, 0.8, 0.8, 0.8] * n},
        sharex=False,
    )
    for i, (name, df) in enumerate(sessions):
        df["wiggle"] = (
            df[["x_g", "y_g", "z_g"]]
            .rolling(W, center=True, min_periods=1)
            .std()
            .sum(axis=1)
        )
        spans = gap_spans(df)
        base = i * rows_per
        ax_raw = axes[base]
        ax_wig = axes[base + 1]
        ax_ori = axes[base + 2]
        ax_dyn = axes[base + 3]
        xlim = (df.elapsed_s.min(), df.elapsed_s.max())

        # Row 1: raw accelerometer
        shade(ax_raw, spans, label="between-word gap")
        ax_raw.plot(df.elapsed_s, df.x_g, lw=0.5, color="#d62728", label="x")
        ax_raw.plot(df.elapsed_s, df.y_g, lw=0.5, color="#2ca02c", label="y")
        ax_raw.plot(df.elapsed_s, df.z_g, lw=0.5, color="#1f77b4", label="z")
        ax_raw.set_title(name, fontsize=10, loc="left")
        ax_raw.set_ylabel("accel (g)")
        ax_raw.set_xlim(xlim)
        ax_raw.legend(loc="upper right", fontsize=7, ncol=4)

        # Row 2: wiggle (high-freq content)
        shade(ax_wig, spans)
        ax_wig.plot(df.elapsed_s, df.wiggle, lw=0.7, color="#6a3d9a",
                    label=f"wiggle (rolling stdev, w={W})")
        ax_wig.set_ylabel("wiggle")
        ax_wig.set_xlim(xlim)
        ax_wig.legend(loc="upper right", fontsize=7)

        # Row 3: orientation (pitch + roll from gravity estimate)
        shade(ax_ori, spans)
        ax_ori.plot(df.elapsed_s, df.pitch, lw=0.7, color="#e6550d", label="pitch")
        ax_ori.plot(df.elapsed_s, df.roll, lw=0.7, color="#3182bd", label="roll")
        ax_ori.set_ylabel("angle (°)")
        ax_ori.set_xlim(xlim)
        ax_ori.legend(loc="upper right", fontsize=7)
        ax_ori.set_title("orientation (from gravity estimate)", fontsize=9, loc="left")

        # Row 4: dynamic acceleration magnitude
        shade(ax_dyn, spans)
        ax_dyn.plot(df.elapsed_s, df.dyn_mag, lw=0.5, color="#31a354",
                    label="dynamic |a|")
        dyn_smooth = df.dyn_mag.rolling(W, center=True, min_periods=1).mean()
        ax_dyn.plot(df.elapsed_s, dyn_smooth, lw=1.0, color="#006d2c",
                    label=f"smoothed (w={W})")
        ax_dyn.set_ylabel("|dyn accel| (g)")
        ax_dyn.set_xlim(xlim)
        ax_dyn.legend(loc="upper right", fontsize=7)
        ax_dyn.set_title("dynamic accel (gravity removed)", fontsize=9, loc="left")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(OVERVIEW_PATH, dpi=130, bbox_inches="tight")
    print(f"wrote {OVERVIEW_PATH}")


# ---------------------------------------------------------------------------
# (2) Zoom: one gap, raw signal + "high-frequency content" feature
# ---------------------------------------------------------------------------

def plot_zoom(sessions):
    # Pick the longest session and find its first non-leading gap (skip the
    # initial "pen not started yet" stretch which is also writing==0).
    name, df = max(sessions, key=lambda s: len(s[1]))
    spans = gap_spans(df)
    # Prefer the third gap for context — still surrounded by writing on both sides.
    gap = spans[min(2, len(spans) - 1)]
    pad = 1.2
    t = df.elapsed_s.values
    lo, hi = gap[0] - pad, gap[1] + pad
    mask = (t >= lo) & (t <= hi)
    sub = df[mask].copy()

    # Rolling stdev of each axis — a smoothing-tolerant proxy for "how wiggly
    # is the signal right now". Pen-on-paper strokes spike this up; air travel
    # damps it down.
    W = 9  # ~0.18 s
    sub["wiggle"] = (
        sub[["x_g", "y_g", "z_g"]]
        .rolling(W, center=True, min_periods=1)
        .std()
        .sum(axis=1)
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})

    ax = axes[0]
    shade(ax, [gap], label="true gap")
    ax.plot(sub.elapsed_s, sub.x_g, lw=0.8, color="#d62728", label="x")
    ax.plot(sub.elapsed_s, sub.y_g, lw=0.8, color="#2ca02c", label="y")
    ax.plot(sub.elapsed_s, sub.z_g, lw=0.8, color="#1f77b4", label="z")
    ax.set_ylabel("accel (g)")
    ax.set_title(f"{name}  —  zoom on one inter-word gap")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    ax = axes[1]
    shade(ax, [gap])
    ax.plot(sub.elapsed_s, sub.wiggle, lw=1.0, color="#6a3d9a",
            label=f"rolling stdev sum  (w={W})")
    ax.set_ylabel("wiggle")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("high-frequency content — drops inside the gap")

    plt.tight_layout()
    plt.savefig(ZOOM_PATH, dpi=130, bbox_inches="tight")
    print(f"wrote {ZOOM_PATH}")


def main():
    csvs = sorted(SESSIONS_DIR.glob("session_*.csv"))
    sessions = [(p.name, load(p)) for p in csvs]
    plot_overview(sessions)
    plot_zoom(sessions)


if __name__ == "__main__":
    main()
