"""
server/session.py

Session finalization: writes annotated CSV from session buffer + events.
"""

import csv
from pathlib import Path


def write_session_csv(csv_path, buf, start, events, word_order):
    """
    Write an annotated CSV from a session's raw buffer.

    Args:
        csv_path: output CSV path
        buf: list of (x, y, z, t, writing_flag) tuples
        start: session start timestamp
        events: list of event dicts (word_start, word_stop, newline)
        word_order: list of {"sample_id", "word", "line"} from client
    """
    if not buf:
        return

    # Pair word_start/word_stop events
    word_ranges = []
    pending_start = None
    for evt in events:
        if evt["type"] == "word_start":
            pending_start = evt["t"]
        elif evt["type"] == "word_stop" and pending_start is not None:
            word_ranges.append((
                pending_start, evt["t"], evt.get("sample_id", ""),
            ))
            pending_start = None

    # Map sample_id -> (word, line) from client's final ordering
    word_map = {wo["sample_id"]: (wo["word"], wo.get("line", 1)) for wo in word_order}

    annotated = []
    for t_start, t_stop, sid in word_ranges:
        w, ln = word_map.get(sid, ("", 1))
        annotated.append((t_start, t_stop, w, ln))

    # Detect line-change gaps between consecutive words
    newline_gaps = []
    for i in range(len(annotated) - 1):
        _, t_stop_a, _, line_a = annotated[i]
        t_start_b, _, _, line_b = annotated[i + 1]
        if line_a != line_b:
            newline_gaps.append((t_stop_a, t_start_b))

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["elapsed_s", "x_g", "y_g", "z_g", "writing", "word", "newline"])
        for x, y, z, t, writing_flag in buf:
            elapsed = t - start
            word_label = ""
            for t_s, t_e, w, _ln in annotated:
                if t_s <= t <= t_e:
                    word_label = w
                    break
            nl = 0
            if not writing_flag:
                for g_start, g_end in newline_gaps:
                    if g_start <= t <= g_end:
                        nl = 1
                        break
            wr.writerow([f"{elapsed:.4f}", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                         writing_flag, word_label, nl])
