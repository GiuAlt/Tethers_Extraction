#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect tethers in force curves and interactively select which tethers to keep.

Requirements:
    pip install numpy pandas matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib
# choose an interactive backend if needed:
# matplotlib.use("TkAgg")   # or "Qt5Agg"
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from scipy.signal import savgol_filter

# ---------- helpers for detection ----------
def _mad(a):
    med = np.median(a)
    return np.median(np.abs(a - med))

def _find_flat_index(y, start, direction, dy, flat_mult=3.0, run_len=8, max_search=400):
    N = len(y)
    sigma_d = 1.4826 * _mad(dy) + 1e-12
    flat_thr = flat_mult * sigma_d
    lo = max(0, start - max_search)
    hi = min(N, start + max_search + 1)
    idx = start
    consec = 0
    while lo <= idx < hi:
        if abs(dy[idx]) < flat_thr:
            consec += 1
            if consec >= run_len:
                return int(np.clip(idx - (run_len // 2) * direction, 0, N - 1))
        else:
            consec = 0
        idx += direction
    return int(np.clip(start, 0, N - 1))

def _find_drop_start_left(y, step_idx, dy, drop_mult=6.0, run_len=6, max_search=400):
    N = len(y)
    sigma_d = 1.4826 * _mad(dy) + 1e-12
    drop_thr = -drop_mult * sigma_d
    lo = max(0, step_idx - max_search)
    consec = 0
    for idx in range(step_idx, lo - 1, -1):
        if dy[idx] < drop_thr:
            consec += 1
        else:
            if consec >= run_len:
                return max(lo, min(N - 1, idx + 1))
            consec = 0
    return step_idx

def choose_pre_and_post_points(x, y, step_idx, run_len_flat=8, max_search=400):
    dy = np.gradient(y, x) if x is not None else np.gradient(y)
    s_idx = _find_flat_index(y, start=step_idx, direction=+1, dy=dy,
                             run_len=run_len_flat, max_search=max_search)
    pre_drop_idx = _find_drop_start_left(y, step_idx=step_idx, dy=dy,
                                         drop_mult=6.0, run_len=6, max_search=max_search)
    pre_idx = _find_flat_index(y, start=pre_drop_idx, direction=-1, dy=dy,
                               run_len=run_len_flat, max_search=max_search)
    return int(pre_idx), int(s_idx)

# ---------- interactive review (all tethers at once) ----------
def review_tethers_overview(x, y, df, valid_pairs, title):
    """
    One figure with the full curve and all detected tethers.
    Lets the user tick which tethers to keep.
    Keyboard shortcuts: 1..9 toggle tether, a=all, n=none, Enter=done, Esc=discard all.
    Returns: keep_mask (list[bool]) with same length as valid_pairs.
    """
    keep_mask = [True] * len(valid_pairs)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y, label="Curve", lw=1.2, alpha=0.8)

    lines, points, labels = [], [], []
    for k, (i_pair, j_pair) in enumerate(valid_pairs):
        xi, yi = float(df["X"].iloc[i_pair]), float(df["Y"].iloc[i_pair])
        xj, yj = float(df["X"].iloc[j_pair]), float(df["Y"].iloc[j_pair])
        p = ax.scatter([xi, xj], [yi, yj], s=180, zorder=3)
        l, = ax.plot([xj, xj], [yi, yj], lw=3)
        points.append(p); lines.append(l); labels.append(f"Tether {k}")

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("Signal"); ax.grid(True); ax.legend(loc="best")

    cb_ax = fig.add_axes([0.83, 0.25, 0.15, 0.6])
    cbtns = CheckButtons(cb_ax, labels, [True]*len(labels))
    cb_ax.set_title("Keep?")

    done_ax   = fig.add_axes([0.83, 0.12, 0.15, 0.07])
    cancel_ax = fig.add_axes([0.83, 0.04, 0.15, 0.07])
    done_btn   = Button(done_ax, "DONE")
    cancel_btn = Button(cancel_ax, "CANCEL")

    def _toggle(label):
        idx = labels.index(label)
        keep_mask[idx] = not keep_mask[idx]
        alpha = 1.0 if keep_mask[idx] else 0.2
        points[idx].set_alpha(alpha)
        lines[idx].set_alpha(alpha)
        fig.canvas.draw_idle()

    def _done(_): plt.close(fig)
    def _cancel(_):
        for i in range(len(keep_mask)): keep_mask[i] = False
        plt.close(fig)

    def _on_key(ev):
        if ev.key in list("123456789"):
            i = int(ev.key) - 1
            if 0 <= i < len(labels): cbtns.set_active(i)
        elif ev.key == "a":
            for i in range(len(labels)):
                if not keep_mask[i]: cbtns.set_active(i)
        elif ev.key == "n":
            for i in range(len(labels)):
                if keep_mask[i]: cbtns.set_active(i)
        elif ev.key == "enter":
            _done(None)
        elif ev.key == "escape":
            _cancel(None)

    cbtns.on_clicked(_toggle)
    done_btn.on_clicked(_done)
    cancel_btn.on_clicked(_cancel)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show(block=True)
    return keep_mask

# ---------- main parameters ----------
MIN_TETHER_LEN   = 0.2   # µm
MIN_SPACING_X    = 1.0
MIN_TETHER_FORCE = 0.02
xx = 1.0  # sigma multiplier for detection

# ---------- processing loop ----------
tf_all = []
kept_curves = []
discarded_curves = []

for c in range(30):
    data2 = All2[All2["Curve"] == c]
    if len(data2) == 0:
        continue

    # --- prepare signal ---
    c_f_def = savgol_filter(data2["Deflection"].values, window_length=21, polyorder=2)
    c_f_ts  = data2["TS"].values
    Cell_number = data2["Cell_number"].iloc[0]
    Condition   = data2["Condition"].iloc[0]

    min_index = int(np.argmin(c_f_def))
    y = c_f_def[min_index:]
    x = c_f_ts[min_index:]

    # reversed for detection
    x_rev, y_rev = np.flip(x), np.flip(y)
    tail = y_rev[:min(1000, len(y_rev))]
    sigma = np.std(tail) if len(tail) else 0.0
    threshold = max(1e-12, xx * sigma)

    # step detection
    step_positions = []
    N = len(y)
    for i in range(2, N):
        if abs(y_rev[i] - y_rev[i - 2]) > threshold:
            j = N - i
            pre_idx, s_idx = choose_pre_and_post_points(x, y, step_idx=j)
            step_positions.extend([s_idx, pre_idx])
    step_positions.reverse()

    # spacing filter
    pairs = list(zip(step_positions[0::2], step_positions[1::2]))
    kept_pairs = []
    last_x = np.inf
    for s_idx, p_idx in reversed(pairs):
        s_idx = int(np.clip(s_idx, 0, len(x) - 1))
        p_idx = int(np.clip(p_idx, 0, len(x) - 1))
        if last_x - x[s_idx] >= MIN_SPACING_X:
            kept_pairs.append((s_idx, p_idx))
            last_x = x[s_idx]
    kept_pairs.reverse()
    if not kept_pairs:
        print(f"[Curve {c}] no valid step points after spacing.")
        discarded_curves.append(c)
        continue

    # DataFrame of detected points
    kept_positions = [idx for s, p in kept_pairs for idx in (s, p)]
    df = pd.DataFrame({
        "Index": kept_positions,
        "X": [x[i] for i in kept_positions],
        "Y": [y[i] for i in kept_positions],
    })

    # pair indices ordered by X
    n_pts = len(df)
    even_n = n_pts - (n_pts % 2)
    raw_pairs = list(zip(range(0, even_n, 2), range(1, even_n, 2)))
    pairs_idx = []
    for i_pair, j_pair in raw_pairs:
        xi, xj = float(df["X"].iloc[i_pair]), float(df["X"].iloc[j_pair])
        pairs_idx.append((i_pair, j_pair) if xi <= xj else (j_pair, i_pair))

    # tether filters
    x0 = float(np.min(x))
    valid_pairs = []
    for i_pair, j_pair in pairs_idx:
        L  = float(df["X"].iloc[j_pair] - x0)
        TF = float(df["Y"].iloc[j_pair] - df["Y"].iloc[i_pair])
        if L >= MIN_TETHER_LEN and TF >= MIN_TETHER_FORCE:
            valid_pairs.append((i_pair, j_pair))
    if not valid_pairs:
        print(f"[Curve {c}] no tethers ≥ {MIN_TETHER_LEN} µm after filtering.")
        discarded_curves.append(c)
        continue

    # interactive selection of tethers
    title = f"Cell {Cell_number} | Curve {c} | {len(valid_pairs)} tether(s) ≥ {MIN_TETHER_LEN} µm"
    keep_mask = review_tethers_overview(x, y, df, valid_pairs, title)

    # build output rows
    rows = []
    kept_count = 0
    for keep, (i_pair, j_pair) in zip(keep_mask, valid_pairs):
        if not keep:
            continue
        kept_count += 1
        rows.append({
            "Type":        Cell_type,
            "Condition":   Condition,
            "Cell number": Cell_number,
            "Interval":    Timepoint,
            "Curve":       c,
            "Tet_F":       float(df["Y"].iloc[j_pair] - df["Y"].iloc[i_pair]),
            "Tet_Length":  float(df["X"].iloc[j_pair] - x0),
            "endadhesion": float(x0),
            "min_adh":     float(x0),
            "Bin":         kept_count - 1,
        })

    if kept_count == 0:
        print(f"[Curve {c}] all tethers discarded by user.")
        discarded_curves.append(c)
        continue

    tf_all.append(pd.DataFrame(rows))
    kept_curves.append(c)
    print(f"[Curve {c}] kept {kept_count}/{len(valid_pairs)} tether(s).")

# concatenate final table
tf_all = pd.concat(tf_all, ignore_index=True) if tf_all else pd.DataFrame()
print(f"\nProcessing complete. Curves kept: {kept_curves} | fully discarded: {discarded_curves}")
