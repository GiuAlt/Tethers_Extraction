#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect tethers in force curves and interactively select which tethers to keep.

What changed from the previous version
--------------------------------------
1. MANUAL TETHERS: in the review window, click "ADD TETHER" (or press "m"),
   then click the BOTTOM point and the TOP point of a step the detector missed.
   Manual tethers go through the SAME length/force filters as auto ones; if a
   manual pick fails the filters it is rejected and the title tells you why.
2. EVERY CURVE IS SHOWN: the loop no longer skips curves where auto-detection
   found nothing. Those open with no pre-drawn tethers so you can add by hand,
   or just press DONE to record the curve as having no tether.

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

# ---------- helpers for detection (UNCHANGED) ----------
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

# ---------- interactive review (auto tethers + manual two-point add) ----------
def review_tethers_overview(x, y, auto_tethers, title, x0, min_len, min_force):
    """
    Show the full curve, the auto-detected tethers, and let the user curate.

    Controls
    --------
    Checkboxes          tick/untick which tethers to keep
    ADD TETHER button   enter add-mode, then click BOTTOM point, then TOP point
    DONE / Enter        finish this curve
    CANCEL / Esc        discard ALL tethers on this curve
    keys 1..9           toggle a tether's checkbox
    key  a / n          keep all / keep none
    key  m              toggle add-mode

    Parameters
    ----------
    auto_tethers : list of (xb, yb, xt, yt)   # bottom point, then top point
    x0           : reference x (curve minimum x) used for tether length
    min_len,
    min_force    : the same filters applied to auto detections

    Returns
    -------
    list of accepted (xb, yb, xt, yt) tuples  (kept auto + accepted manual)
    """
    # Each tether is a dict carrying its geometry, its keep flag, its kind,
    # and the matplotlib artists used to draw it (so we can dim/remove them).
    tethers = []

    # Mutable holders so the nested callbacks can share state cleanly.
    state = {
        "cbtns": None,        # current CheckButtons widget (rebuilt on add)
        "labels": [],         # current checkbox labels, same order as `tethers`
        "add_mode": False,    # are we waiting for manual clicks?
        "pending": [],        # clicks collected so far in add-mode
        "temp_artists": [],   # temporary "x" markers shown while clicking
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    # keep the plot area left of the widget column (widgets start at x=0.83)
    fig.subplots_adjust(left=0.09, right=0.80, top=0.90, bottom=0.12)
    ax.plot(x, y, label="Curve", lw=1.2, alpha=0.8)
    


    def _metrics(t):
        """Length and force of a tether, matching the auto-detector convention."""
        top_y, bot_y = max(t["yb"], t["yt"]), min(t["yb"], t["yt"])
        top_x = t["xt"] if t["yt"] >= t["yb"] else t["xb"]  # x of higher-force point
        return (top_x - x0), (top_y - bot_y)

    def _draw(t):
        """Draw a tether (two markers + a connecting line) and store its artists."""
        t["scatter"] = ax.scatter([t["xb"], t["xt"]], [t["yb"], t["yt"]],
                                   s=180, zorder=3)
        t["line"], = ax.plot([t["xb"], t["xt"]], [t["yb"], t["yt"]], lw=3)

    # --- draw the auto-detected tethers up front ---
    for (xb, yb, xt, yt) in auto_tethers:
        t = {"xb": xb, "yb": yb, "xt": xt, "yt": yt, "keep": True, "kind": "Auto"}
        _draw(t)
        tethers.append(t)

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("Signal"); ax.grid(True); ax.legend(loc="best")

    # --- widget axes on the right-hand side ---
    cb_ax     = fig.add_axes([0.83, 0.32, 0.15, 0.55])
    add_ax    = fig.add_axes([0.83, 0.22, 0.15, 0.07])
    done_ax   = fig.add_axes([0.83, 0.12, 0.15, 0.07])
    cancel_ax = fig.add_axes([0.83, 0.02, 0.15, 0.07])

    add_btn    = Button(add_ax, "ADD TETHER")
    done_btn   = Button(done_ax, "DONE")
    cancel_btn = Button(cancel_ax, "CANCEL")

    def _toggle(label):
        """Checkbox callback: flip a tether's keep flag and dim it if dropped."""
        k = state["labels"].index(label)
        tethers[k]["keep"] = not tethers[k]["keep"]
        alpha = 1.0 if tethers[k]["keep"] else 0.2
        tethers[k]["scatter"].set_alpha(alpha)
        tethers[k]["line"].set_alpha(alpha)
        fig.canvas.draw_idle()

    def _rebuild_checkboxes():
        """(Re)create the checkbox panel from the current list of tethers."""
        cb_ax.clear()
        cb_ax.set_title("Keep?")
        labels  = [f"{t['kind']} {k}" for k, t in enumerate(tethers)]
        actives = [t["keep"] for t in tethers]
        if labels:
            cbtns = CheckButtons(cb_ax, labels, actives)
            cbtns.on_clicked(_toggle)
        else:
            cbtns = None
            cb_ax.text(0.5, 0.5, "(none yet)\nuse ADD TETHER",
                       ha="center", va="center", transform=cb_ax.transAxes)
            cb_ax.set_xticks([]); cb_ax.set_yticks([])
        state["cbtns"]  = cbtns
        state["labels"] = labels
        fig.canvas.draw_idle()

    def _set_add_mode(on):
        """Turn manual add-mode on/off and update the button label/colour."""
        state["add_mode"] = on
        add_btn.label.set_text("CLICK 2 PTS…" if on else "ADD TETHER")
        add_ax.set_facecolor("0.8" if on else "1.0")
        fig.canvas.draw_idle()

    def _clear_pending():
        """Forget half-finished clicks and remove their temporary markers."""
        for m in state["temp_artists"]:
            m.remove()
        state["temp_artists"].clear()
        state["pending"].clear()

    def _finalize_manual():
        """Turn the two collected clicks into a tether, applying the filters."""
        (x1, y1), (x2, y2) = state["pending"]
        # bottom = lower-force point, top = higher-force point
        if y1 <= y2:
            t = {"xb": x1, "yb": y1, "xt": x2, "yt": y2}
        else:
            t = {"xb": x2, "yb": y2, "xt": x1, "yt": y1}
        t.update({"keep": True, "kind": "Manual"})

        L, TF = _metrics(t)
        _clear_pending()
        _set_add_mode(False)

        # same filters as the auto-detector
        if L < min_len or TF < min_force:
            ax.set_title(f"{title}\nREJECTED manual pick: "
                         f"L={L:.3f} µm, F={TF:.3f} nN "
                         f"(need ≥ {min_len} µm and ≥ {min_force} nN)")
            fig.canvas.draw_idle()
            return

        _draw(t)
        tethers.append(t)
        _rebuild_checkboxes()
        ax.set_title(f"{title}   (added manual tether: "
                     f"L={L:.2f} µm, F={TF:.3f} nN)")
        fig.canvas.draw_idle()

    def _on_click(ev):
        """Collect clicks only while in add-mode and only inside the curve axes."""
        if not state["add_mode"] or ev.inaxes is not ax:
            return
        if ev.xdata is None or ev.ydata is None:
            return
        state["pending"].append((ev.xdata, ev.ydata))
        # show a temporary marker so the click is clearly registered
        m = ax.scatter([ev.xdata], [ev.ydata], s=120, marker="x",
                       color="k", zorder=5)
        state["temp_artists"].append(m)
        fig.canvas.draw_idle()
        if len(state["pending"]) == 2:
            _finalize_manual()

    def _add(_):
        _clear_pending()
        _set_add_mode(not state["add_mode"])

    def _done(_):
        plt.close(fig)

    def _cancel(_):
        for t in tethers:
            t["keep"] = False
        plt.close(fig)

    def _on_key(ev):
        cbtns = state["cbtns"]
        if ev.key in list("123456789") and cbtns is not None:
            i = int(ev.key) - 1
            if 0 <= i < len(tethers):
                cbtns.set_active(i)
        elif ev.key == "a" and cbtns is not None:
            for k in range(len(tethers)):
                if not tethers[k]["keep"]:
                    cbtns.set_active(k)
        elif ev.key == "n" and cbtns is not None:
            for k in range(len(tethers)):
                if tethers[k]["keep"]:
                    cbtns.set_active(k)
        elif ev.key == "m":
            _add(None)
        elif ev.key == "enter":
            _done(None)
        elif ev.key == "escape":
            _cancel(None)

    # wire everything up
    _rebuild_checkboxes()
    add_btn.on_clicked(_add)
    done_btn.on_clicked(_done)
    cancel_btn.on_clicked(_cancel)
    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show(block=True)

    # return the geometry of every kept tether (auto kept + manual accepted)
    return [(t["xb"], t["yb"], t["xt"], t["yt"]) for t in tethers if t["keep"]]

# ---------- main parameters ----------
MIN_TETHER_LEN   = 0.05  # µm
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

    # reference x (curve minimum x), used for tether length
    x0 = float(np.min(x))

    # --- build auto-detected tethers as (bottom, top) endpoint tuples ---
    # (this replaces the old df/valid_pairs bookkeeping; may be empty)
    auto_tethers = []
    if kept_pairs:
        kept_positions = [idx for s, p in kept_pairs for idx in (s, p)]
        df = pd.DataFrame({
            "Index": kept_positions,
            "X": [x[i] for i in kept_positions],
            "Y": [y[i] for i in kept_positions],
        })

        # pair indices ordered by X (left point, right point)
        n_pts = len(df)
        even_n = n_pts - (n_pts % 2)
        raw_pairs = list(zip(range(0, even_n, 2), range(1, even_n, 2)))
        pairs_idx = []
        for i_pair, j_pair in raw_pairs:
            xi, xj = float(df["X"].iloc[i_pair]), float(df["X"].iloc[j_pair])
            pairs_idx.append((i_pair, j_pair) if xi <= xj else (j_pair, i_pair))

        # apply the length/force filters, store survivors as (bottom, top)
        for i_pair, j_pair in pairs_idx:
            L  = float(df["X"].iloc[j_pair] - x0)
            TF = float(df["Y"].iloc[j_pair] - df["Y"].iloc[i_pair])
            if L >= MIN_TETHER_LEN and TF >= MIN_TETHER_FORCE:
                xi_, yi_ = float(df["X"].iloc[i_pair]), float(df["Y"].iloc[i_pair])
                xj_, yj_ = float(df["X"].iloc[j_pair]), float(df["Y"].iloc[j_pair])
                if yi_ <= yj_:
                    auto_tethers.append((xi_, yi_, xj_, yj_))
                else:
                    auto_tethers.append((xj_, yj_, xi_, yi_))

    # --- ALWAYS open the review window, even with zero auto tethers ---
    title = (f"Cell {Cell_number} | Curve {c} | "
             f"{len(auto_tethers)} auto tether(s) ≥ {MIN_TETHER_LEN} µm")
    kept_tethers = review_tethers_overview(
        x, y, auto_tethers, title, x0, MIN_TETHER_LEN, MIN_TETHER_FORCE
    )

    # --- build output rows from whatever the user accepted ---
    rows = []
    for bin_i, (xb, yb, xt, yt) in enumerate(kept_tethers):
        top_y, bot_y = max(yb, yt), min(yb, yt)
        top_x = xt if yt >= yb else xb
        rows.append({
            "Type":        Condition.split("_")[0],  # "NINJKO"/"WT" from "NINJKO_T9_C1"
            "Condition":   Condition,
            "Cell number": Cell_number,
            "Interval":    Timepoint,      # global from the loading script
            "Curve":       c,
            "Tet_F":       float(top_y - bot_y),
            "Tet_Length":  float(top_x - x0),
            "endadhesion": float(x0),
            "min_adh":     float(x0),
            "Bin":         bin_i,
        })

    if not rows:
        print(f"[Curve {c}] no tethers kept.")
        discarded_curves.append(c)
        continue

    tf_all.append(pd.DataFrame(rows))
    kept_curves.append(c)
    print(f"[Curve {c}] kept {len(rows)} tether(s).")

# concatenate final table
tf_all = pd.concat(tf_all, ignore_index=True) if tf_all else pd.DataFrame()
print(f"\nProcessing complete. Curves kept: {kept_curves} | "
      f"fully discarded: {discarded_curves}")