#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:28:46 2025

@author: giuliaam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.signal import savgol_filter

# ---------- helpers ----------
def mad(x):
    if len(x) == 0: return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def robust_line_fit(x, y, iters=3):
    X = np.column_stack([x, np.ones_like(x)])
    w = np.ones_like(y, dtype=float)
    a = b = 0.0
    for _ in range(iters):
        WX = X * w[:, None]
        theta, *_ = np.linalg.lstsq(WX, y * w, rcond=None)
        a, b = float(theta[0]), float(theta[1])
        r = y - (a*x + b)
        s = mad(r)
        if s <= 0: break
        c = 4.685 * s
        u = r / c
        w = np.clip(1 - u*u, 0, 1)
        w[u >= 1] = 0.0
    return a, b

# ---------- interactor ----------
class BaselineCorrector:
    """
    Mouse-driven baseline picker:
      - Drag to select baseline region
      - Enter: accept & save correction for this curve
      - r: reset selection
      - s: skip curve
      - q: quit
      - 1/2/3: polynomial degree for baseline fit over selected span (default 1)
    """
    def __init__(self, All, max_curves=30, smooth_wl=21, smooth_po=2):
        self.All = All
        self.curve_ids = sorted(All['Curve'].unique().tolist())[:max_curves]
        self.smooth_wl = smooth_wl
        self.smooth_po = smooth_po

        self.results = []      # list of DataFrames for All2
        self.i = 0             # index in curve_ids
        self.degree = 1        # baseline polynomial degree

        self.fig, self.ax = plt.subplots(figsize=(9.2, 4.2))
        self.ax2 = self.ax.twinx()  # optional overlay axis for messages if wanted
        self.ax2.set_yticks([])
        self.ax2.set_ylabel("")

        self.span = SpanSelector(
            self.ax, self.on_select, "horizontal",
            useblit=True, interactive=True, drag_from_anywhere=True,
            props=dict(alpha=0.2, facecolor="orange"),
            minspan=0.0
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.baseline_line = None
        self.corrected_line = None
        self.selection_patch = None
        self.text = None

        self.load_curve()
        plt.tight_layout()
        plt.show()

    # ------------ data prep ------------
    def prep_trace(self, data):
        y = data["Deflection"].values
        if len(y) >= max(7, self.smooth_wl):
            y = savgol_filter(y, window_length=min(self.smooth_wl, len(y)//2*2-1), polyorder=self.smooth_po)
        x = data["TS"].values

        # crop from absolute minimum (your heuristic)
        j = int(np.argmin(y))
        return x[j:].astype(float), y[j:].astype(float)

    # ------------ plotting ------------
    def load_curve(self):
        # clean existing overlays
        self.baseline_line = None
        self.corrected_line = None
        self.selection_patch = None
        self.sel_mask = None
        
        self._baseline = None
        self._y_corr = None
        self.ax.clear()
        self.ax2.clear(); self.ax2.set_yticks([])
        self.text = None  
        if self.i >= len(self.curve_ids):
            self.finish()
            return

        t = self.curve_ids[self.i]
        data = self.All[self.All["Curve"] == t]
        self.curve_meta = dict(
            Curve=t,
            Condition=data["Condition"].iloc[0],
            Cell_number=data["Cell_number"].iloc[0]
        )
        self.x_raw, self.y_raw = self.prep_trace(data)

        self.ax.plot(self.x_raw, self.y_raw, label="Original (cropped)", alpha=0.6)
        self.ax.set_xlabel("Distance (µm)")
        self.ax.set_ylabel("Force (nN)")
        self.ax.set_title(f'{self.curve_meta["Condition"]}  |  Curve {t}')
        self.ax.grid(True, alpha=0.35)
        self.ax.legend(loc="best")

        self.msg("Drag to select baseline region · Enter=accept · r=redo · s=skip · q=quit · 1/2/3=poly degree")
        self.fig.canvas.draw_idle()

        # clean existing overlays
        self.baseline_line = None
        self.corrected_line = None
        self.selection_patch = None
        self.sel_mask = None

    def msg(self, s):
        """Update or create the status label without calling remove()."""
        if getattr(self, "text", None) is not None and self.text.axes is self.ax:
            # label still attached to this Axes: just update the text
            self.text.set_text(s)
        else:
            # recreate it
            self.text = self.ax.text(
                0.01, 0.98, s, transform=self.ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.8", alpha=0.8)
            )
        self.fig.canvas.draw_idle()
    # ------------ interactions ------------
    def on_select(self, xmin, xmax):
        if xmin == xmax:  # ignore clicks w/o span
            return
        if xmin > xmax: xmin, xmax = xmax, xmin
        m = (self.x_raw >= xmin) & (self.x_raw <= xmax)
        if m.sum() < 5:
            self.msg("Selection too small; drag a wider baseline span.")
            return
        self.sel_mask = m

        # fit baseline of chosen degree on selected region, apply to whole trace
        if self.degree == 1:
            a, b = robust_line_fit(self.x_raw[m], self.y_raw[m])
            baseline = a * self.x_raw + b
        else:
            # polynomial fit (ordinary LS) on selected region
            coeffs = np.polyfit(self.x_raw[m], self.y_raw[m], deg=self.degree)
            baseline = np.polyval(coeffs, self.x_raw)

        y_corr = self.y_raw - baseline

        # draw selection patch
        if self.selection_patch:
            try: self.selection_patch.remove()
            except Exception: pass
        self.selection_patch = self.ax.axvspan(xmin, xmax, color="orange", alpha=0.15, zorder=0)

        # draw/update baseline and corrected
        if self.baseline_line is None:
            self.baseline_line, = self.ax.plot(self.x_raw, baseline, lw=2.0, label=f"Baseline (deg {self.degree})")
            self.corrected_line, = self.ax.plot(self.x_raw, y_corr, lw=1.6, label="Corrected")
            self.ax.legend(loc="best")
        else:
            self.baseline_line.set_ydata(baseline)
            self.baseline_line.set_label(f"Baseline (deg {self.degree})")
            self.corrected_line.set_ydata(y_corr)
        self.msg("Press Enter to accept · r=redo · change degree with 1/2/3")
        self.fig.canvas.draw_idle()

        # stash to save on Enter
        self._baseline = baseline
        self._y_corr = y_corr

    def on_key(self, event):
        if event.key in ("1", "2", "3"):
            self.degree = int(event.key)
            # recompute if we already have a selection
            if self.sel_mask is not None:
                self.on_select(self.x_raw[self.sel_mask].min(), self.x_raw[self.sel_mask].max())
            return
        if event.key == "r":
            self.load_curve()
            return
        if event.key == "s":
            self.msg("Skipped.")
            self.i += 1
            self.load_curve()
            return
        if event.key == "q":
            plt.close(self.fig)
            return
        if event.key == "enter":
            if self.sel_mask is None or not hasattr(self, "_y_corr"):
                self.msg("No selection yet. Drag a baseline span, then press Enter.")
                return
            # save result
            B = pd.DataFrame({
                "Deflection": self._y_corr,
                "TS": self.x_raw,
                "Curve": self.curve_meta["Curve"],
                "Condition": self.curve_meta["Condition"],
                "Cell_number": self.curve_meta["Cell_number"],
                "baseline_degree": self.degree
            })
            self.results.append(B)
            self.i += 1
            self.load_curve()

    def finish(self):
        plt.close(self.fig)
        if self.results:
            All2 = pd.concat(self.results, ignore_index=True)
            print(f"Baseline-corrected curves: {All2['Curve'].nunique()} (rows={len(All2)})")
            # expose as attribute so you can grab it
            self.All2 = All2
        else:
            print("No corrected curves saved.")
            self.All2 = pd.DataFrame()

if __name__ == "__main__":
    import matplotlib
    # Choose an interactive backend you have available
    try:
        matplotlib.use("MacOSX")   # good on macOS without extra installs
    except Exception:
        pass

    # Expect `All` to already exist in your console/session
    bc = BaselineCorrector(All, max_curves=30)


    # Block until the UI window is closed
    while plt.fignum_exists(bc.fig.number):
        plt.pause(0.1)
    
    # Now finish() has run; grab the table
    All2 = getattr(bc, "All2", pd.DataFrame())
    print(len(All2), "rows in All2")
    
    

