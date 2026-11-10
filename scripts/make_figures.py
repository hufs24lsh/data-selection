#!/usr/bin/env python3
"""Draws the README figures from results/seed42.json.

  assets/energy_vs_score.png       score against end-to-end GPU energy, one panel per testbed
  assets/mixed_pool_composition.png medical share of the selected 10% in the mixed pools

  python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_ID, C_PD, C_RAND, C_FULL = "#2a78d6", "#8d7150", "#1baf7a", MUTED     # ShallowFrontier in HUFS Gold (as on the poster), full data in neutral gray
PANELS = ["qwen_math10k", "llama_med10k", "qwen_mix20k", "llama_mix20k"]
METRIC = dict(math="math, mean@16", med="medical, MCF accuracy", mixed="(math + medical) / 2")
MIXED = [("llama_mix20k", "Llama-3-8B\n20K pool"), ("qwen_mix20k", "Qwen2.5-7B\n20K pool"), ("llama_mix10k", "Llama-3-8B\n10K pool")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _point(ax, x, y, color, marker, size, z=3):
    ax.scatter([x], [y], s=size, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=z)


def _label(ax, text, x, y, ha="center", dx=0):
    """direct label under a marker, in secondary ink (never the series colour)"""
    ax.annotate(text, (x, y), xytext=(dx, -15), textcoords="offset points", ha=ha, fontsize=8.5, color=INK2)


def energy_vs_score(res, out, size=(11.6, 2.9), dpi=220, star=340):   # sized so its text reads at README width
    fig, axes = plt.subplots(1, len(PANELS), figsize=size)
    for ax, key in zip(axes, PANELS):
        tb = res["testbeds"][key]
        m = tb["methods"]
        rounds = [m["instructdiff_t%d" % r] for r in (1, 2, 3)]
        xs, ys = [r["kwh"] for r in rounds], [r["score"] for r in rounds]
        ax.plot(xs, ys, color=C_ID, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        for r, (x, y) in enumerate(zip(xs, ys), 1):
            _point(ax, x, y, C_ID, "o", 64)
            if r == 1:                                            # labels sit on the side the line does not use
                at = dict(xytext=(-9, 0), ha="right", va="center")
            elif r == 3:
                at = dict(xytext=(9, 0), ha="left", va="center")
            elif y > max(ys[0], ys[2]):                           # peak: both lines go down, so the label goes above
                at = dict(xytext=(0, 10), ha="center", va="bottom")
            elif y < min(ys[0], ys[2]):                           # valley: both lines go up, so the label goes below
                at = dict(xytext=(0, -12), ha="center", va="top")
            else:
                at = dict(xytext=(9, 0), ha="left", va="center")
            ax.annotate("T%d" % r, (x, y), textcoords="offset points", fontsize=8.5, color=INK2, **at)
        pts = [(x, y) for x, y in zip(xs, ys)]
        for rk, marker, text, size in (("random", "s", "Random", 60), ("balanced_random", "^", "Balanced random", 90)):
            if m.get(rk, {}).get("score") is not None:
                _point(ax, m[rk]["kwh"], m[rk]["score"], C_RAND, marker, size)
                _label(ax, text, m[rk]["kwh"], m[rk]["score"], ha="left", dx=-6)
                pts.append((m[rk]["kwh"], m[rk]["score"]))
        if m.get("full", {}).get("score") is not None:
            _point(ax, m["full"]["kwh"], m["full"]["score"], C_FULL, "D", 52)
            _label(ax, "Full data", m["full"]["kwh"], m["full"]["score"])
            pts.append((m["full"]["kwh"], m["full"]["score"]))
        pd = m["shallowfrontier"]
        _point(ax, pd["kwh"], pd["score"], C_PD, "*", star, z=4)
        ax.annotate("SF", (pd["kwh"], pd["score"]), xytext=(0.46 * star ** 0.5, 0.35 * star ** 0.5), textcoords="offset points", fontsize=9, color=INK, fontweight="bold")
        pts.append((pd["kwh"], pd["score"]))
        x1 = max(p[0] for p in pts)
        y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
        ax.set_xlim(0, x1 * 1.18)
        pad = (y1 - y0) * 0.22 + 0.15
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_title(tb["label"], loc="left", pad=8)
        ax.set_xlabel("End-to-end GPU energy (kWh)")
        ax.set_ylabel("Score: " + METRIC[tb["metric"]])
    handles = [Line2D([], [], color=C_ID, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label="InstructDiff T1 → T3 (cumulative rounds)"),
               Line2D([], [], color=C_PD, lw=0, marker="*", ms=14, mec=SURFACE, mew=1.5, label="ShallowFrontier (SF, one calibration)"),
               Line2D([], [], color=C_RAND, lw=0, marker="s", ms=7, mec=SURFACE, mew=1.5, label="Random 10%"),
               Line2D([], [], color=C_RAND, lw=0, marker="^", ms=9, mec=SURFACE, mew=1.5, label="Balanced random 10%"),
               Line2D([], [], color=C_FULL, lw=0, marker="D", ms=6.5, mec=SURFACE, mew=1.5, label="Full data")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=9.5, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def mixed_composition(res, out):
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    names = [("instructdiff_t1", "T1"), ("instructdiff_t2", "T2"), ("instructdiff_t3", "T3"), ("shallowfrontier", "SF")]
    width, step = 0.17, 1.0
    ticks, labels = [], []
    for g, (key, glabel) in enumerate(MIXED):
        comp = res["composition"][key]
        for i, (mk, short) in enumerate(names):
            c = comp[mk]
            share = 100.0 * c.get("med", 0) / (c.get("med", 0) + c.get("math", 0))
            x = g * step + (i - 1.5) * (width + 0.04)
            ax.bar([x], [share], width=width, color=C_PD if mk == "shallowfrontier" else C_ID, zorder=3)
            ax.annotate("%.1f%%" % share if share < 10 else "%.0f%%" % share, (x, share), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
            ticks.append(x)
            labels.append(short)
        ax.text(g * step, -16, glabel, ha="center", va="top", fontsize=9, color=INK)
    ax.axhline(50, color=MUTED, lw=1, zorder=2)
    ax.text(len(MIXED) * step - 0.55, 51.5, "pool: 50% medical", fontsize=8.5, color=INK2, ha="right")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, len(MIXED) * step - 0.45)
    ax.set_ylabel("Medical share of the selection (%)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=C_ID, lw=6, label="InstructDiff rounds T1–T3"), Line2D([], [], color=C_PD, lw=6, label="ShallowFrontier (SF)")]
    ax.legend(handles=handles, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    res = json.load(open(os.path.join(ROOT, "results", "seed42.json"), encoding="utf-8"))
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    energy_vs_score(res, os.path.join(ROOT, "assets", "energy_vs_score.png"))
    mixed_composition(res, os.path.join(ROOT, "assets", "mixed_pool_composition.png"))
    print("wrote assets/energy_vs_score.png, assets/mixed_pool_composition.png")


if __name__ == "__main__":
    main()
