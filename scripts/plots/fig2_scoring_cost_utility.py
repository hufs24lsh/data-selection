"""Scoring-cost versus downstream-utility figure from frozen results."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from common import METHOD_COLORS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/cost/final_cost_comparison.csv"

with SOURCE.open(newline="", encoding="utf-8") as f:
    rows = {row["method"]: row for row in csv.DictReader(f)}

# The original experimental key is retained for data compatibility.
# The public display name is simply K128.
keys = ["T1", "K256", "true-K128"]
assert set(rows) == set(keys)

display = {
    "T1": "T1",
    "K256": "K256",
    "true-K128": "K128",
}

colors = {
    "T1": METHOD_COLORS["T1"],
    "K256": METHOD_COLORS["K256"],
    "true-K128": METHOD_COLORS["K128"],
}

data = {}

for key in keys:
    row = rows[key]
    data[key] = {
        "tokens": int(row["scoring_processed_tokens_2models"]),
        "macro": float(row["macro"]),
        "reduction": float(row["scoring_token_reduction_vs_t1_pct"]),
    }

expected_tokens = {
    "T1": 14510380,
    "K256": 10282340,
    "true-K128": 7702010,
}

for key in keys:
    assert data[key]["tokens"] == expected_tokens[key]
    data[key]["tokens_m"] = data[key]["tokens"] / 1_000_000

t1_tokens = data["T1"]["tokens"]

for key in keys:
    computed_reduction = 100 * (1 - data[key]["tokens"] / t1_tokens)
    assert abs(computed_reduction - data[key]["reduction"]) < 1e-8

assert abs(data["T1"]["macro"] - 47.1) < 1e-9
assert abs(data["K256"]["macro"] - 46.68940115767946) < 1e-9
assert abs(data["true-K128"]["macro"] - 46.60570178330249) < 1e-9

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor="white")
ax.set_facecolor("white")

# Preserve the approved V2 ranges.
ax.set_xlim(6.8, 15.25)
ax.set_ylim(45.5, 48.0)

ax.grid(True, alpha=0.13, linewidth=0.8)
ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#BBBBBB")

# Horizontal comparison lines.
connectors = [
    ("K256", 46.08, 0.06, "bottom"),
    ("true-K128", 45.86, -0.06, "top"),
]

for key, y, offset, vertical_alignment in connectors:
    x1 = data[key]["tokens_m"]
    x2 = data["T1"]["tokens_m"]

    ax.plot(
        [x1, x2],
        [y, y],
        color=colors[key],
        linewidth=1.8,
        alpha=0.95,
        solid_capstyle="round",
        zorder=2,
    )

    ax.text(
        (x1 + x2) / 2,
        y + offset,
        f"-{data[key]['reduction']:.1f}%",
        color=colors[key],
        ha="center",
        va=vertical_alignment,
        fontsize=10.8,
        fontweight="semibold",
    )

order = ["true-K128", "K256", "T1"]

for key in order:
    ax.scatter(
        data[key]["tokens_m"],
        data[key]["macro"],
        s=145,
        color=colors[key],
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )

label_offsets = {
    "T1": (-10, 8, "right"),
    "K256": (8, 6, "left"),
    "true-K128": (8, -12, "left"),
}

for key in order:
    dx, dy, ha = label_offsets[key]

    ax.annotate(
        display[key],
        (data[key]["tokens_m"], data[key]["macro"]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=11.5,
        color=colors[key],
        fontweight="semibold",
    )

ax.set_xlabel("Processed scoring tokens, two models (millions)")
ax.set_ylabel("Macro score")

# Requested change 1: Bold title.
ax.set_title(
    "Scoring-cost reduction with minimal utility change",
    pad=12,
    fontweight="bold",
)

# Requested change 2: K128 in the information box.
textbox = "T1        14.51M\nK256    10.28M\nK128      7.70M"

ax.text(
    0.03,
    0.96,
    textbox,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10.4,
    bbox=dict(
        boxstyle="round,pad=0.38",
        facecolor="#F7F7F7",
        edgecolor="#B8B8B8",
        linewidth=1.0,
    ),
)

fig.tight_layout()

png_path = ROOT / "figs/scoring_cost_utility.png"

fig.savefig(png_path, dpi=300, facecolor="white")
plt.close(fig)

print("Saved:", png_path.relative_to(ROOT))
print("MAIN_FIGURE: PASS")
