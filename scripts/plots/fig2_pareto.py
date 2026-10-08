import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import save, setup

setup()

ROOT = Path(__file__).resolve().parents[2]
COST_CSV = ROOT / "results" / "cost" / "final_cost_comparison.csv"

measured = []
with COST_CSV.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        measured.append(
            {
                "name": row["method"],
                "energy": float(row["gross_gpu_kwh"]),
                "macro": float(row["macro"]),
            }
        )

# Full20K was not measured to completion. Keep it visually and
# semantically separate from the measured frontier.
full20k_estimate = {
    "name": "Full20K (Estimated)",
    "energy": 1.902000,
    "macro": 43.2700,
}

ann = {
    "T1": {"xytext": (10, 12), "ha": "left", "va": "bottom"},
    "K256": {"xytext": (-10, 14), "ha": "right", "va": "bottom"},
    "true-K128": {"xytext": (-10, -16), "ha": "right", "va": "top"},
}

fig, ax = plt.subplots(figsize=(8.8, 5.6))

xs = [d["energy"] for d in measured]
ys = [d["macro"] for d in measured]

# The line connects measured methods only.
ax.plot(xs, ys, marker="o", linewidth=1.8, label="Measured E2E GPU energy")

for d in measured:
    a = ann[d["name"]]
    ax.annotate(
        f'{d["name"]}\nMacro {d["macro"]:.2f}\n{d["energy"]:.3f} kWh',
        (d["energy"], d["macro"]),
        xytext=a["xytext"],
        textcoords="offset points",
        ha=a["ha"],
        va=a["va"],
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=0, shrinkB=4),
    )

ax.scatter(
    [full20k_estimate["energy"]],
    [full20k_estimate["macro"]],
    marker="X",
    s=90,
    facecolors="none",
    edgecolors="black",
    linewidths=1.2,
    label="Full20K energy estimate",
)
ax.annotate(
    "Full20K\nMacro 43.27\n1.902 kWh (Estimated)",
    (full20k_estimate["energy"], full20k_estimate["macro"]),
    xytext=(-10, 12),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
    arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=0, shrinkB=4),
)

ax.axhline(46.6, linestyle="--", linewidth=1.0)
ax.text(
    1.30,
    46.615,
    "Frozen utility threshold = 46.60",
    fontsize=9,
    va="bottom",
)

ax.annotate(
    "Preferred direction\n(higher score, lower measured energy)",
    xy=(0.62, 46.98),
    xytext=(1.18, 46.18),
    fontsize=9,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", lw=1.0),
)

ax.set_xlabel("Matched E2E GPU Energy (kWh)")
ax.set_ylabel("Macro Score")
ax.set_title("Performance-Energy Trade-off")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, loc="lower left")

ax.set_xlim(0.50, 2.05)
ax.set_ylim(43.0, 47.3)

fig.text(
    0.5,
    0.01,
    "E2E scope: calibration + candidate scoring + final 2K fine-tuning; "
    "official downstream evaluation excluded. Full20K energy is estimated.",
    ha="center",
    fontsize=8,
)

save(fig, "figs/performance_energy_pareto.png")
print("Saved: figs/performance_energy_pareto.png")
