import csv
from pathlib import Path

import matplotlib.pyplot as plt
from common import DISPLAY_NAME, METHOD_COLORS, save, setup

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
    "K128": {"xytext": (-10, -16), "ha": "right", "va": "top"},
}

fig, ax = plt.subplots(figsize=(8.8, 5.6))

xs = [d["energy"] for d in measured]
ys = [d["macro"] for d in measured]

# The line connects measured methods only.
ax.plot(
    xs,
    ys,
    linewidth=1.8,
    color="#A8B2BB",
    zorder=1,
    label="Measured E2E GPU energy",
)

# Connect T1 to Full20K visually, while distinguishing the estimate.
t1_point = next(d for d in measured if d["name"] == "T1")

ax.plot(
    [t1_point["energy"], full20k_estimate["energy"]],
    [t1_point["macro"], full20k_estimate["macro"]],
    linestyle="--",
    linewidth=1.8,
    color="#A8B2BB",
    zorder=1,
    label="To Full20K (estimated)",
)

# Show the three measured methods in the shared project colors.
for d in measured:
    method = DISPLAY_NAME[d["name"]]
    ax.scatter(
        d["energy"],
        d["macro"],
        s=90,
        color=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

for d in measured:
    a = ann[d["name"]]
    ax.annotate(
        f"{DISPLAY_NAME[d['name']]}\nMacro {d['macro']:.2f}\n{d['energy']:.3f} kWh",
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
    marker="o",
    s=90,
    facecolors="#8A929B",
    edgecolors="white",
    linewidths=1.2,
    label="_nolegend_",
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

ax.axhspan(
    46.6,
    47.2,
    color=METHOD_COLORS["K256"],
    alpha=0.055,
    zorder=0,
)
ax.text(
    1.30,
    47.12,
    "High-performance region",
    fontsize=9,
    va="top",
    color="#505F6B",
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
ax.set_title("Performance-Energy Trade-off", fontweight="bold")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, loc="lower left")

ax.set_xlim(0.50, 2.05)
ax.set_ylim(43.0, 47.3)


save(fig, "figs/performance_energy_pareto.png")
print("Saved: figs/performance_energy_pareto.png")
