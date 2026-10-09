import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from common import METHOD_COLORS, save, setup

setup()

root = Path(__file__).resolve().parents[2]
with (root / "results/cost/final_cost_comparison.csv").open(newline="") as stream:
    cost = {row["method"]: row for row in csv.DictReader(stream)}

data = {
    method: {
        "macro": json.loads(
            (root / f"results/performance/{method.lower()}_summary.json").read_text()
        )["macro"],
        "energy": float(cost[method]["gross_gpu_kwh"]),
    }
    for method in ("T1", "K256", "K128")
}

base_macro = json.loads(
    (root / "results/performance/base_summary.json").read_text()
)["macro"]
t1_macro = data["T1"]["macro"]
t1_energy = data["T1"]["energy"]

labels = list(data.keys())
values = []
for name in labels:
    d = data[name]
    sue = ((d["macro"] - base_macro) / (t1_macro - base_macro)) / (
        d["energy"] / t1_energy
    )
    values.append(sue)

fig, ax = plt.subplots(figsize=(8.2, 5.4))
x = list(range(len(labels)))

ax.plot(
    x,
    values,
    linewidth=1.5,
    color="#BEC5CC",
    zorder=1,
)

for i, method in enumerate(labels):
    ax.scatter(
        i,
        values[i],
        s=95,
        color=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

# 겹침 방지를 위해 라벨을 바깥쪽으로 강하게 분리
ann = {
    "T1": {"xytext": (-42, 18), "ha": "right", "va": "bottom"},
    "K256": {"xytext": (0, -28), "ha": "center", "va": "top"},
    "K128": {"xytext": (42, 18), "ha": "left", "va": "bottom"},
}

for i, (label, v) in enumerate(zip(labels, values)):
    a = ann[label]
    ax.annotate(
        f"{label}\n{v:.3f}",
        (i, v),
        xytext=a["xytext"],
        textcoords="offset points",
        ha=a["ha"],
        va=a["va"],
        fontsize=9,
        color=METHOD_COLORS[label],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=0, shrinkB=4),
    )

ax.axhline(1.0, linestyle="--", linewidth=1.0)
ax.text(-0.35, 1.003, "T1 baseline = 1.00", fontsize=9, va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("SUE")
ax.set_xlabel("Method")
ax.set_title("Sustainable Utility Efficiency", fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)

# 여백을 좀 더 넉넉하게
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(min(values) - 0.015, max(values) + 0.04)

save(fig, "figs/sue.png")
print("Saved: figs/sue.png")
