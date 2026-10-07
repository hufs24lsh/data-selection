import matplotlib.pyplot as plt
from common import setup, save

setup()

data = [
    {"name": "T1",        "energy": 0.682982, "macro": 47.1000},
    {"name": "K256",      "energy": 0.644048, "macro": 46.6894},
    {"name": "true-K128", "energy": 0.605340, "macro": 46.6057},
]

# 수동 배치: 겹치지 않게
offsets = {
    "T1":        (8, 10),
    "K256":      (8, -16),
    "true-K128": (-70, -6),
}

fig, ax = plt.subplots(figsize=(7.4, 5.4))

xs = [d["energy"] for d in data]
ys = [d["macro"] for d in data]

ax.plot(xs, ys, marker="o", linewidth=1.6)

for d in data:
    dx, dy = offsets[d["name"]]
    ax.annotate(
        f'{d["name"]}\nMacro={d["macro"]:.2f}\n{d["energy"]:.3f} kWh',
        (d["energy"], d["macro"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", lw=0.8)
    )

ax.axhline(46.6, linestyle="--", linewidth=1.0)
ax.text(
    min(xs) + 0.002, 46.6 + 0.015,
    "Target macro = 46.6",
    fontsize=9, va="bottom"
)

ax.set_xlabel("End-to-End GPU Energy (kWh)")
ax.set_ylabel("Macro Score")
ax.set_title("Performance–Energy Trade-off")
ax.grid(True, alpha=0.25)

ax.set_xlim(0.595, 0.695)
ax.set_ylim(46.50, 47.18)

save(fig, "report_figures/out/figure2_pareto.png")
print("Saved: report_figures/out/figure2_pareto.png")
