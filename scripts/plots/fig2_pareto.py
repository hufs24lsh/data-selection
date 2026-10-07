import matplotlib.pyplot as plt
from common import setup, save

setup()

data = [
    {"name": "Full20K",   "energy": 1.902000, "macro": 43.2700},
    {"name": "T1",        "energy": 0.682982, "macro": 47.1000},
    {"name": "K256",      "energy": 0.644048, "macro": 46.6894},
    {"name": "true-K128", "energy": 0.605340, "macro": 46.6057},
]

fig, ax = plt.subplots(figsize=(7.6, 5.4))

xs = [d["energy"] for d in data]
ys = [d["macro"] for d in data]

# 연결선은 과하지 않게
ax.plot(xs, ys, linewidth=1.2, alpha=0.7)
ax.scatter(xs, ys, s=70)

offsets = {
    "Full20K":   (8, -2),
    "T1":        (10, 10),
    "K256":      (10, -16),
    "true-K128": (-78, -2),
}

for d in data:
    dx, dy = offsets[d["name"]]
    ax.annotate(
        f'{d["name"]}\nMacro {d["macro"]:.2f}\n{d["energy"]:.3f} kWh',
        (d["energy"], d["macro"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", lw=0.8)
    )

ax.set_xlabel("End-to-End GPU Energy (kWh)")
ax.set_ylabel("Macro Score")
ax.set_title("Performance-Energy Trade-off")
ax.grid(True, alpha=0.25)

# 너무 과장되지 않게 y축 범위를 넉넉하게
ax.set_xlim(0.50, 2.05)
ax.set_ylim(42.0, 47.8)

save(fig, "figs/performance_energy_pareto.png")
print("Saved: figs/performance_energy_pareto.png")
