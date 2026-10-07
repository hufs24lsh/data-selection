import matplotlib.pyplot as plt
from common import setup, save

setup()

data = [
    {"name": "T1",              "energy": 0.682982, "macro": 47.1000, "estimated": False},
    {"name": "K256",            "energy": 0.644048, "macro": 46.6894, "estimated": False},
    {"name": "true-K128",       "energy": 0.605340, "macro": 46.6057, "estimated": False},
    {"name": "Full20K (est.)",  "energy": 1.902000, "macro": 43.2700, "estimated": True},
]

offsets = {
    "T1":             (8, 10),
    "K256":           (8, -16),
    "true-K128":      (8, 12),
    "Full20K (est.)": (-105, 18),
}

fig, ax = plt.subplots(figsize=(8.0, 5.5))

measured = [d for d in data if not d["estimated"]]
ax.plot(
    [d["energy"] for d in measured],
    [d["macro"] for d in measured],
    marker="o",
    linewidth=1.6,
    label="Measured E2E energy",
)

full = next(d for d in data if d["estimated"])
ax.scatter(
    [full["energy"]],
    [full["macro"]],
    marker="s",
    s=55,
    label="Estimated Full20K energy",
)

for d in data:
    dx, dy = offsets[d["name"]]
    energy_label = f'{d["energy"]:.3f} kWh'
    if d["estimated"]:
        energy_label += " (estimated)"

    ax.annotate(
        f'{d["name"]}\nMacro={d["macro"]:.2f}\n{energy_label}',
        (d["energy"], d["macro"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
        arrowprops=dict(arrowstyle="-", lw=0.8),
    )

ax.axhline(46.6, linestyle="--", linewidth=1.0)
ax.text(
    0.575, 46.63,
    "Utility threshold = 46.60",
    fontsize=9,
    va="bottom",
)

ax.set_xlabel("End-to-End GPU Energy (kWh)")
ax.set_ylabel("Macro Score")
ax.set_title("Performance–Energy Trade-off")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True)

ax.set_xlim(0.55, 2.05)
ax.set_ylim(42.8, 47.4)

save(fig, "figs/performance_energy_pareto.png")
print("Saved: figs/performance_energy_pareto.png")
