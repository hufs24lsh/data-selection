import matplotlib.pyplot as plt
from common import setup, save

setup()

# Public release values
data = [
    {"name": "Full20K",    "energy": 1.902000, "macro": 43.2700},
    {"name": "T1",         "energy": 0.682982, "macro": 47.1000},
    {"name": "K256",       "energy": 0.644048, "macro": 46.6894},
    {"name": "true-K128",  "energy": 0.605340, "macro": 46.6057},
]

# Non-overlapping annotation layout
ann = {
    "Full20K":   {"xytext": (10, 10),   "ha": "left",  "va": "bottom"},
    "T1":        {"xytext": (10, 12),   "ha": "left",  "va": "bottom"},
    "K256":      {"xytext": (-10, 14),  "ha": "right", "va": "bottom"},
    "true-K128": {"xytext": (-10, -16), "ha": "right", "va": "top"},
}

fig, ax = plt.subplots(figsize=(8.8, 5.6))

xs = [d["energy"] for d in data]
ys = [d["macro"] for d in data]

# Connecting line
ax.plot(xs, ys, marker="o", linewidth=1.8)

# Annotations
for d in data:
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

# Highlight ShallowFrontier region
ax.axhspan(46.55, 47.15, alpha=0.08)
ax.text(
    1.28, 47.11,
    "High-performance region",
    fontsize=9,
    va="top"
)

# Arrow showing desirable direction
ax.annotate(
    "Preferred direction\n(higher score, lower energy)",
    xy=(0.62, 46.98),
    xytext=(1.18, 46.18),
    fontsize=9,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", lw=1.0),
)

# Target line
ax.axhline(46.6, linestyle="--", linewidth=1.0)
ax.text(
    1.30, 46.615,
    "Utility threshold = 46.6",
    fontsize=9,
    va="bottom"
)

ax.set_xlabel("End-to-End GPU Energy (kWh)")
ax.set_ylabel("Macro Score")
ax.set_title("Performance-Energy Trade-off")
ax.grid(True, alpha=0.25)

# 핵심: x축은 넓게, y축은 촘촘하게
ax.set_xlim(0.50, 2.05)
ax.set_ylim(43.0, 47.3)

save(fig, "figs/performance_energy_pareto.png")
print("Saved: figs/performance_energy_pareto.png")
