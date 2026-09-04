# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""Plot component-validation sweep (Table 10) — one figure per metric."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- data -----------------------------------------------------------
configs = [
    "Look Face from ML",
    "Look Object from ML",
    "Smile from ML",
    "Vocal from ML",
    "Look Face + Look Object",
    "Look Face + Smile",
    "Look Face + Vocal",
    "Look Object + Smile",
    "Look Object + Vocal",
    "Smile + Vocal",
    "LF + LO + Smile",
    "LF + LO + Vocal",
    "LF + Smile + Vocal",
    "LO + Smile + Vocal",
    "All behaviors from ML",
    "Ground Truth",
]

# (mean, lo_ci, hi_ci)
data = {
    "Sensitivity (%)": [
        (76.3, 67.7, 84.9),
        (80.9, 75.0, 86.9),
        (80.7, 74.6, 86.9),
        (81.5, 74.8, 88.2),
        (77.0, 68.6, 85.5),
        (81.0, 71.5, 90.4),
        (74.5, 66.2, 82.9),
        (84.1, 77.1, 91.1),
        (80.1, 72.5, 87.7),
        (82.1, 71.1, 93.1),
        (78.1, 69.2, 87.0),
        (73.3, 65.2, 81.4),
        (78.9, 68.2, 89.6),
        (81.8, 72.5, 91.2),
        (69.4, 60.7, 78.1),
        (80.9, 75.2, 86.6),
    ],
    "Specificity (%)": [
        (79.9, 77.4, 82.4),
        (80.1, 77.7, 82.5),
        (79.7, 77.1, 82.2),
        (79.8, 76.9, 82.7),
        (79.6, 77.1, 82.1),
        (78.4, 76.0, 80.9),
        (79.1, 76.9, 81.3),
        (78.1, 75.0, 81.1),
        (79.1, 75.9, 82.3),
        (77.2, 74.2, 80.2),
        (75.8, 73.3, 78.4),
        (76.9, 73.5, 80.3),
        (77.9, 75.0, 80.9),
        (76.5, 72.7, 80.3),
        (76.8, 72.8, 80.7),
        (81.3, 79.0, 83.6),
    ],
    "PPV (%)": [
        (23.5, 19.8, 27.2),
        (24.7, 21.8, 27.5),
        (24.4, 21.5, 27.4),
        (24.9, 21.7, 28.1),
        (23.6, 19.6, 27.7),
        (23.3, 19.3, 27.3),
        (22.5, 19.2, 25.7),
        (23.7, 20.6, 26.8),
        (24.2, 20.4, 27.9),
        (22.8, 19.5, 26.1),
        (20.5, 18.3, 22.8),
        (20.9, 17.2, 24.7),
        (22.6, 19.0, 26.2),
        (22.5, 19.1, 26.0),
        (19.8, 16.9, 22.8),
        (26.0, 22.7, 29.3),
    ],
    "AUROC": [
        (0.87, 0.83, 0.90),
        (0.88, 0.85, 0.91),
        (0.88, 0.85, 0.90),
        (0.87, 0.84, 0.90),
        (0.85, 0.81, 0.89),
        (0.86, 0.83, 0.90),
        (0.86, 0.82, 0.89),
        (0.87, 0.84, 0.90),
        (0.87, 0.83, 0.90),
        (0.87, 0.84, 0.89),
        (0.85, 0.82, 0.89),
        (0.83, 0.78, 0.88),
        (0.85, 0.81, 0.89),
        (0.85, 0.82, 0.89),
        (0.82, 0.78, 0.87),
        (0.88, 0.86, 0.90),
    ],
    "Accuracy (%)": [
        (79.7, 77.2, 82.1),
        (80.2, 78.1, 82.3),
        (79.8, 77.5, 82.0),
        (79.9, 77.2, 82.6),
        (79.5, 76.8, 82.2),
        (78.6, 76.0, 81.2),
        (78.7, 76.5, 80.9),
        (78.5, 75.8, 81.1),
        (79.2, 76.2, 82.2),
        (77.6, 74.8, 80.4),
        (76.0, 73.9, 78.1),
        (76.6, 73.3, 80.0),
        (78.0, 75.2, 80.8),
        (76.9, 73.5, 80.3),
        (76.2, 72.8, 79.6),
        (81.3, 79.2, 83.4),
    ],
}

# Group boundaries (for background shading)
# 0-3: single, 4-9: dual, 10-13: triple, 14: all-ML, 15: GT
group_spans = [
    (0, 3,  "#e8f4f8", "Single channel"),
    (4, 9,  "#fef9e7", "Dual channel"),
    (10, 13, "#fdf2f8", "Triple channel"),
    (14, 14, "#fdfefe", "All ML"),
    (15, 15, "#eafaf1", "Ground Truth"),
]

n = len(configs)
y = np.arange(n)

for metric, vals in data.items():
    means  = np.array([v[0] for v in vals])
    lo_ci  = np.array([v[1] for v in vals])
    hi_ci  = np.array([v[2] for v in vals])
    xerr_lo = means - lo_ci
    xerr_hi = hi_ci - means

    fig, ax = plt.subplots(figsize=(9, 7))

    # Group shading
    for (i0, i1, color, label) in group_spans:
        ax.axhspan(i0 - 0.5, i1 + 0.5, color=color, alpha=0.6, zorder=0)

    # Ground Truth reference line
    gt_mean = means[-1]
    ax.axvline(gt_mean, color="#27ae60", linewidth=1.4,
               linestyle="--", alpha=0.8, zorder=1, label=f"GT = {gt_mean}")

    # Colors: GT green, All-ML orange-red, others steel blue
    point_colors = []
    for i, cfg in enumerate(configs):
        if cfg == "Ground Truth":
            point_colors.append("#27ae60")
        elif cfg == "All behaviors from ML":
            point_colors.append("#e74c3c")
        else:
            point_colors.append("#2980b9")

    for i in range(n):
        ax.errorbar(
            means[i], y[i],
            xerr=[[xerr_lo[i]], [xerr_hi[i]]],
            fmt="none",
            ecolor=point_colors[i],
            elinewidth=1.2, capsize=3, alpha=0.7, zorder=2,
        )
    ax.scatter(means, y, c=point_colors, s=55, zorder=3,
               edgecolors="white", linewidths=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(configs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f"Component-validation sweep — {metric}", fontsize=12, pad=10)
    ax.grid(axis="x", alpha=0.3, zorder=0)

    # Legend patches
    legend_handles = [
        mpatches.Patch(color="#27ae60", label="Ground Truth"),
        mpatches.Patch(color="#e74c3c", label="All behaviors from ML"),
        mpatches.Patch(color="#2980b9", label="Partial substitution"),
        mpatches.Patch(color="#e8f4f8", label="Single channel"),
        mpatches.Patch(color="#fef9e7", label="Dual channel"),
        mpatches.Patch(color="#fdf2f8", label="Triple channel"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              framealpha=0.9)

    metric_slug = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
    fname = f"Figures/sweep_{metric_slug}.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"saved {fname}")
    plt.close(fig)
