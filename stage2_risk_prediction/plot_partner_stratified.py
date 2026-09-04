# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""Per-behavior longitudinal plots of partner-stratified means with
individual data points overlaid.

For each of 12 behaviors, produces a plot with:
  - X-axis: age (18, 24, 36 months)
  - Y-axis: behavior value (rate events/min or proportion)
  - 4 curves: ASD-Parent, ASD-Examiner, Non-ASD-Parent, Non-ASD-Examiner
    (color = group, line style = partner)
  - Shaded band: 95% CI of the mean (mean +/- 1.96*SE) in matching color

Aggregation matches the paper's per-(subject, visit) convention:
  parent value = mean across the parent sessions (Parent1, Parent2, Parent)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "PATH/TO/FEATURES/aggregate/corrected_partnercontrast_source.csv"
OUTDIR = "partner_stratified_plots"
BEHAVIORS = [
    "smile_rate", "smile_prop",
    "lookface_rate", "lookface_prop",
    "lookobject_rate", "lookobject_prop",
    "vocal_rate", "vocal_prop",
    "social_smile_rate", "social_smile_prop",
    "social_vocal_rate", "social_vocal_prop",
]
AGES = [18, 24, 36]
COLOR_ASD = "tab:red"
COLOR_NON = "tab:blue"


def parse_partner(fn):
    if "Examiner" in fn:
        return "Examiner"
    if "Parent" in fn:
        return "Parent"
    return None


def units_label(b):
    return "events/min" if b.endswith("_rate") else "proportion"


def pretty(b):
    return b.replace("_rate", " (rate, events/min)").replace(
        "_prop", " (prop)").replace("_", " ")


def aggregate(df, b):
    """Per (sub_id, visit, partner): mean behavior value + ASD label.
    Multiple parent sessions within a visit are averaged."""
    sub = df.dropna(subset=[b, "asd"]).copy()
    return (sub.groupby(["sub_id", "visit", "partner"], as_index=False)
            .agg(val=(b, "mean"), asd=("asd", "first")))


def plot_one(ax, agg, b):
    """Render one behavior's plot into the given axes."""
    style = {
        ("ASD", "Parent"): dict(color=COLOR_ASD, ls="-", marker="o"),
        ("ASD", "Examiner"): dict(color=COLOR_ASD, ls="--", marker="s"),
        ("Non-ASD", "Parent"): dict(color=COLOR_NON, ls="-", marker="o"),
        ("Non-ASD", "Examiner"): dict(color=COLOR_NON, ls="--", marker="s"),
    }
    rng = np.random.default_rng(42)
    for (group_name, partner), st in style.items():
        asd_flag = 1 if group_name == "ASD" else 0
        cell = agg[(agg.asd == asd_flag) & (agg.partner == partner)
                   & (agg.visit.isin(AGES))]
        # Per-age mean and 95% CI of the mean (mean +/- 1.96*SE)
        stats = cell.groupby("visit")["val"].agg(["mean", "std", "count"]).reindex(AGES)
        means = stats["mean"]
        se = stats["std"] / np.sqrt(stats["count"])
        ci = 1.96 * se
        lo = (means - ci).to_numpy(dtype=float)
        hi = (means + ci).to_numpy(dtype=float)
        ax.plot(AGES, means.to_numpy(dtype=float), color=st["color"], ls=st["ls"],
                marker=st["marker"], markersize=9, linewidth=2,
                label=f"{group_name}–{partner}", zorder=4)
        # Shaded 95% CI band around the mean curve
        ax.fill_between(AGES, lo, hi, color=st["color"], alpha=0.15,
                        linewidth=0, zorder=2)
        # # Scatter individuals (small jitter on age for visibility)
        # if len(cell) > 0:
        #     jitter = rng.uniform(-0.6, 0.6, len(cell))
        #     ax.scatter(cell["visit"] + jitter, cell["val"],
        #                color=st["color"], alpha=0.25, s=22,
        #                marker=st["marker"], edgecolors="none", zorder=2)
    ax.set_xticks(AGES)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(units_label(b))
    ax.set_title(pretty(b))
    ax.grid(alpha=0.25, zorder=1)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)


def main():
    df = pd.read_csv(CSV)
    df["partner"] = df["filename"].apply(parse_partner)
    df = df[df["partner"].isin(["Parent", "Examiner"])]
    df["asd"] = (df["asd_longitudinal"] == "ASD").astype(int)

    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Individual PNGs
    for b in BEHAVIORS:
        agg = aggregate(df, b)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_one(ax, agg, b)
        plt.tight_layout()
        out = f"{OUTDIR}/{b}.png"
        plt.savefig(out, dpi=130)
        plt.close()
        n_asd = ((agg.asd == 1) & (agg.visit.isin(AGES))).sum()
        n_non = ((agg.asd == 0) & (agg.visit.isin(AGES))).sum()
        print(f"  {b:<22}  n_ASD={n_asd:>3}  n_Non={n_non:>4}  →  {out}")

    # 2) Multipanel grid
    fig, axes = plt.subplots(3, 4, figsize=(26, 16))
    for ax, b in zip(axes.flat, BEHAVIORS):
        agg = aggregate(df, b)
        plot_one(ax, agg, b)
    plt.tight_layout()
    grid_out = f"{OUTDIR}/grid_all_behaviors.png"
    plt.savefig(grid_out, dpi=120)
    plt.close()
    print(f"\nMultipanel grid → {grid_out}")
    print(f"\nAll outputs in: {OUTDIR}/")


if __name__ == "__main__":
    main()
