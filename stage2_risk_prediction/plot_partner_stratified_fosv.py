# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""Partner-stratified mean behavior plots for fully ML-estimated (FOSV)
behaviors — the ML counterpart of plot_partner_stratified.py (which uses
ground-truth). Reads the hj_{test,asdtrain}_FOSV aggregate (all behaviors
from ML), restricts to the same matched reference cohort, and produces the
same 3x4 grid: 4 curves per panel (ASD/Non-ASD x Parent/Examiner) with
95% CI bands.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AGG = "PATH/TO/FEATURES/aggregate/"
GT_CSV = "PATH/TO/FEATURES/aggregate/corrected_partnercontrast_source.csv"  # reference-cohort filename set
OUTDIR = "partner_stratified_plots_fosv"
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
    sub = df.dropna(subset=[b, "asd"]).copy()
    return (sub.groupby(["sub_id", "visit", "partner"], as_index=False)
            .agg(val=(b, "mean"), asd=("asd", "first")))


def plot_one(ax, agg, b):
    style = {
        ("ASD", "Parent"): dict(color=COLOR_ASD, ls="-", marker="o"),
        ("ASD", "Examiner"): dict(color=COLOR_ASD, ls="--", marker="s"),
        ("Non-ASD", "Parent"): dict(color=COLOR_NON, ls="-", marker="o"),
        ("Non-ASD", "Examiner"): dict(color=COLOR_NON, ls="--", marker="s"),
    }
    for (group_name, partner), st in style.items():
        asd_flag = 1 if group_name == "ASD" else 0
        cell = agg[(agg.asd == asd_flag) & (agg.partner == partner)
                   & (agg.visit.isin(AGES))]
        stats = cell.groupby("visit")["val"].agg(["mean", "std", "count"]).reindex(AGES)
        means = stats["mean"]
        se = stats["std"] / np.sqrt(stats["count"])
        ci = 1.96 * se
        lo = (means - ci).to_numpy(dtype=float)
        hi = (means + ci).to_numpy(dtype=float)
        ax.plot(AGES, means.to_numpy(dtype=float), color=st["color"], ls=st["ls"],
                marker=st["marker"], markersize=9, linewidth=2,
                label=f"{group_name}-{partner}", zorder=4)
        ax.fill_between(AGES, lo, hi, color=st["color"], alpha=0.15,
                        linewidth=0, zorder=2)
    ax.set_xticks(AGES)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(units_label(b))
    ax.set_title(pretty(b))
    ax.grid(alpha=0.25, zorder=1)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)


def main():
    # reference-cohort filename set (same cohort as the GT plot)
    cohort_files = set(pd.read_csv(GT_CSV)["filename"].unique())
    df = pd.concat([
        pd.read_csv(AGG + "hj_test_FOSV_Aggregated.csv"),
        pd.read_csv(AGG + "hj_asdtrain_FOSV_Aggregated.csv"),
    ], ignore_index=True)
    df = df[df["filename"].isin(cohort_files)].copy()
    df["partner"] = df["filename"].apply(parse_partner)
    df = df[df["partner"].isin(["Parent", "Examiner"])]
    df["asd"] = (df["asd_longitudinal"] == "ASD").astype(int)

    os.makedirs(OUTDIR, exist_ok=True)
    for b in BEHAVIORS:
        agg = aggregate(df, b)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_one(ax, agg, b)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{b}.png", dpi=130)
        plt.close()

    fig, axes = plt.subplots(3, 4, figsize=(26, 16))
    for ax, b in zip(axes.flat, BEHAVIORS):
        plot_one(ax, aggregate(df, b), b)
    plt.tight_layout()
    grid_out = f"{OUTDIR}/grid_all_behaviors_fosv.png"
    plt.savefig(grid_out, dpi=120)
    plt.close()
    print(f"FOSV grid -> {grid_out}")
    print(f"cohort: {df['sub_id'].nunique()} subjects, "
          f"{df[df.asd==1]['sub_id'].nunique()} ASD")


if __name__ == "__main__":
    main()
