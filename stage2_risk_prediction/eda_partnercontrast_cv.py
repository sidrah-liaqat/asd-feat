# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""Stratified 10-fold cross-validation for within-visit partner contrast as a
single-feature ASD-risk score under asd_longitudinal labeling, matching the
paper's Section 4.3.4 evaluation convention (SMOTE+Tomek imbalance handling,
95% CIs derived from cross-validation folds).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              precision_score, recall_score, confusion_matrix)
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

CSV = "PATH/TO/FEATURES/aggregate/corrected_partnercontrast_source.csv"
BEHAVIORS = ["smile_rate", "smile_prop",
             "lookface_rate", "lookface_prop",
             "lookobject_rate", "lookobject_prop",
             "vocal_rate", "vocal_prop",
             "social_smile_rate", "social_smile_prop",
             "social_vocal_rate", "social_vocal_prop"]
AGES = [18, 24, 36]
N_FOLDS = 10
SEED = 0


def parse_session(fname):
    # Released naming: <sub_id>_<visit>_{Examiner,Parent1,Parent2,Parent}.csv
    if "Examiner" in fname: return "Examiner"
    if fname.endswith("_Parent1.csv"): return "Parent1"
    if fname.endswith("_Parent2.csv"): return "Parent2"
    if fname.endswith("_Parent.csv"): return "Parent"
    return "Unknown"


def build_mm():
    df = pd.read_csv(CSV)
    df["session"] = df["filename"].apply(parse_session)
    rows = []
    for (sid, visit), g in df.groupby(["sub_id", "visit"]):
        ex = g[g.session == "Examiner"]
        par = g[g.session.isin(["Parent1", "Parent2"])]
        if len(ex) == 0 or len(par) == 0:
            continue
        rec = {"sub_id": sid, "visit": int(visit),
               "asd_long": 1 if g["asd_longitudinal"].iloc[0] == "ASD" else 0}
        for b in BEHAVIORS:
            rec[f"mm_{b}"] = par[b].mean() - ex[b].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def cv_one_cell(X, y, n_folds=N_FOLDS, seed=SEED):
    """Stratified k-fold CV. SMOTE+Tomek applied on train fold only.
    Returns per-fold AUROC, sens, spec, PPV, accuracy."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    metrics = {"auroc": [], "sens": [], "spec": [], "ppv": [], "acc": []}
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # SMOTE+Tomek + LR on train; LR on raw 1-D feature
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smt", SMOTETomek(random_state=seed)),
            ("lr", LogisticRegression(max_iter=1000)),
        ])
        try:
            pipe.fit(Xtr, ytr)
        except ValueError:
            # SMOTE may fail if minority class has too few samples in a fold;
            # fall back to class_weight balanced LR.
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(class_weight="balanced",
                                          max_iter=1000)),
            ])
            pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        if len(np.unique(yte)) < 2:
            continue
        metrics["auroc"].append(roc_auc_score(yte, proba))
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        acc = accuracy_score(yte, pred)
        metrics["sens"].append(sens)
        metrics["spec"].append(spec)
        metrics["ppv"].append(ppv)
        metrics["acc"].append(acc)
    return {k: np.array(v) for k, v in metrics.items()}


def fmt(arr):
    a = np.asarray(arr)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n/a"
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    lo, hi = m - 1.96 * se, m + 1.96 * se
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def main():
    mm = build_mm()
    print(f"{'behavior':<22} {'age':>4} {'n_ASD':>5} {'n_Non':>5}  "
          f"{'CV AUROC':>20}  {'Sens':>20}  {'Spec':>20}")
    print("-" * 110)

    cell_results = []
    for b in BEHAVIORS:
        for age in AGES:
            sub = mm[mm.visit == age].dropna(subset=[f"mm_{b}", "asd_long"])
            X = sub[[f"mm_{b}"]].values
            y = sub["asd_long"].values
            n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
            if n_pos < N_FOLDS:
                # Need at least one minority sample per fold
                print(f"{b:<22} {age:>4} {n_pos:>5} {n_neg:>5}  "
                      f"(skipped: {n_pos} ASD < {N_FOLDS} folds)")
                cell_results.append({"behavior": b, "age": age,
                                     "skipped": True})
                continue
            res = cv_one_cell(X, y)
            print(f"{b:<22} {age:>4} {n_pos:>5} {n_neg:>5}  "
                  f"{fmt(res['auroc']):>20}  "
                  f"{fmt(res['sens']):>20}  {fmt(res['spec']):>20}")
            cell_results.append({
                "behavior": b, "age": age, "skipped": False,
                "n_asd": n_pos, "n_non": n_neg,
                "auroc_mean": np.mean(res["auroc"]),
                "auroc_lo": np.mean(res["auroc"]) - 1.96 * np.std(res["auroc"], ddof=1)/np.sqrt(len(res["auroc"])),
                "auroc_hi": np.mean(res["auroc"]) + 1.96 * np.std(res["auroc"], ddof=1)/np.sqrt(len(res["auroc"])),
                "sens_mean": np.nanmean(res["sens"]),
                "spec_mean": np.nanmean(res["spec"]),
                "ppv_mean": np.nanmean(res["ppv"]),
                "acc_mean": np.nanmean(res["acc"]),
                "all_auroc": res["auroc"].tolist(),
                "all_sens": res["sens"].tolist(),
                "all_spec": res["spec"].tolist(),
                "all_ppv": res["ppv"].tolist(),
                "all_acc": res["acc"].tolist(),
            })
    df = pd.DataFrame(cell_results)
    df.to_csv("partnercontrast_cv_results.csv", index=False)
    print("\nsaved partnercontrast_cv_results.csv")

    # Detailed metrics for the headline cells
    print("\n" + "=" * 80)
    print("HEADLINE CELLS — full metrics (paper Tables 9/10 format)")
    print("=" * 80)
    headline = [("lookface_rate", 24), ("lookface_rate", 36),
                ("social_smile_prop", 24), ("social_smile_prop", 36),
                ("social_vocal_rate", 24), ("social_vocal_rate", 36),
                ("social_vocal_prop", 36)]
    print(f"  {'cell':<28} {'AUROC':>16} {'Sens':>16} {'Spec':>16} "
          f"{'PPV':>16} {'Acc':>16}")
    for b, age in headline:
        r = next(x for x in cell_results
                 if x["behavior"] == b and x["age"] == age and not x["skipped"])
        cell_name = f"{b} @ {age}m"
        print(f"  {cell_name:<28} {fmt(r['all_auroc']):>16} "
              f"{fmt(r['all_sens']):>16} {fmt(r['all_spec']):>16} "
              f"{fmt(r['all_ppv']):>16} {fmt(r['all_acc']):>16}")


if __name__ == "__main__":
    main()
