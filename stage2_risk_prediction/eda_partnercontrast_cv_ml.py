# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""Per-behavior AUROC of within-visit partner contrast under ML substitutions.

Same protocol as eda_partnercontrast_cv.py (10-fold stratified CV, SMOTE+Tomek,
LR, asd_longitudinal label, 12-feature within-visit partner contrast set), but loads the
indicator-specific aggregate so within-visit partner contrasts are computed on the
ML-substituted source columns. Cohort is restricted to the same reference
filename set used in the G run, for direct row-by-row comparability with
Table 11.

Usage:
    python eda_partnercontrast_cv_ml.py FS
    python eda_partnercontrast_cv_ml.py FV
    python eda_partnercontrast_cv_ml.py FOSV
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              confusion_matrix)
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

AGG = "PATH/TO/FEATURES/aggregate/"
GT_CSV = "PATH/TO/FEATURES/aggregate/corrected_partnercontrast_source.csv"  # reference-cohort filename set
BEHAVIORS = ["smile_rate", "smile_prop",
             "lookface_rate", "lookface_prop",
             "lookobject_rate", "lookobject_prop",
             "vocal_rate", "vocal_prop",
             "social_smile_rate", "social_smile_prop",
             "social_vocal_rate", "social_vocal_prop"]
AGES = [24, 36]  # 18m excluded: insufficient ASD instances (n=9 < 10 folds)
N_FOLDS = 10
SEED = 0


def parse_session(fname):
    # Released naming: <sub_id>_<visit>_{Examiner,Parent1,Parent2,Parent}.csv
    if "Examiner" in fname: return "Examiner"
    if fname.endswith("_Parent1.csv"): return "Parent1"
    if fname.endswith("_Parent2.csv"): return "Parent2"
    if fname.endswith("_Parent.csv"): return "Parent"
    return "Unknown"


def build_mm(indicator):
    """Load hj_*_{ind}_*, restrict to the reference cohort, pivot to per-(sub,visit)."""
    gt = pd.read_csv(GT_CSV)
    cohort_files = set(gt["filename"].unique())

    df = pd.concat([
        pd.read_csv(AGG + f"hj_test_{indicator}_Aggregated.csv"),
        pd.read_csv(AGG + f"hj_asdtrain_{indicator}_Aggregated.csv"),
    ], ignore_index=True)
    df = df[df["filename"].isin(cohort_files)].copy()
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
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = {"auroc": [], "sens": [], "spec": []}
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smt", SMOTETomek(random_state=seed)),
            ("lr", LogisticRegression(max_iter=1000)),
        ])
        try:
            pipe.fit(Xtr, ytr)
        except ValueError:
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
        out["auroc"].append(roc_auc_score(yte, proba))
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        out["sens"].append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        out["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def ci_str(arr):
    a = np.asarray(arr); a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n/a"
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    return f"{m:.2f} [{m-1.96*se:.2f}, {m+1.96*se:.2f}]"


def main(indicator):
    mm = build_mm(indicator)
    print(f"\n=== Indicator: {indicator} (12-feature within-visit partner contrast, "
          f"reference cohort) ===")
    print(f"n={len(mm)} subject-visit instances, "
          f"{mm.asd_long.sum()} ASD")
    print(f"{'behavior':<22} {'age':>4} {'n_ASD':>5} {'n_Non':>5}  "
          f"{'AUROC':>22}  {'Delta':>10}")
    print("-" * 80)
    out = []
    for b in BEHAVIORS:
        for age in AGES:
            sub = mm[mm.visit == age].dropna(subset=[f"mm_{b}", "asd_long"])
            X = sub[[f"mm_{b}"]].values
            y = sub["asd_long"].values
            n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
            if n_pos < N_FOLDS:
                print(f"{b:<22} {age:>4} {n_pos:>5} {n_neg:>5}  (skipped)")
                continue
            res = cv_one_cell(X, y)
            delta = (sub[sub.asd_long == 1][f"mm_{b}"].mean()
                     - sub[sub.asd_long == 0][f"mm_{b}"].mean())
            print(f"{b:<22} {age:>4} {n_pos:>5} {n_neg:>5}  "
                  f"{ci_str(res['auroc']):>22}  {delta:+10.3f}")
            a = res["auroc"]
            a = a[~np.isnan(a)]
            m = a.mean()
            se = a.std(ddof=1) / np.sqrt(len(a))
            out.append({
                "indicator": indicator, "behavior": b, "age": age,
                "n_asd": n_pos, "n_non": n_neg,
                "auroc_mean": m, "auroc_lo": m - 1.96 * se,
                "auroc_hi": m + 1.96 * se,
                "delta": delta,
            })
    df = pd.DataFrame(out)
    df.to_csv(f"partnercontrast_cv_results_{indicator}.csv", index=False)
    print(f"\nsaved partnercontrast_cv_results_{indicator}.csv")


if __name__ == "__main__":
    main(sys.argv[1])
