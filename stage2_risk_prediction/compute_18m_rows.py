# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""Compute 18m per-behavior within-visit partner contrast AUROC [CI] and Delta for GT (G) and
all-ML (FOSV), to extend Tables 11 and 13 with an 18-month column.
Uses adaptive folds (n_splits = min(10, n_ASD)) since only 9 ASD
dual-session instances exist at 18m; 24m/36m keep 10-fold (unchanged).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

AGG = "PATH/TO/FEATURES/aggregate/"
GT_CSV = "PATH/TO/FEATURES/aggregate/corrected_partnercontrast_source.csv"
BEHAVIORS = ["smile_rate","smile_prop","lookface_rate","lookface_prop",
             "lookobject_rate","lookobject_prop","vocal_rate","vocal_prop",
             "social_smile_rate","social_smile_prop","social_vocal_rate","social_vocal_prop"]
AGE = 18
SEED = 0

def parse(fn):
    # Released naming: <sub_id>_<visit>_{Examiner,Parent1,Parent2,Parent}.csv
    if "Examiner" in fn: return "Examiner"
    if fn.endswith("_Parent1.csv"): return "Parent1"
    if fn.endswith("_Parent2.csv"): return "Parent2"
    if fn.endswith("_Parent.csv"): return "Parent"
    return None

def build_mm(indicator):
    cohort_files = set(pd.read_csv(GT_CSV)["filename"].unique())
    if indicator == "G":
        df = pd.read_csv(GT_CSV)
    else:
        df = pd.concat([pd.read_csv(AGG+f"hj_test_{indicator}_Aggregated.csv"),
                        pd.read_csv(AGG+f"hj_asdtrain_{indicator}_Aggregated.csv")],
                       ignore_index=True)
        df = df[df["filename"].isin(cohort_files)]
    df = df.copy(); df["session"] = df["filename"].apply(parse)
    rows = []
    for (sid, v), g in df[df.visit==AGE].groupby(["sub_id","visit"]):
        ex = g[g.session=="Examiner"]; par = g[g.session.isin(["Parent1","Parent2","Parent"])]
        if len(ex)==0 or len(par)==0: continue
        rec = {"asd": 1 if g["asd_longitudinal"].iloc[0]=="ASD" else 0}
        for b in BEHAVIORS:
            rec[f"mm_{b}"] = par[b].mean() - ex[b].mean()
        rows.append(rec)
    return pd.DataFrame(rows)

def cv_auroc(X, y, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aurocs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2: continue
        pipe = ImbPipeline([("sc",StandardScaler()),("smt",SMOTETomek(random_state=SEED)),
                            ("lr",LogisticRegression(max_iter=1000))])
        try: pipe.fit(X[tr], y[tr])
        except ValueError:
            pipe = ImbPipeline([("sc",StandardScaler()),
                                ("lr",LogisticRegression(class_weight="balanced",max_iter=1000))])
            pipe.fit(X[tr], y[tr])
        aurocs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:,1]))
    a = np.array(aurocs)
    m = a.mean(); se = a.std(ddof=1)/np.sqrt(len(a))
    return m, m-1.96*se, m+1.96*se, len(a)

for indicator in ["G", "FOSV"]:
    mm = build_mm(indicator)
    npos = int(mm.asd.sum()); nfold = min(10, npos)
    print(f"\n=== {indicator} @ 18m (n={len(mm)}, ASD={npos}, folds={nfold}) ===")
    print(f"{'behavior':<20}{'AUROC [95% CI]':<26}{'Delta':>9}")
    for b in BEHAVIORS:
        sub = mm.dropna(subset=[f"mm_{b}","asd"])
        X = sub[[f"mm_{b}"]].values; y = sub["asd"].values
        delta = sub[sub.asd==1][f"mm_{b}"].mean() - sub[sub.asd==0][f"mm_{b}"].mean()
        m, lo, hi, nf = cv_auroc(X, y, nfold)
        print(f"{b:<20}{f'{m:.2f} [{lo:.2f}, {hi:.2f}]':<26}{delta:>+9.2f}")
