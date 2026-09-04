# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""
Compute extended clinical metrics (MCC, LR+, LR-, PPV@3.2%) per fold from
the raw fold-level sensitivity/specificity stored in the results JSONs, then
aggregate as mean +/- 1.96*SE (matching the ci95 used in the experiment scripts).

MCC is evaluated at the cohort prevalence; PPV@3.2% recalibrates to the
3.2% U.S. general-population ASD prevalence via Bayes' rule.
"""
import json
import numpy as np
import pandas as pd

GEN_PREV = 0.032

DATAPATH = 'PATH/TO/FEATURES/aggregate/'
PARTNERCONTRAST_VISITS = [18, 24, 36]
FEATURES = [
    'visit', 'gender',
    'total_smile_fq', 'smile_rate', 'total_smile_duration', 'smile_prop',
    'total_lookface_fq', 'lookface_rate', 'total_lookface_dur', 'lookface_prop',
    'total_lookobject_fq', 'lookobject_rate', 'total_lookobject_dur', 'lookobject_prop',
    'total_vocal_fq', 'vocal_rate', 'total_vocal_duration', 'vocal_prop',
    'total_social_smile_fq', 'social_smile_rate', 'total_social_smile_duration', 'social_smile_prop',
    'total_social_vocal_fq', 'social_vocal_rate', 'total_social_vocal_duration', 'social_vocal_prop',
]


def ci95(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    half = 1.96 * se
    return mean, mean - half, mean + half


def keras_prevalence():
    df = pd.concat([
        pd.read_csv(DATAPATH + 'hj_test_G_Aggregated.csv'),
        pd.read_csv(DATAPATH + 'hj_asdtrain_G_Aggregated.csv'),
    ], ignore_index=True)
    df['asd_longitudinal'] = df['asd_longitudinal'].map({'Non-ASD': 0, 'ASD': 1})
    df.dropna(subset=FEATURES + ['asd_longitudinal', 'sub_id', 'filename'], inplace=True)
    return df['asd_longitudinal'].mean()


def partnercontrast_cohort_prevalence():
    df = pd.concat([
        pd.read_csv(DATAPATH + 'hj_test_G_Aggregated.csv'),
        pd.read_csv(DATAPATH + 'hj_asdtrain_G_Aggregated.csv'),
    ], ignore_index=True)
    df['asd_longitudinal'] = df['asd_longitudinal'].map({'Non-ASD': 0, 'ASD': 1})
    df = df[df['visit'].isin(PARTNERCONTRAST_VISITS)]
    df.dropna(subset=FEATURES + ['asd_longitudinal', 'sub_id', 'filename'], inplace=True)
    df['is_exam'] = df['filename'].str.lower().str.contains('examiner')
    # Keep only subject-VISITS that have both an examiner and a parent session.
    # (Matching by subject alone would also retain that subject's single-partner
    # visits, inflating the matched cohort.)
    valid = set()
    for (sub, vis), grp in df.groupby(['sub_id', 'visit']):
        if grp['is_exam'].any() and (~grp['is_exam']).any():
            valid.add((sub, vis))
    df = df[[(s, v) in valid for s, v in zip(df['sub_id'], df['visit'])]]
    inst = df.groupby(['sub_id', 'visit'])['asd_longitudinal'].first()
    return inst.mean(), len(inst), int(inst.sum())


def extended(sens, spec, prev):
    """sens, spec as fractions in [0,1]."""
    tp = prev * sens
    fn = prev * (1 - sens)
    tn = (1 - prev) * spec
    fp = (1 - prev) * (1 - spec)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else np.inf
    lr_neg = (1 - sens) / spec if spec > 0 else np.inf
    ppv_cal = (sens * GEN_PREV) / (sens * GEN_PREV + (1 - spec) * (1 - GEN_PREV)) \
        if (sens * GEN_PREV + (1 - spec) * (1 - GEN_PREV)) > 0 else 0.0
    return mcc, lr_pos, lr_neg, ppv_cal


def process(json_path, prev, label):
    d = json.load(open(json_path))
    rows = []
    for ind, r in d.items():
        sens = np.array(r['raw_sens'], dtype=float)
        spec = np.array(r['raw_spec'], dtype=float)
        if sens.max() > 1.5:  # stored as percent
            sens = sens / 100.0
        if spec.max() > 1.5:
            spec = spec / 100.0
        mcc_f, lrp_f, lrn_f, ppvc_f = [], [], [], []
        for s, sp in zip(sens, spec):
            m, lp, ln, pc = extended(s, sp, prev)
            mcc_f.append(m)
            lrp_f.append(lp)
            lrn_f.append(ln)
            ppvc_f.append(pc)
        mcc = ci95(mcc_f)
        lrp = ci95([x for x in lrp_f if np.isfinite(x)])
        lrn = ci95(lrn_f)
        ppvc = ci95(ppvc_f)
        rows.append({
            'indicator': ind,
            'mcc': round(mcc[0], 3), 'mcc_lo': round(mcc[1], 3), 'mcc_hi': round(mcc[2], 3),
            'lrpos': round(lrp[0], 2), 'lrpos_lo': round(lrp[1], 2), 'lrpos_hi': round(lrp[2], 2),
            'lrneg': round(lrn[0], 2), 'lrneg_lo': round(lrn[1], 2), 'lrneg_hi': round(lrn[2], 2),
            'ppv32': round(ppvc[0] * 100, 1), 'ppv32_lo': round(ppvc[1] * 100, 1),
            'ppv32_hi': round(ppvc[2] * 100, 1),
        })
    out = pd.DataFrame(rows)
    print(f'\n=== {label} (prevalence={prev:.4f}) ===')
    print(out.to_string(index=False))
    return out


if __name__ == '__main__':
    p_keras = keras_prevalence()
    p_mm, n_mm, n_asd = partnercontrast_cohort_prevalence()
    print(f'Exp1 keras prevalence = {p_keras:.4f}')
    print(f'Partnercontrast cohort: {n_mm} subject-visit instances, {n_asd} ASD, '
          f'prevalence = {p_mm:.4f}')

    a = process('ml_features_keras_results.json', p_keras, 'Exp1: 26-dim full (Table 10)')
    b = process('26dim_prob_agg_results.json', p_mm, 'Exp2: 26-dim prob-agg baseline (matched cohort)')
    c = process('partnercontrast_augmented_results.json', p_mm, 'Exp3: 34-dim augmented (matched cohort)')

    a.to_csv('extended_metrics_keras.csv', index=False)
    b.to_csv('extended_metrics_probagg.csv', index=False)
    c.to_csv('extended_metrics_augmented.csv', index=False)
    print('\nSaved extended_metrics_{keras,probagg,augmented}.csv')
