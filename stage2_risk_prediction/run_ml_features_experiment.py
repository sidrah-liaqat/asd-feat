# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""
Replication of Section 5.4 experiment (Table 10) using ML-generated behavioral features.

For each source indicator, pools hj_test_* and hj_asdtrain_* CSVs (behaviortrain
is excluded to avoid contaminating the behavior model's training data).
Runs stratified 10-fold cross-validation (keeping all visits of a subject in the
same fold) with SMOTE+Tomek imbalance handling and an MLP classifier.
Reports sensitivity, specificity, PPV, AUROC, accuracy with 95% CIs.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.combine import SMOTETomek
import json

DATAPATH = 'PATH/TO/FEATURES/aggregate/'

# All available source indicators
SOURCE_INDICATORS = [
    'G',
    'F', 'O', 'S', 'V',
    'FO', 'FS', 'FV', 'OS', 'OV', 'SV',
    'FOS', 'FOV', 'FSV', 'OSV',
    'FOSV',
]

FEATURES = [
    'visit', 'gender',
    'total_smile_fq', 'smile_rate', 'total_smile_duration', 'smile_prop',
    'total_lookface_fq', 'lookface_rate', 'total_lookface_dur', 'lookface_prop',
    'total_lookobject_fq', 'lookobject_rate', 'total_lookobject_dur', 'lookobject_prop',
    'total_vocal_fq', 'vocal_rate', 'total_vocal_duration', 'vocal_prop',
    'total_social_smile_fq', 'social_smile_rate', 'total_social_smile_duration', 'social_smile_prop',
    'total_social_vocal_fq', 'social_vocal_rate', 'total_social_vocal_duration', 'social_vocal_prop',
]


def load_data(indicator):
    test_path = DATAPATH + f'hj_test_{indicator}_Aggregated.csv'
    train_path = DATAPATH + f'hj_asdtrain_{indicator}_Aggregated.csv'
    df = pd.concat([pd.read_csv(test_path), pd.read_csv(train_path)], ignore_index=True)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['asd_longitudinal'] = df['asd_longitudinal'].map({'Non-ASD': 0, 'ASD': 1})
    df.dropna(subset=FEATURES + ['asd_longitudinal', 'sub_id'], inplace=True)
    return df


def ci95(vals):
    arr = np.array(vals)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    half = 1.96 * se
    return mean, mean - half, mean + half


def run_cv(df, n_splits=10, random_state=42):
    X = df[FEATURES].values
    y = df['asd_longitudinal'].values.astype(int)
    groups = df['sub_id'].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    sens_list, spec_list, ppv_list, auroc_list, acc_list = [], [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip fold if test set has no ASD cases (can't compute sensitivity)
        if y_test.sum() == 0 or (y_test == 0).sum() == 0:
            print(f'  Fold {fold_idx}: skipping (degenerate test fold)')
            continue

        # SMOTE + Tomek Links to handle class imbalance on training set
        try:
            smt = SMOTETomek(random_state=random_state)
            X_res, y_res = smt.fit_resample(X_train, y_train)
        except Exception as e:
            print(f'  Fold {fold_idx}: SMOTE failed ({e}), using raw training data')
            X_res, y_res = X_train, y_train

        # Scale features (fit on resampled training, transform test)
        sc = MinMaxScaler()
        X_res = sc.fit_transform(X_res)
        X_test_scaled = sc.transform(X_test)

        # MLP with early stopping; (256, 128) balances accuracy and runtime
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )
        clf.fit(X_res, y_res)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        acc = (tp + tn) / len(y_test)

        try:
            auroc = roc_auc_score(y_test, y_prob)
        except Exception:
            auroc = 0.5

        sens_list.append(sens)
        spec_list.append(spec)
        ppv_list.append(ppv)
        auroc_list.append(auroc)
        acc_list.append(acc)

    if not sens_list:
        return None

    return {
        'n_folds': len(sens_list),
        'sens': ci95(sens_list),
        'spec': ci95(spec_list),
        'ppv': ci95(ppv_list),
        'auroc': ci95(auroc_list),
        'acc': ci95(acc_list),
        'raw_sens': sens_list,
        'raw_spec': spec_list,
        'raw_ppv': ppv_list,
        'raw_auroc': auroc_list,
        'raw_acc': acc_list,
    }


def fmt(mean, lo, hi, pct=True):
    if pct:
        return f'{mean*100:.1f} ({lo*100:.1f}--{hi*100:.1f})'
    else:
        return f'{mean:.2f} ({lo:.2f}--{hi:.2f})'


def indicator_label(ind):
    mapping = {
        'G': 'Ground Truth',
        'F': 'Look Face from ML',
        'O': 'Look Object from ML',
        'S': 'Smile from ML',
        'V': 'Vocal from ML',
        'FO': 'Look Face + Look Object from ML',
        'FS': 'Look Face + Smile from ML',
        'FV': 'Look Face + Vocal from ML',
        'OS': 'Look Object + Smile from ML',
        'OV': 'Look Object + Vocal from ML',
        'SV': 'Smile + Vocal from ML',
        'FOS': 'Look Face + Look Object + Smile from ML',
        'FOV': 'Look Face + Look Object + Vocal from ML',
        'FSV': 'Look Face + Smile + Vocal from ML',
        'OSV': 'Look Object + Smile + Vocal from ML',
        'FOSV': 'All behaviors from ML',
    }
    return mapping.get(ind, ind)


if __name__ == '__main__':
    all_results = {}

    for indicator in SOURCE_INDICATORS:
        print(f'\nRunning indicator: {indicator}')
        df = load_data(indicator)
        print(f'  Rows: {len(df)}, Subjects: {df.sub_id.nunique()}, '
              f'ASD: {df.asd_longitudinal.sum()}, Non-ASD: {(df.asd_longitudinal==0).sum()}')
        res = run_cv(df)
        if res is None:
            print(f'  ERROR: no valid folds')
            continue
        all_results[indicator] = res
        print(f'  Folds completed: {res["n_folds"]}')
        s, sl, sh = res['sens']
        sp, spl, sph = res['spec']
        p, pl, ph = res['ppv']
        a, al, ah = res['auroc']
        ac, acl, ach = res['acc']
        print(f'  Sens={s*100:.1f}% ({sl*100:.1f}--{sh*100:.1f})')
        print(f'  Spec={sp*100:.1f}% ({spl*100:.1f}--{sph*100:.1f})')
        print(f'  PPV ={p*100:.1f}% ({pl*100:.1f}--{ph*100:.1f})')
        print(f'  AUROC={a:.2f} ({al:.2f}--{ah:.2f})')
        print(f'  Acc ={ac*100:.1f}% ({acl*100:.1f}--{ach*100:.1f})')

    # Save raw results
    save_data = {k: {mk: list(mv) if isinstance(mv, (np.ndarray, list)) else mv
                     for mk, mv in v.items()}
                 for k, v in all_results.items()}
    with open('PATH/TO/FEATURES/ml_features_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print('\nResults saved to ml_features_results.json')

    # Print LaTeX table rows
    print('\n\n=== LaTeX table rows ===')
    for ind in SOURCE_INDICATORS:
        if ind not in all_results:
            continue
        res = all_results[ind]
        s, sl, sh = res['sens']
        sp, spl, sph = res['spec']
        p, pl, ph = res['ppv']
        a, al, ah = res['auroc']
        ac, acl, ach = res['acc']
        label = indicator_label(ind)
        print(
            f'{label:<45} & '
            f'\\makecell{{{s*100:.1f} \\\\ ({sl*100:.1f}--{sh*100:.1f})}} & '
            f'\\makecell{{{sp*100:.1f} \\\\ ({spl*100:.1f}--{sph*100:.1f})}} & '
            f'\\makecell{{{p*100:.1f} \\\\ ({pl*100:.1f}--{ph*100:.1f})}} & '
            f'\\makecell{{{a:.2f} \\\\ ({al:.2f}--{ah:.2f})}} & '
            f'\\makecell{{{ac*100:.1f} \\\\ ({acl*100:.1f}--{ach*100:.1f})}} \\\\'
        )

    # Save summary CSV
    rows = []
    for ind in SOURCE_INDICATORS:
        if ind not in all_results:
            continue
        res = all_results[ind]
        rows.append({
            'indicator': ind,
            'label': indicator_label(ind),
            'sens_mean': res['sens'][0] * 100,
            'sens_lo': res['sens'][1] * 100,
            'sens_hi': res['sens'][2] * 100,
            'spec_mean': res['spec'][0] * 100,
            'spec_lo': res['spec'][1] * 100,
            'spec_hi': res['spec'][2] * 100,
            'ppv_mean': res['ppv'][0] * 100,
            'ppv_lo': res['ppv'][1] * 100,
            'ppv_hi': res['ppv'][2] * 100,
            'auroc_mean': res['auroc'][0],
            'auroc_lo': res['auroc'][1],
            'auroc_hi': res['auroc'][2],
            'acc_mean': res['acc'][0] * 100,
            'acc_lo': res['acc'][1] * 100,
            'acc_hi': res['acc'][2] * 100,
        })
    pd.DataFrame(rows).to_csv(
        'PATH/TO/FEATURES/ml_features_results.csv', index=False
    )
    print('Summary CSV saved to ml_features_results.csv')
