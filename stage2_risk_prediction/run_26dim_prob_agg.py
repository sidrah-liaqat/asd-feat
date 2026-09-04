# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
"""
26-dim classifier restricted to the partnercontrast cohort (18/24/36 months,
subjects with both examiner and parent sessions).

All sessions are kept as separate training/test rows. At test time,
softmax probabilities are averaged per (sub_id, visit) before thresholding,
giving one prediction per subject×visit — directly comparable to the
34-dim augmented experiment.

Indicators: G, F, O, S, V, FV, FOSV
Same Keras model, fold3 splits, 10-run ensemble as run_ml_features_keras.py.
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AlphaDropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

DATAPATH  = 'PATH/TO/FEATURES/aggregate/'
SPLITPATH = 'PATH/TO/FEATURES/data_splits/standard_filenames/fold3/'
FOLD      = 3
SUBFOLDS  = [0, 1, 2, 4, 5, 6, 7, 8, 9]
N_RUNS    = 10
EPOCHS    = 1000
BATCH_SIZE = 64
LR        = 0.0009

PARTNERCONTRAST_VISITS     = [18, 24, 36]
PARTNERCONTRAST_INDICATORS = ['G', 'F', 'O', 'S', 'V', 'FS', 'FV', 'FOSV']

FEATURES = [
    'visit', 'gender',
    'total_smile_fq', 'smile_rate', 'total_smile_duration', 'smile_prop',
    'total_lookface_fq', 'lookface_rate', 'total_lookface_dur', 'lookface_prop',
    'total_lookobject_fq', 'lookobject_rate', 'total_lookobject_dur', 'lookobject_prop',
    'total_vocal_fq', 'vocal_rate', 'total_vocal_duration', 'vocal_prop',
    'total_social_smile_fq', 'social_smile_rate', 'total_social_smile_duration', 'social_smile_prop',
    'total_social_vocal_fq', 'social_vocal_rate', 'total_social_vocal_duration', 'social_vocal_prop',
]


def get_partnercontrast_sub_ids(indicator='G'):
    """Subjects with both examiner and parent sessions at 18/24/36m in GT."""
    test  = pd.read_csv(DATAPATH + f'hj_test_{indicator}_Aggregated.csv')
    train = pd.read_csv(DATAPATH + f'hj_asdtrain_{indicator}_Aggregated.csv')
    df = pd.concat([test, train], ignore_index=True)
    df['asd_longitudinal'] = df['asd_longitudinal'].map({'Non-ASD': 0, 'ASD': 1})
    df = df[df['visit'].isin(PARTNERCONTRAST_VISITS)]
    df.dropna(subset=FEATURES + ['asd_longitudinal', 'sub_id', 'filename'], inplace=True)
    df['is_exam'] = df['filename'].str.lower().str.contains('examiner')
    valid = set()
    for (sub, vis), grp in df.groupby(['sub_id', 'visit']):
        if grp['is_exam'].any() and (~grp['is_exam']).any():
            valid.add(sub)
    return valid


def load_data(indicator, partnercontrast_sub_ids):
    """Load all sessions, filtered to partnercontrast visits and subjects."""
    test  = pd.read_csv(DATAPATH + f'hj_test_{indicator}_Aggregated.csv')
    train = pd.read_csv(DATAPATH + f'hj_asdtrain_{indicator}_Aggregated.csv')
    df = pd.concat([test, train], ignore_index=True)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['asd_longitudinal'] = df['asd_longitudinal'].map({'Non-ASD': 0, 'ASD': 1})
    df = df[df['visit'].isin(PARTNERCONTRAST_VISITS)]
    df = df[df['sub_id'].isin(partnercontrast_sub_ids)]
    df.dropna(subset=FEATURES + ['asd_longitudinal', 'sub_id', 'filename'], inplace=True)
    return df


def load_split_filenames(split_file):
    with open(split_file) as f:
        return set(line.strip() + '.csv' for line in f if line.strip())


def build_model(input_dim, seed):
    tf.random.set_seed(seed)
    model = Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        AlphaDropout(0.5),
        Dense(512, activation='relu'),
        AlphaDropout(0.5),
        Dense(512, activation='relu'),
        AlphaDropout(0.5),
        Dense(2, activation='softmax'),
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        metrics=['accuracy'],
    )
    return model


def ci95(vals):
    arr = np.array(vals)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    half = 1.96 * se
    return mean, mean - half, mean + half


def run_experiment(indicator, partnercontrast_sub_ids):
    df = load_data(indicator, partnercontrast_sub_ids)
    print(f'  Sessions: {len(df)}, Subjects: {df.sub_id.nunique()}, '
          f'ASD sessions: {df.asd_longitudinal.sum()}, '
          f'Non-ASD sessions: {(df.asd_longitudinal==0).sum()}')
    sys.stdout.flush()

    sens_list, spec_list, ppv_list, auroc_list, acc_list = [], [], [], [], []

    for subfold in SUBFOLDS:
        test_names  = load_split_filenames(SPLITPATH + f'test_fold{FOLD}_subfold{subfold}.txt')
        train_names = load_split_filenames(SPLITPATH + f'asdtrain_fold{FOLD}_subfold{subfold}.txt')

        df_train = df[df['filename'].isin(train_names)]
        df_test  = df[df['filename'].isin(test_names)]

        if df_test['asd_longitudinal'].sum() == 0:
            print(f'  Subfold {subfold}: skipping (no ASD sessions in test set)')
            continue

        X_train = df_train[FEATURES].values.astype(float)
        y_train = df_train['asd_longitudinal'].values.astype(int)
        X_test  = df_test[FEATURES].values.astype(float)

        # Ground-truth label at (sub_id, visit) level for evaluation
        test_groups = df_test[['sub_id', 'visit', 'asd_longitudinal']].copy()

        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros = ros.fit_resample(X_train, y_train)
        try:
            smt = SMOTETomek(random_state=42)
            X_res, y_res = smt.fit_resample(X_ros, y_ros)
        except Exception as e:
            print(f'  Subfold {subfold}: SMOTE failed ({e}), using ROS data')
            X_res, y_res = X_ros, y_ros

        sc = MinMaxScaler()
        X_res_sc  = sc.fit_transform(X_res)
        X_test_sc = sc.transform(X_test)
        y_res_cat = keras.utils.to_categorical(y_res, 2)

        # 10-run ensemble: accumulate session-level probabilities
        prob_sum = np.zeros((len(X_test), 2))
        for run in range(N_RUNS):
            model = build_model(X_res_sc.shape[1], seed=run)
            es  = EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='min')
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100,
                                    verbose=0, epsilon=1e-4, mode='min')
            model.fit(X_res_sc, y_res_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.1, callbacks=[es, rlr], verbose=0)
            prob_sum += model.predict(X_test_sc, verbose=0)
            keras.backend.clear_session()

        session_probs = prob_sum / N_RUNS  # shape (n_sessions, 2)

        # Average probabilities per (sub_id, visit) → one prediction per instance
        test_groups = test_groups.reset_index(drop=True)
        test_groups['prob_asd'] = session_probs[:, 1]
        visit_preds = (test_groups
                       .groupby(['sub_id', 'visit'])
                       .agg(prob_asd=('prob_asd', 'mean'),
                            asd_longitudinal=('asd_longitudinal', 'first'))
                       .reset_index())

        y_test  = visit_preds['asd_longitudinal'].values.astype(int)
        y_prob  = visit_preds['prob_asd'].values
        y_pred  = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        acc   = (tp + tn) / len(y_test)
        try:
            auroc = roc_auc_score(y_test, y_prob)
        except Exception:
            auroc = 0.5

        n_instances = len(visit_preds)
        sens_list.append(sens); spec_list.append(spec)
        ppv_list.append(ppv);   auroc_list.append(auroc); acc_list.append(acc)
        print(f'  Subfold {subfold}: n={n_instances}  '
              f'Sens={sens*100:.1f}  Spec={spec*100:.1f}  '
              f'PPV={ppv*100:.1f}  AUROC={auroc:.2f}  Acc={acc*100:.1f}')
        sys.stdout.flush()

    if not sens_list:
        return None
    return {
        'n_folds': len(sens_list),
        'sens': ci95(sens_list), 'spec': ci95(spec_list),
        'ppv': ci95(ppv_list),   'auroc': ci95(auroc_list), 'acc': ci95(acc_list),
        'raw_sens': sens_list, 'raw_spec': spec_list, 'raw_ppv': ppv_list,
        'raw_auroc': auroc_list, 'raw_acc': acc_list,
    }


def indicator_label(ind):
    mapping = {
        'G': 'Ground Truth', 'F': 'Look Face from ML', 'O': 'Look Object from ML',
        'S': 'Smile from ML', 'V': 'Vocal from ML',
        'FV': 'Look Face + Vocal from ML', 'FOSV': 'All behaviors from ML',
    }
    return mapping.get(ind, ind)


if __name__ == '__main__':
    partnercontrast_sub_ids = get_partnercontrast_sub_ids('G')
    print(f'Partnercontrast cohort: {len(partnercontrast_sub_ids)} subjects\n')

    all_results = {}
    for indicator in PARTNERCONTRAST_INDICATORS:
        print(f'\nRunning indicator: {indicator}')
        sys.stdout.flush()
        res = run_experiment(indicator, partnercontrast_sub_ids)
        if res is None:
            print('  ERROR: no valid folds')
            continue
        all_results[indicator] = res
        s, sl, sh    = res['sens']
        sp, spl, sph = res['spec']
        p, pl, ph    = res['ppv']
        a, al, ah    = res['auroc']
        ac, acl, ach = res['acc']
        print(f'  >>> Folds={res["n_folds"]}  '
              f'Sens={s*100:.1f}% ({sl*100:.1f}--{sh*100:.1f})  '
              f'Spec={sp*100:.1f}%  PPV={p*100:.1f}%  '
              f'AUROC={a:.2f}  Acc={ac*100:.1f}%')
        sys.stdout.flush()

    # Save JSON
    save_data = {k: {mk: list(mv) if isinstance(mv, (np.ndarray, list)) else mv
                     for mk, mv in v.items()}
                 for k, v in all_results.items()}
    outjson = 'PATH/TO/FEATURES/26dim_prob_agg_results.json'
    with open(outjson, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nResults saved to {outjson}')

    # LaTeX
    print('\n\n=== LaTeX table rows ===')
    for ind in PARTNERCONTRAST_INDICATORS:
        if ind not in all_results:
            continue
        res = all_results[ind]
        s, sl, sh    = res['sens']
        sp, spl, sph = res['spec']
        p, pl, ph    = res['ppv']
        a, al, ah    = res['auroc']
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
        sys.stdout.flush()

    # CSV
    rows = []
    for ind in PARTNERCONTRAST_INDICATORS:
        if ind not in all_results:
            continue
        res = all_results[ind]
        rows.append({
            'indicator': ind, 'label': indicator_label(ind),
            'sens_mean': res['sens'][0]*100,  'sens_lo': res['sens'][1]*100,  'sens_hi': res['sens'][2]*100,
            'spec_mean': res['spec'][0]*100,  'spec_lo': res['spec'][1]*100,  'spec_hi': res['spec'][2]*100,
            'ppv_mean':  res['ppv'][0]*100,   'ppv_lo':  res['ppv'][1]*100,   'ppv_hi':  res['ppv'][2]*100,
            'auroc_mean': res['auroc'][0],     'auroc_lo': res['auroc'][1],    'auroc_hi': res['auroc'][2],
            'acc_mean':  res['acc'][0]*100,   'acc_lo':  res['acc'][1]*100,   'acc_hi':  res['acc'][2]*100,
        })
    outcsv = 'PATH/TO/FEATURES/26dim_prob_agg_results.csv'
    pd.DataFrame(rows).to_csv(outcsv, index=False)
    print(f'Summary CSV saved to {outcsv}')
