# FEATURES_MANIFEST — feature files, scripts, and the paper tables they map to

This manifest documents (a) the aggregate feature files the Stage-2 code
consumes and how they are named, and (b) which paper table each Stage-2 script
reproduces. All feature paths in the code are `PATH/TO/FEATURES/...`
placeholders; point them at a local copy of the full ASD-FEAT feature release.

## Labeling-source indicators

Feature and result files are keyed by an `indicator` naming the source of the
behavior labels used to build them:

| Indicator | Source |
|---|---|
| `G` | ground truth — human-coded labels |
| `F` / `O` / `S` / `V` | single ML detector: Look-Face / Look-Object / Smile / Social-Vocalization |
| `FS` / `FV` | two ML detectors combined |
| `FOSV` | all four ML detectors |

## Aggregate feature files (Stage-2 inputs)

Located under `PATH/TO/FEATURES/aggregate/`:

| File pattern | Contents |
|---|---|
| `hj_test_{indicator}_Aggregated.csv` | per-session aggregate behavior features, held-out test fold |
| `hj_asdtrain_{indicator}_Aggregated.csv` | same, training partition |
| `corrected_partnercontrast_source.csv` | per-session source rows for the within-visit partner-contrast analyses (§5.4) |

Column schema of the `hj_*` files (per behavior channel — smile, lookface,
lookobject, vocal, social_smile, social_vocal — `fq`, `rate`, `duration`,
`prop`) is documented alongside the dataset release; the same schema is used by
every `hj_*` file.

Fold-3 split filename lists are read from
`PATH/TO/FEATURES/data_splits/standard_filenames/fold3/`.

## Stage-2 scripts → paper tables

| Script | Reproduces |
|---|---|
| `temporal_to_aggregate.py` | builds the `hj_*` aggregate files from Stage-1 frame predictions (or human labels) |
| `aggregate_to_predictions.py` | end-to-end aggregate-features → ASD-risk prediction (reference pipeline) |
| `testing_classifiers.py` | shared LR / RF / MLP training + cross-validation used by the drivers |
| `run_ml_features_experiment.py` | **Table 10** component-validation sweep (G…FOSV) |
| `run_ml_features_keras.py` | **Table 10** sweep, Keras-MLP classifier variant |
| `plot_sweep.py` | figure of the **Table 10** sweep, one panel per metric |
| `run_partnercontrast_experiment.py` | **Table 13** augmented 38-dim classifier (26 standard + 12 partner-contrast) and §5.6 panels |
| `run_26dim_matched.py` | **Table 13** 26-dim baseline on the matched partner-contrast cohort (filter-test / averaged modes) |
| `run_26dim_prob_agg.py` | **Table 13** 26-dim baseline, per-(subject,visit) probability-averaged variant |
| `eda_partnercontrast_cv.py` | **Table 11** partner-contrast AUROC + Δ (human-coded) |
| `eda_partnercontrast_cv_ml.py` | **Table 13** partner-contrast AUROC under ML substitution |
| `compute_18m_rows.py` | 18-month rows of **Tables 11 / 13** |
| `plot_partner_stratified.py` | **Table 12** partner-stratified means (human-coded) |
| `plot_partner_stratified_fosv.py` | partner-stratified means under all-ML (FOSV) labeling |
| `compute_extended_metrics.py` | **Table 15** MCC / LR± / PPV@prevalence |

## Result files

The `run_*` and `compute_*` scripts write their result CSV/JSON next to the
configured features path (`PATH/TO/FEATURES/...`). File names are printed at the
end of each run.

