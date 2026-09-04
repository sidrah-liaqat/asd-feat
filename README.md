# ASD-FEAT — code release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22261227.svg)](https://doi.org/10.5281/zenodo.22261227)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code accompanying *ASD-FEAT: A Multi-Modal Infant Video-Derived Dataset for Early ASD
Risk Prediction*. It covers the paper's two-stage pipeline:

- **Stage 1 — behavior detection** (`stage1_behavior_detection/`): transformer
  detectors for the four child behaviors (Look-Face, Look-Object, Smile,
  Social-Vocalization) from the per-frame multi-modal features.
- **Stage 2 — ASD risk prediction** (`stage2_risk_prediction/`): summarizes the
  frame-level behavior labels into per-session aggregate features and trains /
  evaluates the ASD-risk classifiers.

See `FEATURES_MANIFEST.md` for the mapping from each Stage-2 script and feature
file to the tables it reproduces in the paper.

## Dataset

The ASD-FEAT dataset is distributed separately via Zenodo under a persistent DOI:
**[10.5281/zenodo.22261227](https://doi.org/10.5281/zenodo.22261227)**. It contains
derived features only (facial and eye landmarks, head and gaze pose, action units,
I3D embeddings, and mel spectrograms) with de-identified metadata — no raw video or
audio. Access is managed through the repository's approved procedures; see the
paper's "Licensing, Access, and Ethics" section.

## Citing

If you use this code or the dataset, please cite the paper and the dataset DOI
above. Repository metadata for citation managers is in `CITATION.cff`.

## Layout

```
stage1_behavior_detection/   Stage-1 training + inference (see its own README.md)
stage2_risk_prediction/      Stage-2 feature building, classifiers, and analyses
features/                    (empty) drop the full feature release here — see features/README.md
FEATURES_MANIFEST.md         file -> paper-table manifest and feature-file naming
```

## Setup

```bash
git clone https://github.com/sidrah-liaqat/asd-feat.git
cd asd-feat
python -m venv .venv && source .venv/bin/activate
pip install -r stage1_behavior_detection/requirements.txt
```

Stage 1 requires PyTorch (the pinned build is CUDA 12.4; swap for a CPU build if
needed). Stage 2 additionally uses scikit-learn, imbalanced-learn and, for the
Keras classifier variants, TensorFlow. Note the pinned `numpy==1.24.1` — NumPy 2.x
breaks the PyTorch `from_numpy` bridge for these builds.

## Getting the features

The code reads pre-computed aggregate features from a features directory. All
paths in the scripts are written as `PATH/TO/FEATURES/...` placeholders; point
them at a local copy of the full ASD-FEAT feature release (hosted on the
repository described in the paper's "Licensing, Access, and Ethics" section).

The expected directory layout is documented in
[`features/README.md`](features/README.md), and the naming of every feature file
in [`FEATURES_MANIFEST.md`](FEATURES_MANIFEST.md). Reproducing the paper's
tables requires the full-cohort feature release from Zenodo.

## Stage 2 — how the pieces fit

1. `temporal_to_aggregate.py` — turns Stage-1 frame-level predictions (or the
   human-coded labels) into per-session aggregate behavior statistics.
2. The `run_*` drivers pool the test and train aggregate files for a chosen
   labeling source and train / cross-validate the Stage-2 classifiers.
3. `testing_classifiers.py` — shared LR/RF/MLP training + CV evaluation.
4. `compute_extended_metrics.py` — MCC / LR± / PPV@prevalence for the reported
   configurations.
5. Partner-contrast analyses (`eda_partnercontrast_cv*.py`,
   `run_partnercontrast_experiment.py`, `plot_partner_stratified*.py`) reproduce
   the within-visit partner-contrast results (§5.4), in human-coded and ML
   variants.

## Labeling-source indicators

Aggregate feature files and results are keyed by a labeling-source indicator:
`G` = ground truth (human) · `F`/`O`/`S`/`V` = single ML detector
(Look-Face / Look-Object / Smile / Social-Vocalization) · `FS`/`FV` = two-channel ·
`FOSV` = all four ML detectors.

## License

MIT — see [`LICENSE`](LICENSE).

Note that the released features are produced with third-party tools whose
licenses propagate to downstream use (OpenFace 2.0 and Gaze360 are
non-commercial/research-only). See the dataset's datasheet for the full table.
