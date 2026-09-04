# features/ — drop the feature release here

This directory is intentionally empty in the code release. It is where the
full ASD-FEAT **feature release** goes so the Stage-2 scripts can find it.

The scripts refer to feature paths as `PATH/TO/FEATURES/...`; either edit those
placeholders to point here, or set them to wherever you unpacked the release.
Expected sub-structure:

```
features/
  parquet_files/                              per-recording frame-level features
  i3dfeatures/frame16/{rgb,flow}/             I3D embeddings
  audio_features/                             mel spectrograms
  aggregate/
    hj_test_{G,F,O,S,V,FS,FV,FOSV}_Aggregated.csv
    hj_asdtrain_{G,F,O,S,V,FS,FV,FOSV}_Aggregated.csv
    corrected_partnercontrast_source.csv
  labels/subject_labels.csv                   sub_id, visit, gender, asd_longitudinal, asd_36
  labels/session_labels.csv                   filename, sub_id, visit, gender, asd_longitudinal
  data_splits/standard_filenames/fold{0..9}/  split filename lists
```

See `../FEATURES_MANIFEST.md` for the file naming and the mapping to paper
tables, and `asd_data_sample/aggregate_features/` for a 15-session slice in the
same aggregate-feature format.
