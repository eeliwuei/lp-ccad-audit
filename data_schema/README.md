# Data schema

`data/S1_Data.csv` is the canonical tidy table for values used in the main manuscript, tables, and figures. Its `record_id` field is stable within this release candidate. The `source_artifact` field records provenance inside the frozen internal archive; a historical filename in that field is not a current dataset name or a legal-clearance claim.

`data/S2_Data.csv` contains available arm-level and seed-level derived endpoints, including all 12 corrected-pipeline composite follow-up runs and the source-excluded aggregate evaluation records.

`data/S3_Data.csv` contains aggregate recorded filename-family counts. These families are filename-derived technical labels, not verified acquisition sources or independent clusters.

`data/S4_Data.csv` contains 7,000 resampling replicate rows, their seven summaries, and three retrospective equivalence calculations. Resampling is conditional on trained models and does not estimate training-seed uncertainty.

`data/S6_Data.csv` contains exact nominal v2/v3 schedule-epoch assignments, the 12 run--schedule bindings, source-exclusion contrast triples and aggregate counts, resampling and equivalence parameters, aggregate coordinate-admission audit records, and frozen-artifact integrity locators. Its interpretation and privacy boundary is documented in `S6_Data_README.md`.

The exact schedule YAMLs are also provided under `configs/frozen_protocol/` and `configs/factorial_schedules/`. Schedule release supports configuration inspection; it does not resolve the one-to-one confounding between v3 schedule realization and training seed or reproduce training.

No raw image or video media, source-identifying filename list, ground-truth annotation file, per-image prediction file, or checkpoint is included.
