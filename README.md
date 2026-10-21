# LP-CCAD multi-axis confirmation audit: analytical replay

This repository accompanies the manuscript **“A multi-axis confirmation audit of label-space curriculum distillation under evaluation and pipeline variation.”** It replays the derived numerical record used in the manuscript and regenerates the four publication figures from machine-readable tables.

## What this release supports

- H2 validation: the replay recomputes the three per-seed C4-M minus C4MixFT-M differences (seeds 42, 1337, and 20260703), the protocol-specified two-seed mean (seeds 42 and 1337), and the three-seed mean from the five-decimal arm endpoints in `data/S2_Data.csv`, and compares each value with the stored record in `data/S1_Data.csv`.
- Evaluation routes: the replay recomputes the per-seed differences and the two-seed means on both routes, and the shift between the route means. The 2,019-image route uses the full-precision endpoints in `data/S2_Data.csv`. The 2,553-image route uses the six-decimal arm endpoints in `data/S1_Data.csv`, because S2 Data has no per-seed endpoints for that route. The stored 2,553-image mean (-0.009111530600) carries more digits than these endpoints and agrees with the recomputed mean at six decimals.
- 12-run composite follow-up: the replay recomputes the four cell means and sample SDs, the per-seed order-axis, tail-axis, and difference-of-differences contrasts, their means over seeds, and the descriptive 95% paired-t intervals (t = 4.302652729911275, df = 2) from the 12 run endpoints in `data/S2_Data.csv`, and compares each value with the stored record in `data/S1_Data.csv`.
- Comparison rule: a recomputed value passes when it differs from the stored value by at most half a unit in the last digit printed in the manuscript. `run_all.sh` stops with a `RECOMPUTATION MISMATCH` error if any value fails. The keys checked against `expected_outputs/canonical_results.json` are still taken from the stored records in `data/S1_Data.csv`: the two validation means are averaged from the stored per-seed contrasts, the admission-gate fraction is computed from the stored counts (1,947 of 9,334 box-eligible rows), and the route means, route shift, follow-up means, and order-axis interval are read as stored.
- Filename-derived grouping resamples and retrospective equivalence calculations in `data/S4_Data.csv` are provided for inspection; the replay does not read or recompute them.
- Exact v2/v3 epoch schedules, the 12 run--schedule bindings, all seven original-to-source-excluded contrast triples, resampling/TOST parameters, and aggregate admission-gate records in `data/S6_Data.csv` and `configs/` are provided for inspection; the tests check their checksums and structure, and no value is recomputed from them.
- Fig 1-Fig 4 can be regenerated from the stored records in `data/S1_Data.csv` (optional dependencies, see below).

Run the portable replay with:

```sh
./run_all.sh
```

With the optional figure dependencies from `environment.yml` installed,
regenerate the publication figures with:

```sh
python3 figures/generate_all_figures.py
```

## Scope boundary

This package is an **analytical replay** of the derived numerical record. It does not contain raw third-party images or videos, model checkpoints, the teacher evidence bank, or the training and distillation code.

Two companion Supporting Information files take the record down to the image level. S7 Data lists every image in the training, validation, and evaluation splits (20,376 rows) with its provider, provider path, SHA-256 digest, size, and per-class box counts (provider revisions per source are in S1 Appendix, Table S18). S8 File contains the validation and evaluation ground truth and 71 sets of per-image predictions from 45 training runs, covering 97 endpoints (the 26 reported in the main text and the other final-checkpoint endpoints of these runs), with scripts that recompute COCO mAP50-95 from those files without any image. For 157 images from two sources (D2 and D7) whose displayed terms do not clearly permit re-release of their annotations under CC BY 4.0, S8 File withholds the 101 boxes and rebuilds them from the providers' own annotation files. S7 Data, released with this revision, supersedes the earlier notes in this package (in `data_schema/` and in the S3 and S6 Data tables) that source-identifying records were omitted.

None of these files can regenerate the training runs, estimate training stochasticity beyond the recorded endpoints, show that filename families are acquisition-source clusters, or establish legal or ethical clearance for the third-party media.

The source exclusion represented in the data is a post-outcome sensitivity analysis. It is not a rights-clearance claim. The available evaluation route was previously consulted and is not described as independent or first-look. The follow-up contrasts jointly index seed, schedule realization, and pipeline conditions and are not interpreted as pure causal effects.

## Directory guide

- `analysis/`: replay code that reads the stored records and recomputes the H2 differences, route differences, and follow-up contrasts and intervals from the endpoints.
- `configs/`: replay configuration plus exact frozen v1/v2 schedules under `frozen_protocol/` and the v3 composite schedules under `factorial_schedules/`.
- `data/`: reader-facing derived tables.
- `data_schema/`: field and interpretation notes.
- `expected_outputs/`: canonical numerical sentinels and generated replay output.
- `figures/`: figure generator, vector sources, and provenance map.
- `tests/`: portable standard-library checks.

`configs/SCHEDULE_SHA256SUMS` binds all three schedule registries to their
frozen archive bytes. The portable tests verify those digests, the number and
150-epoch shape of every schedule, and the byte/structural integrity of
`data/S6_Data.csv`. `tests/test_recomputation.py` checks that every recomputed
value agrees with its stored record at printed precision and that the replay
exits with an error when a stored value or an endpoint is changed.

## Release and citation status

This directory is release `v2.1.0-plos-r1`, accompanying the revised confirmation-audit manuscript (PLOS ONE, PONE-D-26-43559). It is published under the MIT license (see `LICENSE`); see `CITATION.cff` for citation metadata. The content of this release is identical to the `S5_File.zip` Supporting Information archive submitted with the manuscript; the per-file checksums in `SHA256SUMS` bind both copies.

## Release history

- `v2.1.0-plos-r1` (2026-10-20): current release, prepared for the first revision. The scope section now points to S7 Data and S8 File. The replay now recomputes the H2 differences, the evaluation-route differences, and the follow-up contrasts and intervals from the endpoints and compares them with the stored records; `tests/test_recomputation.py` checks that comparison, and `configs/replay_config.json` lists the arms, seeds, cells, and t value it uses. Data files and schedules are unchanged. Fig 3 now prints the validation differences at five decimals, as archived, instead of padding them to six; Fig 1, Fig 2, and Fig 4 are unchanged.
- `v2.0.2-plos-rev2` (2026-08-29): synchronized the repository metadata and README with the PLOS ONE submission; no data or analysis change relative to `v2.0.1-plos-rev2`.
- `v2.0.1-plos-rev2` (2026-08-29): raised figure text sizes to the PLOS 8 pt minimum and regenerated Fig 1-Fig 4; no data or analysis change.
- `v2.0-plos-rev2` (2026-08-29): confirmation-audit release accompanying the manuscript prepared for PLOS ONE. It supersedes the framing in tags `v1.0-submission`, `v1.0.1-submission`, and `v1.1-submission`, whose README described a "locked test", a "single consumed pass", and a randomized follow-up under an earlier journal submission; those descriptions are superseded by the bounded confirmation-audit framing. The old tags are preserved unchanged as a historical record.
