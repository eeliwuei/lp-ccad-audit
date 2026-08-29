# LP-CCAD multi-axis confirmation audit: analytical replay

This repository accompanies the manuscript **“A multi-axis confirmation audit of label-space curriculum distillation under evaluation and pipeline variation.”** It replays the derived numerical record used in the manuscript and regenerates the four publication figures from machine-readable tables.

## What this release supports

- Recalculation of the protocol-specified fixed-pair validation mean from the available seed endpoints.
- Recalculation of the original and source-excluded evaluation summaries recorded in the frozen evidence archive.
- Recalculation of the 12-run composite follow-up contrasts and their descriptive intervals.
- Inspection of filename-derived grouping resamples and retrospective equivalence calculations.
- Inspection of exact v2/v3 epoch schedules, the 12 run--schedule bindings, all seven original-to-source-excluded contrast triples, resampling/TOST parameters, and aggregate admission-gate records.
- Regeneration of Fig 1-Fig 4 from the derived data release.

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

This is an **analytical replay**, not end-to-end training or raw-media-dependent evaluation. It does not include raw third-party image/video media, source-identifying filenames, ground-truth annotation files, per-image prediction files, checkpoints, or a validated institutional data-access route. Consequently, it cannot independently reconstruct COCO metrics from images and predictions, estimate training stochasticity beyond the recorded endpoints, validate filename families as acquisition-source clusters, or establish legal/ethical clearance.

The source exclusion represented in the data is a post-outcome sensitivity analysis. It is not a rights-clearance claim. The available evaluation route was previously consulted and is not described as independent or first-look. The follow-up contrasts jointly index seed, schedule realization, and pipeline conditions and are not interpreted as pure causal effects.

## Directory guide

- `analysis/`: aggregate replay code.
- `configs/`: replay configuration plus exact frozen v1/v2 schedules under `frozen_protocol/` and the v3 composite schedules under `factorial_schedules/`.
- `data/`: reader-facing derived tables.
- `data_schema/`: field and interpretation notes.
- `expected_outputs/`: canonical numerical sentinels and generated replay output.
- `figures/`: figure generator, vector sources, and provenance map.
- `tests/`: portable standard-library checks.

`configs/SCHEDULE_SHA256SUMS` binds all three schedule registries to their
frozen archive bytes. The portable tests verify those digests, the number and
150-epoch shape of every schedule, and the byte/structural integrity of
`data/S6_Data.csv`.

## Release and citation status

This directory is release `v2.0.2-plos-rev2`, accompanying the confirmation-audit manuscript prepared for PLOS ONE. It is published under the MIT license (see `LICENSE`); see `CITATION.cff` for citation metadata. The content of this release is identical to the `S5_File.zip` Supporting Information archive submitted with the manuscript; the per-file checksums in `SHA256SUMS` bind both copies.

## Release history

- `v2.0.2-plos-rev2` (2026-08-29): current release. Synchronizes the repository metadata and README with the PLOS ONE submission; no data or analysis change relative to `v2.0.1-plos-rev2`.
- `v2.0.1-plos-rev2` (2026-08-29): raised figure text sizes to the PLOS 8 pt minimum and regenerated Fig 1-Fig 4; no data or analysis change.
- `v2.0-plos-rev2` (2026-08-29): confirmation-audit release accompanying the manuscript prepared for PLOS ONE. It supersedes the framing in tags `v1.0-submission`, `v1.0.1-submission`, and `v1.1-submission`, whose README described a "locked test", a "single consumed pass", and a randomized follow-up under an earlier journal submission; those descriptions are superseded by the bounded confirmation-audit framing. The old tags are preserved unchanged as a historical record.
