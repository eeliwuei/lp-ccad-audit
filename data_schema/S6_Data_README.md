# S6 Data: schedule, binding, and aggregate audit configuration

`S6_Data.csv` is a processed, machine-readable extraction from the frozen
evidence archive. It was added during the REV2 minimal-data audit to make
scientifically relevant configuration and aggregate audit records easier to
discover without publishing third-party media or source-identifying rows.

## Contents

- all 150 epoch assignments for each of the nine historical v2 schedules;
- all 150 epoch assignments for each of the nine v3 composite-follow-up
  schedule realizations;
- the recorded binding of the 12 follow-up endpoints to the fixed reference or
  v3 schedule realization and training seed;
- the filename-derived resampling count, seed, group count, image count, and
  contrast count;
- original-versus-source-excluded means and changes for all seven historical
  contrasts;
- the retrospective TOST margin and multiplicity-family description;
- aggregate source-exclusion counts and the historical prefix rule;
- aggregate historical admission-audit and later formal-gate records; and
- SHA-256/size locators for the frozen schedule, analysis, test, exclusion, and
  gate artifacts used to build this file.

## Interpretation boundary

This file does not include raw images, annotations, predictions, checkpoints,
source-identifying filenames, basename hashes, or individual gate-intake rows.
It does not establish a dataset licence, publication or redistribution right,
ethics determination, data-controller approval, or reviewer-access mechanism.
Schedule rows represent nominal configurations and do not measure effective
optimization dose. The v3 schedule realizations are bound one-to-one to
training seeds, not crossed over seeds. Resampling remains conditional on
already-trained models and uses filename-derived groups. The TOST configuration
is retrospective and is not evidence of prespecification or equivalence.
The note "source-identifying records intentionally omitted" in the
`evidence_boundary` field describes this file only; S7 Data, released with the
first revision, now provides image-level identifiers.

Historical archive paths containing the former `rights_clean` filename are
retained only as immutable provenance locators; they are not current dataset
names or rights-clearance claims.

## Build provenance

The extraction script is `work/build_rev2_s6_data.rb`. It reads only
`archive/original_read_only/LP_CCAD_CONFIRMATION_FIRST_LATEX_20260828.zip` and
does not modify that archive. No missing source ID, licence, permission, or
institutional fact is inferred.
