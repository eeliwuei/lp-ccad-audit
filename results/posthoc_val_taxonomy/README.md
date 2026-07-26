# Phase-4 validation split: post-hoc error-taxonomy false-positive metric

**What this is.** The **validation-side** counterpart of
`../phase4_test_lock_posthoc_mapped/`. It supplies the one mapped secondary outcome the
manuscript's validation table was missing — the error-taxonomy false-positive quantity
(`error_taxonomy_matched_fp.background_or_spurious` in the frozen decision rule) — for all 18
Phase-4 mainline arm x seed combinations, computed with **exactly the same definition, at the
same operating point, as the test side**, so the two splits are directly comparable.

**This is a post-hoc computation performed on 2026-07-26 that the original validation pass did
not produce as a reportable artefact.** No inference was run. Every number is a deterministic
function of frozen files that existed before this date: the per-arm frozen validation
predictions, the frozen validation labels, and each arm's own frozen validation matched-FP
threshold. Nothing frozen was modified; nothing under `../phase4_test_lock/` was read for arming
or written.

**Why it was needed.** `../../configs/phase4_decision_rule_v1.yaml`
(`four_part_hypothesis_metric_mapping`) maps the four promised outcomes onto six quantities,
because `localization` and `false_positives` each have two members:

| promised outcome | mapped quantities |
|---|---|
| recall | `matched_fp_recall` |
| small objects | `q25_macro_recall_iou50` |
| localization | `unified_val_map75`, `mean_matched_iou` |
| false positives | `error_taxonomy_matched_fp.background_or_spurious`, `op_p90.fp_per_image` |

and it states: *"All four mapped outcomes are reported for every contrast regardless of
direction; selective emphasis is prohibited."* The manuscript reported five quantities per
split and a **different** five on each split — validation omitted the taxonomy FP metric, test
omitted `op_p90.fp_per_image`. This directory removes the validation-side omission. (The
test-side `op_p90.fp_per_image` already exists in
`../phase4_test_lock_posthoc_mapped/mapped_metrics_per_arm.csv`.)

---

## 1. Correction to the stated premise: the numbers already existed, one level down

The task that commissioned this work assumed the validation taxonomy "was never computed for
the KD arms." That is true of the *artefacts a reader would look in*, and false of the
underlying study:

* `../error_taxonomy/` contains only the two Phase-2 C0 baselines
  (`C0_yolo26n_human_only_960_val.json`, `C0s_yolo26s_human_only_960_val.json`) — no KD arm.
* The Phase-4d synthesis record
  (`../phase4_discovery/phase4d_synthesis_full_20260717_120256.json`) carries
  `background_fp_like` as an **empty dict** for every arm it reports.
* **But** `tools/ccad/phase4_freeze_model_full.py` (lines 296-321) *did* run the taxonomy during
  each arm's validation freeze and stored it verbatim at
  `../phase4_selection/<RUN>/selected_checkpoint.json -> candidates[label=="last"].error_taxonomy_matched_fp`.
  All 18 mainline arms have a populated `counts` block there.

**Root cause of the empty `background_fp_like`.** `tools/ccad/phase4d_synthesis.py` line 106:

```python
row["background_fp_like"] = {k: v for k, v in tax.items()
                             if isinstance(v, (int, float)) and "background" in str(k).lower()}
```

`tax` is `error_taxonomy_matched_fp`, whose **top-level** keys are `confidence_threshold`,
`counts`, `confusion_edges`, `localization_errors_by_class`, `fn`. The key `background_fp` lives
one level deeper, inside `counts`. The comprehension therefore never matches anything and
always emits `{}`. This is a one-level-too-shallow reporting bug, not missing data. It also
means the prior test-side README's provenance claim ("the project's own code operationalises
`background_or_spurious` as taxonomy keys containing 'background'") should be read as the
synthesis author's evident *intent* — the code as written never actually reached
`background_fp`. The resolution to `counts["background_fp"]` is unchanged and is the only
reading consistent with both that intent and the freeze record, but nothing in the shipped code
path ever demonstrated it.

**Consequence for this directory: an independent per-arm ground truth exists.** Every value
below was computed from the frozen predictions and then compared against the count block the
original freeze stored. **All 18 arms reproduce all seven taxonomy count keys exactly**
(`reproduces_frozen_freeze_record = True` for every row; column present in the CSV). The
recomputation is therefore verified per arm, not only against the C0 anchor.

---

## 2. Inputs (all read-only, all pre-existing)

| input | path |
|---|---|
| frozen per-arm validation predictions | `../phase4_selection/<RUN>/candidates/last/predictions.json` |
| frozen per-arm validation matched-FP threshold | `../phase4_selection/<RUN>/selected_checkpoint.json -> candidates[label=="last"].matched_fp_threshold` |
| frozen validation GT | `cfg.data.val_list`, loaded by `phase2b_common.load_yolo_gt(cfg, "val", category_base=1)` |
| arm -> run identity | `../phase4_test_lock/test_lock_plan_20260720_0218.json` (`models["<ARM>@<SEED>"].primary.checkpoint`) |
| project config | `../../configs/ccad_project.yaml` |
| C0 sanity ground truth | `../error_taxonomy/C0_yolo26n_human_only_960_val.json`, `../error_taxonomy/C0s_yolo26s_human_only_960_val.json` |
| C0 sanity predictions | `../image_eval/<model>_val/predictions.json` |

The arm -> selection-run mapping is taken from the **same** source the test-side work used (the
lock plan's primary checkpoint path, `parent.parent.name`), so the arm identities on the two
splits are guaranteed identical. For each arm the CSV also records
`checkpoint_sha_matches_lock_plan`; it is `True` for all 18, and
`val_predictions_is_candidates_last` is `True` for all 18 (the `predictions` field in the freeze
record resolves to `candidates/last/predictions.json`).

Validation split as loaded: **2827 images (all 2827 listed images present), 2151 annotations**,
categories `{1: knife, 2: gun, 3: stick}`, final class names `{0: knife, 1: gun, 2: stick}`.

Scoring source file (sha256 at computation time, recorded in `_run_meta.json`):
`tools/ccad/analyze_error_taxonomy.py`
`7731117297c4fed793e7e89898cf8fbf221976a98683e0a561591801013a0e2c` — byte-identical to the file
the test-side re-scoring used and to the one `phase4_freeze_model_full.py` called during the
original freeze.

---

## 3. Exact definition and threshold provenance

For each arm, with `thr` = that arm's own frozen **validation** matched-FP threshold:

```python
preds = phase2b_common.normalize_prediction_rows(
            <arm>/candidates/last/predictions.json, {1,2,3}, "auto", thr)
tax   = analyze_error_taxonomy.classify(val_gt_by_image, preds,
                                        {0:"knife", 1:"gun", 2:"stick"}, 0.0)
background_fp           = tax["counts"]["background_fp"]
background_fp_per_image = background_fp / 2827
```

This is a literal re-execution of the call in `phase4_freeze_model_full.py` lines 296-301, and
the same call the test side made against the test GT. A prediction is counted `background_fp`
when its best same-class IoU is `< 0.1` **and** its best different-class IoU is `< 0.1` — a box
that touches no ground-truth object at all.

**Threshold provenance.** `matched_fp_threshold` is selected by
`tools/ccad/phase4c_select_and_freeze_c0r.py::matched_fp` on the arm's own single-pass
validation operating curve as the point minimising `|fp_per_image - 0.05235231694375663|`
(ties: higher recall, higher precision, lower threshold). `0.05235231694375663` is the C0-Std
reference fp/image (`C0_yolo26n_human_only_960` validation OP-P90, `148/2827`), recorded
identically in all 33 freeze directories and re-verified here per arm
(`reference_fp_per_image_C0Std` column). This is exactly the field the test-side work used, so
the two splits are evaluated at the same per-arm operating point. Note the asymmetry that
follows from the project's own test discipline: on validation this threshold is *in-sample* (it
was selected on this split), whereas on test the same threshold is transferred.

**Alternative aggregate.** `background_fp + partial_wrong_overlap` is reported alongside,
because `background_or_spurious` is not a literal key anywhere in the taxonomy (a
repository-wide grep finds it only on line 91 of the decision rule) and folding
`partial_wrong_overlap` (a prediction overlapping a wrong-class GT at `0.1 <= IoU < 0.5`) into
"spurious" is an equally literal reading of the English. On the locked test the two readings
disagree in sign for H2 and H6a.

---

## 4. Sanity check against the only available ground truth

`../error_taxonomy/C0_yolo26n_human_only_960_val.json` (and the `C0s` companion) were produced
by the original study through `analyze_error_taxonomy.py`'s own CLI at `conf = 0.25` on
`../image_eval/<model>_val/predictions.json`. Recomputed here through the identical code path:

| model | key | stored | recomputed | match |
|---|---|---:|---:|:--:|
| C0_yolo26n_human_only_960 | TP | 1448 | 1448 | yes |
| | localization_error | 128 | 128 | yes |
| | class_confusion | 25 | 25 | yes |
| | duplicate | 98 | 98 | yes |
| | partial_wrong_overlap | 5 | 5 | yes |
| | **background_fp** | **59** | **59** | **yes** |
| | FN | 703 | 703 | yes |
| C0s_yolo26s_human_only_960 | **background_fp** | **44** | **44** | **yes** |

All seven count keys, `fn`, `confusion_edges` (full ordered list) and
`localization_errors_by_class` match exactly for **both** baselines. The script aborts before
writing any arm numbers if this check fails.

These two C0 files are the **Phase-2 baseline models at conf 0.25**, not the Phase-4 `C0-R`
replicate arms at their matched-FP thresholds; they validate the pipeline, they are not the
same quantity as the `C0-R` rows below.

---

## 5. Per-arm results (validation split, 2827 images)

| arm | seed | matched-FP thr | background_fp | bg per image | partial_wrong_overlap | alt count | alt per image | TP | dup | class_conf | loc_err | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0-R | 42 | 0.37723 | 36 | 0.012734 | 4 | 40 | 0.014149 | 863 | 32 | 23 | 53 | 1288 |
| C0-R | 1337 | 0.35283 | 46 | 0.016272 | 1 | 47 | 0.016625 | 880 | 23 | 23 | 55 | 1271 |
| C0-R | 20260703 | 0.40404 | 40 | 0.014149 | 3 | 43 | 0.015210 | 855 | 29 | 25 | 51 | 1296 |
| C1-M | 42 | 0.39646 | 39 | 0.013796 | 3 | 42 | 0.014857 | 882 | 28 | 23 | 55 | 1269 |
| C1-M | 1337 | 0.39841 | 34 | 0.012027 | 2 | 36 | 0.012734 | 896 | 30 | 18 | 64 | 1255 |
| C1-M | 20260703 | 0.39365 | 22 | 0.007782 | 3 | 25 | 0.008843 | 836 | 28 | 28 | 67 | 1315 |
| C2-M | 42 | 0.28915 | 37 | 0.013088 | 1 | 38 | 0.013442 | 859 | 36 | 23 | 51 | 1292 |
| C3-M | 42 | 0.44026 | 32 | 0.011319 | 0 | 32 | 0.011319 | 829 | 42 | 30 | 44 | 1322 |
| CBox-M | 42 | 0.38791 | 39 | 0.013796 | 1 | 40 | 0.014149 | 867 | 28 | 18 | 62 | 1284 |
| C4-M | 42 | 0.49152 | 27 | 0.009551 | 3 | 30 | 0.010612 | 865 | 32 | 28 | 58 | 1286 |
| C4-M | 1337 | 0.29008 | 29 | 0.010258 | 0 | 29 | 0.010258 | 852 | 48 | 21 | 50 | 1299 |
| C4-M | 20260703 | 0.36754 | 31 | 0.010966 | 3 | 34 | 0.012027 | 877 | 39 | 21 | 54 | 1274 |
| C4R-M | 42 | 0.33423 | 31 | 0.010966 | 2 | 33 | 0.011673 | 832 | 40 | 19 | 56 | 1319 |
| C4R-M | 1337 | 0.31784 | 38 | 0.013442 | 1 | 39 | 0.013796 | 893 | 33 | 26 | 50 | 1258 |
| C4MixFT-M | 42 | 0.47158 | 43 | 0.015210 | 2 | 45 | 0.015918 | 841 | 32 | 20 | 51 | 1310 |
| C4MixFT-M | 1337 | 0.42550 | 39 | 0.013796 | 4 | 43 | 0.015210 | 911 | 27 | 23 | 55 | 1240 |
| C4Mix-M | 42 | 0.45936 | 40 | 0.014149 | 3 | 43 | 0.015210 | 826 | 26 | 20 | 59 | 1325 |
| C4Mix-M | 1337 | 0.40127 | 29 | 0.010258 | 6 | 35 | 0.012381 | 890 | 32 | 22 | 59 | 1261 |

`background_fp` spans 22-46 counts over 2827 images (0.0078-0.0163 per image);
`partial_wrong_overlap` spans 0-6. Every row reproduces the count block frozen in its own
`selected_checkpoint.json`. **No arm was missing**: all 18 had locatable frozen validation
predictions.

---

## 6. Seven registered contrasts (treatment minus reference, lower is better)

Mean over the contrast's seeds. Counts are absolute (2827-image denominator); per-image values
are also in `val_taxonomy_contrasts.json`.

| contrast | seeds | mean delta, `background_fp` (reported) | per image | mean delta, `background_fp + partial_wrong_overlap` (alt) | per image | signs agree |
|---|---|---:|---:|---:|---:|:--:|
| H1 C4-M - C1-M | 42, 1337, 20260703 | -2.667 | -0.000943 | -3.333 | -0.001179 | yes |
| H2 C4-M - C4MixFT-M | 42, 1337 | -13.000 | -0.004599 | -14.500 | -0.005129 | yes |
| H3 C4MixFT-M - C4Mix-M | 42, 1337 | +6.500 | +0.002299 | +5.000 | +0.001769 | yes |
| H4 C4-M - C4R-M | 42, 1337 | -6.500 | -0.002299 | -6.500 | -0.002299 | yes |
| H5 C1-M - C0-R | 42, 1337, 20260703 | -9.000 | -0.003184 | -9.000 | -0.003184 | yes |
| H6a C2-M - C1-M | 42 | -2.000 | -0.000707 | -4.000 | -0.001415 | yes |
| H6b C3-M - C1-M | 42 | -7.000 | -0.002476 | -10.000 | -0.003537 | yes |

Per-seed deltas, `background_fp` count:

| contrast | 42 | 1337 | 20260703 |
|---|---:|---:|---:|
| H1 C4-M - C1-M | -12 | -5 | +9 |
| H2 C4-M - C4MixFT-M | -16 | -10 | - |
| H3 C4MixFT-M - C4Mix-M | +3 | +10 | - |
| H4 C4-M - C4R-M | -4 | -9 | - |
| H5 C1-M - C0-R | +3 | -12 | -18 |
| H6a C2-M - C1-M | -2 | - | - |
| H6b C3-M - C1-M | -7 | - | - |

Per-seed deltas, alternative `background_fp + partial_wrong_overlap`:

| contrast | 42 | 1337 | 20260703 |
|---|---:|---:|---:|
| H1 | -12 | -7 | +9 |
| H2 | -15 | -14 | - |
| H3 | +2 | +8 | - |
| H4 | -3 | -10 | - |
| H5 | +2 | -11 | -18 |
| H6a | -4 | - | - |
| H6b | -10 | - | - |

**The two readings agree in sign on all seven contrasts on validation**, unlike the locked test,
where they disagree for H2 (-2.500 vs +1.000) and H6a (0.000 vs -4.000). The validation split
therefore does not reproduce the test split's reading-dependence — worth stating, because it
means the sign ambiguity flagged in `../phase4_test_lock_posthoc_mapped/README.md` is a property
of the test measurement rather than of the metric definition in general. It does not remove the
obligation to say which reading a write-up uses.

Note also that H1 and H5 are not seed-consistent even within validation: H1 is negative at seeds
42 and 1337 and positive at 20260703, H5 positive at 42 and negative at the other two.

---

## 7. Validation vs locked test, side by side (mean delta, counts)

| contrast | TEST `background_fp` | VAL `background_fp` | same sign | TEST alt | VAL alt | same sign |
|---|---:|---:|:--:|---:|---:|:--:|
| H1 C4-M - C1-M | -6.667 | -2.667 | yes | -7.000 | -3.333 | yes |
| H2 C4-M - C4MixFT-M | -2.500 | -13.000 | yes | +1.000 | -14.500 | **no** |
| H3 C4MixFT-M - C4Mix-M | -7.000 | +6.500 | **no** | -4.500 | +5.000 | **no** |
| H4 C4-M - C4R-M | -7.500 | -6.500 | yes | -1.500 | -6.500 | yes |
| H5 C1-M - C0-R | -4.000 | -9.000 | yes | -3.333 | -9.000 | yes |
| H6a C2-M - C1-M | 0.000 | -2.000 | (test is exactly 0) | -4.000 | -4.000 | yes |
| H6b C3-M - C1-M | -11.000 | -7.000 | yes | -17.000 | -10.000 | yes |

Counts are single- to low-double-digit over 2827 (val) / 2553 (test) images, the two splits use
different denominators, and the thresholds are in-sample on validation but transferred on test.
Neither split supports a strong false-positive claim in either direction, and H3 reverses sign
across splits under both readings. This table is context for the write-up, not a verdict: the
confirmatory verdict comes from the confirmatory metric only, and nothing here re-opens it.

---

## 8. Files

| file | contents |
|---|---|
| `val_taxonomy_per_arm.csv` | 18 arms x threshold, `background_fp` count and per-image, `partial_wrong_overlap`, alt aggregate count and per-image, full taxonomy breakdown (TP / duplicate / class_confusion / localization_error / FN), validation image count, per-arm provenance (run, predictions path + sha256, checkpoint sha match), the freeze record's own `background_fp` and the `reproduces_frozen_freeze_record` flag, plus the validation-side values the freeze already carried (`matched_fp_recall`, `op_p90.fp_per_image`, `unified_val_map75`, `q25_macro_recall_iou50`, `mean_matched_iou`) |
| `val_taxonomy_contrasts.json` | the seven registered contrasts: per-seed and mean delta of `background_fp` count and per-image, and of the alternative aggregate; both arms' raw values; per-contrast sign agreement between the two readings; the C0 sanity block |
| `compute_val_taxonomy.py` | the exact script that produced them |
| `_per_arm_rows.json`, `_run_meta.json` | machine-readable side-cars (per-arm confusion edges and localization-by-class breakdowns live in `_per_arm_rows.json`) |

**Interpretation discipline (unchanged, from the decision rule).** These are *secondary*
outcomes. They qualify the headline verdict and never overturn it, and all mapped outcomes are
reported for every contrast regardless of direction. With this directory, both splits can now
show all six mapped quantities.
