# Phase-4 locked-test: post-hoc mapped secondary outcomes

**What this is.** This directory contains a **post-hoc re-scoring of the frozen prediction
files produced by the already-consumed Phase-4 locked-test pass**, performed on
**2026-07-26**. No model inference was run on the test split. The one-time execution lock
(`../phase4_test_lock/EXECUTION_LOCK.json`, status `CONSUMED` at `2026-07-20T02:22:20.998623+00:00`)
was neither read-for-arming nor modified, and nothing under `../phase4_test_lock/` was written.
Every number below is a deterministic function of files that already existed before this date.

**Why it was needed.** The frozen decision rule
`../../configs/frozen_protocol/phase4_decision_rule_v1.yaml` (`four_part_hypothesis_metric_mapping`,
lines 87-98; sha256 `e82c7bbd...4841472`) requires four mapped secondary outcomes to be
reported for *every* confirmatory contrast. The consumption event of 2026-07-20
(`tools/ccad_ops/phase4_test_executor.py`, sha256 `5603bd8b...03e1e24`) ran, per arm, only:

* `tools/ccad/eval_coco_unified.py` -> `unified/metrics.json`
  (COCOeval `mAP50-95 / mAP50 / mAP75 / AP_small / AP_medium / AP_large / AR100` + per-class,
  plus simple `match_metrics`: `Recall@0.5`, `Precision@0.5`, `Recall@0.75`, `Precision@0.75`,
  `FP/image@0.5`, `FN/image@0.5`), and
* the frozen decision rule on the **confirmatory** metric only (`test_results_final.json`).

It did **not** store `matched_fp_recall`, `q25_macro_recall_iou50`, `mean_matched_iou`,
`error_taxonomy_matched_fp.*`, or `op_p90.fp_per_image`. Those are recomputed here from the
frozen per-arm `v/predictions.json`, the frozen test labels, and the frozen per-arm
validation thresholds.

---

## 1. Inputs (all read-only, all pre-existing)

| input | path |
|---|---|
| frozen per-arm test predictions | `../phase4_test_lock/results/<ARM>__primary/v/predictions.json` |
| frozen per-arm COCOeval output (source of `map75`) | `../phase4_test_lock/results/<ARM>__primary/unified/metrics.json` |
| frozen test GT (YOLO labels via frozen split list) | `cfg.data.test_list`, loaded by `phase2b_common.load_yolo_gt(cfg,"test",category_base=1)` |
| frozen per-arm validation thresholds | `../phase4_selection/<RUN>/selected_checkpoint.json` -> `candidates[label=="last"]` |
| frozen train-derived Q25 thresholds | `../size_thresholds_train.json` (sha256 `da33e361...58470cc`) |
| lock plan (arm -> checkpoint identity) | `../phase4_test_lock/test_lock_plan_20260720_0218.json` |
| project config | `../../configs/ccad_project.yaml` (sha256 `0bbef382...795e8548`) |

Scoring source files, with the sha256 in effect at re-scoring time (the first six are the
files the lock plan pins as `evaluator.sources`; those five all still match their pinned sha):

| file | sha256 |
|---|---|
| `tools/ccad/eval_coco_unified.py` | `fbf07e64187a07111def5c82a14592b8d006e01296373d9122f5cdea90ebca94` (pinned, matches) |
| `tools/ccad/eval_size_stratified.py` | `ceccaf0de6b1afe02471da91dbf2686bd6bb093a75bcc27cef4233266f12cb23` (pinned, matches) |
| `tools/ccad/phase4_freeze_model_full.py` | `2e766ab6ed0b43d4bd2755141a4e38fdd30072aea6e8c6e2ff2ae228c6cb1987` (pinned, matches) |
| `tools/ccad/phase4_freeze_model_full_v2.py` | `a8817170ea48fdfeeaea52c3cc008fa106b7ac568bf47ed1e769d0351a58106c` (pinned, matches) |
| `tools/ccad/phase4d_synthesis.py` | `18682599defe8daf8180c6a9ea9ad2a9b0a279b9593df4ef55a21ff30a57e1c5` (pinned, matches) |
| `tools/ccad/phase4c_select_and_freeze_c0r.py` | `2fccf8b1054e8c291222868394184d723c1733c660b83b1f6c43a0a097da5e34` |
| `tools/ccad/phase2b_common.py` | `b35c8b86ac2b2207889f80a3a0f9799a12233dc97cb4015dcf2abf747d82e9ea` |
| `tools/ccad/analyze_error_taxonomy.py` | `7731117297c4fed793e7e89898cf8fbf221976a98683e0a561591801013a0e2c` |
| `tools/ccad/eval_operating_points.py` | `89b911bcf10027f03b3f31fa103374a3082b6c5a77e806aeebd0e78072051f79` |

Test split as loaded: **2553 images (all present), 2088 annotations**, categories
`{1: knife, 2: gun, 3: stick}`. Predictions are normalised with
`phase2b_common.normalize_prediction_rows(pred, {1,2,3}, "auto", 0.0)` -- exactly the
category-base and score-floor defaults that `eval_coco_unified.py` used inside the
consumed pass.

Arm identity was re-verified: for all 18 arms the `last.pt` checkpoint sha256 recorded in
the validation freeze equals the `sha256_frozen` recorded for `primary` in the lock plan
(`checkpoint_sha_matches_lock_plan = True` for every row).

---

## 2. Exact definitions used (one line each, with source)

1. **`matched_fp_recall`** -- recall on the test split of the predictions surviving this arm's
   **frozen validation matched-FP threshold**; matching is greedy, class-aware, IoU >= 0.5, in
   descending score order, one GT per prediction.
   *Threshold source*: `tools/ccad/phase4c_select_and_freeze_c0r.py::matched_fp` (used by
   `tools/ccad/phase4_freeze_model_full.py::freeze_model`), which picks, from the arm's
   single-pass **validation** operating curve, the point minimising
   `|fp_per_image - reference_fp_per_image|` (ties: higher recall, higher precision, lower
   threshold). Recorded per arm as `candidates[label=="last"].matched_fp_threshold`.
   *Test scoring*: `tools/ccad/eval_operating_points.py::evaluate_at_threshold(cfg,"test",...)`.

2. **`q25_macro_recall_iou50`** -- `tools/ccad/eval_size_stratified.py::eval_q25(cfg,"test",pred,thresholds,"auto")`,
   field `Q25_Recall@0.5`, macro-averaged over the three classes by
   `phase4c_select_and_freeze_c0r.py::macro`. The Q25 target set is every test GT box whose
   **relative** area (`area / (width*height)`) is `<= q25_area_ratio`, where `q25_area_ratio`
   is the per-class 25th percentile of *training* GT relative areas
   (`eval_size_stratified.py::train_q25_thresholds`, frozen in `../size_thresholds_train.json`:
   knife `0.013767246528`, gun `0.01307373046875`, stick `0.0458933999999`). Larger same-class
   GT are **ignore** regions (predictions matching them are dropped, not counted FP)
   -- `eval_size_stratified.py::pr_curve_with_ignore`. Test Q25 GT counts: knife 58, gun 275, stick 36.

3. **`map75`** -- COCOeval `mAP75` (`stats[2]`), **copied verbatim** from the arm's stored
   `unified/metrics.json` written by the consumed pass. This is the test-split analogue of the
   decision rule's `unified_val_map75`; the rule names the validation field because it was
   written for the validation freeze. Same evaluator (`faster-coco-eval` fallback in
   `phase2b_common.evaluate_coco`), same GT/prediction COCO files -- nothing was recomputed.

4. **`mean_matched_iou`** -- mean IoU over the matched pairs returned by
   `phase2b_common.greedy_match(gt_by_image_test, preds, iou_thr=0.5, class_aware=True)`, i.e.
   the `mean_matched_iou@0.5` field of `phase4c_select_and_freeze_c0r.py::unified_for_pred`.
   All predictions participate (score floor 0.0).

5. **`taxonomy_background_or_spurious_count` / `_per_image`** -- `counts["background_fp"]` from
   `tools/ccad/analyze_error_taxonomy.py::classify(gt_by_image_test, preds>=matched_fp_threshold,
   {0:"knife",1:"gun",2:"stick"}, 0.0)`, i.e. a surviving prediction whose best same-class IoU
   is `< 0.5` **and** `< 0.1`, and whose best different-class IoU is `< 0.5` and `< 0.1` -- a box
   that touches no GT at all. `_per_image` divides by 2553. Threshold and call signature are
   identical to `phase4_freeze_model_full.py`'s `error_taxonomy_matched_fp`.
   **See the caveat in section 5** -- `background_or_spurious` is not a literal key anywhere.

6. **`op_p90_fp_per_image`** -- FP per test image at this arm's **frozen validation OP-P90
   threshold** (the validation operating point with `precision >= 0.90` maximising recall, then
   minimising fp/image, then minimising threshold -- `phase4c_select_and_freeze_c0r.py::op_p90`,
   recorded as `candidates[label=="last"].op_p90.threshold`). Test scoring again via
   `eval_operating_points.py::evaluate_at_threshold(cfg,"test",...)`; denominator 2553.

---

## 3. C0-Std reference fp/image

**Value used: `0.05235231694375663`.**

* C0-Std is the model `C0_yolo26n_human_only_960` (standard-augmentation YOLO26n, human-only
  pretrain, 960 px), predictions at
  `../image_eval/C0_yolo26n_human_only_960_val/predictions.json`.
* The value is its **validation** OP-P90 fp/image: threshold `0.46578`, precision `0.900067521944632`,
  tp 1333, fp 148, over 2827 validation images -> `148/2827 = 0.05235231694375663`.
* Recorded identically in **every** Phase-4 arm freeze, at
  `../phase4_selection/<RUN>/selected_checkpoint.json` ->
  `selection_rule.reference_fp_model = "C0_yolo26n_human_only_960"`,
  `selection_rule.reference_fp_per_image = 0.05235231694375663`, and at
  `frozen_val_thresholds.json -> reference_c0_std_op_p90.fp_per_image`. All 33 freeze
  directories agree on this single value.

**A second, near-identical value exists and was deliberately NOT used:**
`0.051998585072515036`, in `../operating_points/operating_point_metrics.json`
(`reference_val_fp_per_image`) and copied into `../teacher_selection/frozen_val_thresholds.json`
(`reference_fp_per_image`). It is the *same quantity for the same model on the same split*, but
computed by the older `eval_operating_points.py::select_op_p90`, which searches a **discretised
grid of <=1000 candidate thresholds** instead of the exact per-score operating curve used by
`phase4c_select_and_freeze_c0r.py::op_p90`. The Phase-4 arm freezes -- which are what the
decision rule's mapping governs -- were all built against `0.05235231694375663`, so that is the
reference used here. The `0.0519985...` value belongs to the Phase-2 baseline / teacher-gate path.

---

## 4. Validation-threshold transfer (why thresholds are not re-selected on test)

`matched_fp_recall` and `op_p90.fp_per_image` need a confidence threshold. The project's own
test discipline is explicit and consistent: thresholds are selected on validation only and then
frozen for test.

* `tools/ccad/eval_operating_points.py` (report text): *"OP-P90 thresholds are selected on
  validation only, then frozen for test."* -- `select_op_p90`/`select_matched_fp` run on
  `<model>_val/predictions.json`, then `evaluate_at_threshold(..., "test", ..., val_threshold)`.
* `tools/ccad/run_selected_teacher_test_once.py` does the same for the teacher's locked test
  (`op_p90_threshold_from_val`, `matched_fp_threshold_from_val`).
* `tools/ccad/run_phase4_test_once.py` records `frozen_validation_thresholds` in the pre-test lock.

So the reported `matched_fp_recall` is measured at the arm's own validation matched-FP threshold
applied to test. Because that threshold does not reproduce the reference fp/image *on test*, the
CSV also reports `matched_fp_test_fp_per_image` (the realised test fp/image at that threshold,
0.0556-0.0779 across arms) so the transfer gap is visible.

The CSV additionally carries **explicitly labelled diagnostics** with the `diag_test_reselected_*`
prefix: what the matched-FP / OP-P90 operating points would be if re-selected *on the test
curve itself*. These are **not** the frozen mapped metric and must not be substituted for it;
they are included only so a reader can see that re-selection on test drives every arm to the
same `0.0524873` fp/image (134/2553) and lowers recall for every arm, by -0.0345 to -0.0057.

---

## 5. What could NOT be faithfully recovered

**`error_taxonomy_matched_fp.background_or_spurious` -- the literal key does not exist anywhere
in the project.** The taxonomy producer (`analyze_error_taxonomy.py::classify`, the function
`phase4_freeze_model_full.py` calls to fill `error_taxonomy_matched_fp`) emits exactly these
count keys: `TP`, `duplicate`, `class_confusion`, `localization_error`, `partial_wrong_overlap`,
`background_fp`, `FN`. A repository-wide grep for `background_or_spurious` returns exactly one
hit: line 91 of the decision rule itself.

It was resolved -- **not invented** -- using the project's own resolution of that same name.
`tools/ccad/phase4d_synthesis.py` (pinned in the lock plan's `evaluator.sources`) is the script
that reports the mapped outcomes, and at line 106 it defines

```python
row["background_fp_like"] = {k: v for k, v in tax.items()
                             if isinstance(v, (int, float)) and "background" in str(k).lower()}
```

i.e. it operationalises the decision rule's `background_or_spurious` as every taxonomy key
containing "background", which in this taxonomy is the single key `background_fp`. That is the
value reported here.

**Residual ambiguity, and it is NOT harmless.** An equally literal reading of the English
phrase would fold in `partial_wrong_overlap` (a prediction overlapping a wrong-class GT at
0.1 <= IoU < 0.5 -- arguably "spurious"). The project's own code does not do this, so the
reported metric does not either. But this is a real fork, not a rounding detail: across the 18
arms `partial_wrong_overlap` runs 5-15 counts against a `background_fp` of 14-40, and **the two
readings disagree in sign on two of the seven contrasts**:

| contrast | mean delta, `background_fp` only (reported) | mean delta, `background_fp + partial_wrong_overlap` |
|---|---:|---:|
| H1 C4-M - C1-M | -6.667 | -7.000 |
| **H2 C4-M - C4MixFT-M** | **-2.500** | **+1.000** |
| H3 C4MixFT-M - C4Mix-M | -7.000 | -4.500 |
| H4 C4-M - C4R-M | -7.500 | -1.500 |
| H5 C1-M - C0-R | -4.000 | -3.333 |
| **H6a C2-M - C1-M** | **0.000** | **-4.000** |
| H6b C3-M - C1-M | -11.000 | -17.000 |

Both readings are therefore reported: `mapped_metrics_per_arm.csv` carries
`taxonomy_background_or_spurious_count` (the reported metric),
`taxonomy_partial_wrong_overlap`, and the pre-summed
`alt_background_plus_partial_count` / `_per_image`; `mapped_metrics_contrasts.json` carries an
`alternative_reading_background_plus_partial_wrong_overlap` block per contrast. Given the sign
disagreement on H2 and H6a, **any write-up of the false-positive outcome for those two contrasts
should state which reading it uses.** Counts are small (single- to low-double digits over 2553
images), so neither reading supports a strong FP claim in either direction.

No other mapped metric was unrecoverable.

**Two lesser, disclosed substitutions** (both mechanical, neither discretionary):

* `unified_val_map75` -> the test-split `mAP75` already stored by the consumed pass. The metric
  contract's confirmatory estimand is standard COCOeval; the field name in the mapping says
  "val" only because the mapping was authored for the validation freeze. There is no test
  `unified_val_map75`; substituting the test mAP75 is the only sane reading, and it is a copy,
  not a recomputation.
* The frozen scoring functions bind their split at the call site (`load_yolo_gt(cfg,"val",...)`,
  `split_images(cfg,"val")` inside `phase4c_select_and_freeze_c0r.operating_curve` and
  `unified_for_pred`). For this run those two loaders were re-pointed at the **test** split and
  memoised (the test GT is loaded once and reused across all 18 arms). The arithmetic executed
  is the unmodified frozen bytecode; only the split and the caching changed. The
  Q25 GT/ignore set builder (`eval_size_stratified.build_q25_gt_sets`) was memoised the same way
  -- it is prediction-independent, so this is a pure speedup.

---

## 6. Sanity anchor

`mAP50-95` for `C4-M@42` primary, recomputed from the frozen
`unified/test_gt_coco.json` + `unified/predictions_coco.json` via `phase2b_common.evaluate_coco`
(evaluator: `faster-coco-eval` 1.7.2; `pycocotools` in `envs/ccad_yolo26n` has a numpy ABI
mismatch and raises, so the frozen code path falls through to `faster-coco-eval`, exactly as it
did in the consumed pass):

```
recomputed = 0.4043930717327209  ->  round(.,5) = 0.40439   MATCH
```

---

## 7. Files

| file | contents |
|---|---|
| `mapped_metrics_per_arm.csv` | 18 mainline arms x all four mapped outcomes, plus thresholds, reference fp/image, realised test fp/image, full error-taxonomy breakdown, stored COCOeval values, `diag_test_reselected_*` diagnostics, and the corresponding validation-side frozen values |
| `mapped_metrics_contrasts.json` | the seven frozen confirmatory contrasts (H1, H2, H3, H4, H5, H6a, H6b): per-seed deltas, mean delta, and both arms' raw values for every mapped metric, grouped by the decision rule's four parts, plus the alternative `background_fp + partial_wrong_overlap` reading per contrast |
| `_per_arm_rows.json`, `_run_meta.json` | machine-readable side-cars for the two files above |
| `compute_mapped_metrics.py` | the exact script that produced them |

**Interpretation discipline (unchanged, from the decision rule).** These are *secondary*
outcomes. The headline verdict comes from the confirmatory metric only; secondaries qualify it
but never overturn it, and all four mapped outcomes are reported for every contrast regardless
of direction. Nothing in this directory re-opens the test set, and no verdict is restated here.
