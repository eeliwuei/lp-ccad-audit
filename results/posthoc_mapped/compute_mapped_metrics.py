#!/usr/bin/env python3
"""Post-hoc re-scoring of the FROZEN, already-consumed Phase-4 locked-test predictions.

No inference is run. Only frozen prediction files, frozen GT and frozen
validation thresholds are read. Nothing under phase4_test_lock/ is written.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import os
# Set LPCCAD_PROJECT_ROOT to the restricted project tree (available under the
# data-use agreement); the server path it had at run time is not published.
ROOT = Path(os.environ.get("LPCCAD_PROJECT_ROOT", "."))
sys.path.insert(0, str(ROOT / "tools/ccad"))

from ccad_core import exp_dir, load_config                                   # noqa: E402
from phase2b_common import (                                                 # noqa: E402
    coco_category_ids,
    evaluate_coco,
    greedy_match,
    load_yolo_gt,
    normalize_prediction_rows,
    split_images,
)
import eval_size_stratified as ess                                           # noqa: E402
import eval_operating_points as eop                                          # noqa: E402
from analyze_error_taxonomy import classify as classify_errors               # noqa: E402
import phase4c_select_and_freeze_c0r as base                                 # noqa: E402

CFG = ROOT / "experiments/ccad_yolo26n/configs/ccad_project.yaml"
cfg = load_config(str(CFG))
EXP = exp_dir(cfg)
LOCKDIR = EXP / "eval/phase4_test_lock"
RES = LOCKDIR / "results"
PLAN = LOCKDIR / "test_lock_plan_20260720_0218.json"
OUT = EXP / "eval/phase4_test_lock_posthoc_mapped"
OUT.mkdir(parents=True, exist_ok=True)

FINAL_NAMES = {0: "knife", 1: "gun", 2: "stick"}
PRED_BASE = "auto"          # eval_coco_unified.py default, as used by the consumed executor
SCORE_FLOOR = 0.0           # eval_coco_unified.py default --score-threshold

ARMS = [
    ("C0-R", "42"), ("C0-R", "1337"), ("C0-R", "20260703"),
    ("C1-M", "42"), ("C1-M", "1337"), ("C1-M", "20260703"),
    ("C2-M", "42"), ("C3-M", "42"), ("CBox-M", "42"),
    ("C4-M", "42"), ("C4-M", "1337"), ("C4-M", "20260703"),
    ("C4R-M", "42"), ("C4R-M", "1337"),
    ("C4MixFT-M", "42"), ("C4MixFT-M", "1337"),
    ("C4Mix-M", "42"), ("C4Mix-M", "1337"),
]

CONTRASTS = [
    ("H1_core_annealing_vs_full_kd", "C4-M", "C1-M", ["42", "1337", "20260703"]),
    ("H2_order_given_fixed_tail", "C4-M", "C4MixFT-M", ["42", "1337"]),
    ("H3_terminal_tail_content", "C4MixFT-M", "C4Mix-M", ["42", "1337"]),
    ("H4_direction_joint", "C4-M", "C4R-M", ["42", "1337"]),
    ("H5_kd_usefulness", "C1-M", "C0-R", ["42", "1337", "20260703"]),
    ("H6a_projection_single_view", "C2-M", "C1-M", ["42"]),
    ("H6b_projection_group_view", "C3-M", "C1-M", ["42"]),
]

ALT_METRICS = [
    "alt_background_plus_partial_count",
    "alt_background_plus_partial_per_image",
]

MAPPED_METRICS = [
    "matched_fp_recall",
    "q25_macro_recall_iou50",
    "map75",
    "mean_matched_iou",
    "taxonomy_background_or_spurious_count",
    "taxonomy_background_or_spurious_per_image",
    "op_p90_fp_per_image",
]

FOUR_PART = {
    "recall": ["matched_fp_recall"],
    "small_objects": ["q25_macro_recall_iou50"],
    "localization": ["map75", "mean_matched_iou"],
    "false_positives": ["taxonomy_background_or_spurious_count",
                        "taxonomy_background_or_spurious_per_image",
                        "op_p90_fp_per_image"],
}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- frozen inputs
plan = json.loads(PLAN.read_text())
cat_ids = coco_category_ids(cfg)
cat_set = set(cat_ids)

log("loading frozen test GT (labels only, no inference)")
GT_IMAGES, GT_ANNS, GT_BY_IMAGE = load_yolo_gt(cfg, "test", category_base=1)
IMAGE_COUNT = len([p for p in split_images(cfg, "test") if p.exists()])
log(f"test GT: {len(GT_IMAGES)} images ({IMAGE_COUNT} present), {len(GT_ANNS)} annotations")

# Cache the (prediction-independent) heavy frozen loaders so the frozen scoring
# functions below run on exactly the same objects without re-reading 2553 images.
eop.load_yolo_gt = lambda _cfg, _split, category_base=1: (GT_IMAGES, GT_ANNS, GT_BY_IMAGE)
eop.split_image_count = lambda _cfg, _split: IMAGE_COUNT

Q25_THRESHOLDS_PATH = EXP / "eval/size_thresholds_train.json"
Q25_THRESHOLDS = json.loads(Q25_THRESHOLDS_PATH.read_text())
_q25_sets = ess.build_q25_gt_sets(cfg, "test", Q25_THRESHOLDS)
ess.build_q25_gt_sets = lambda _cfg, _split, _thr: _q25_sets
log(f"Q25 test target counts per category: {dict(_q25_sets[2])}")

# base.operating_curve / base.op_p90 / base.matched_fp are the frozen val-side
# selectors; re-point their split-bound loaders at the test split so the exact
# frozen bytecode produces the TEST-side diagnostic curve.
base.load_yolo_gt = lambda _cfg, _split, category_base=1: (GT_IMAGES, GT_ANNS, GT_BY_IMAGE)
base.split_images = lambda _cfg, _split: [p for p in split_images(cfg, "test") if p.exists()]

REFERENCE_FP_PER_IMAGE = None


def freeze_record(arm: str, seed: str) -> dict:
    """Frozen validation freeze record for this arm's PRIMARY (last.pt) checkpoint."""
    entry = plan["models"][f"{arm}@{seed}"]
    ckpt = Path(entry["primary"]["checkpoint"])
    run = ckpt.parent.parent.name
    sel_dir = EXP / "eval/phase4_selection" / run
    sc = json.loads((sel_dir / "selected_checkpoint.json").read_text())
    cands = sc.get("candidates", [])
    last = [c for c in cands if c.get("label") == "last"]
    if not last:
        raise SystemExit(f"no 'last' candidate in {sel_dir}")
    rec = last[0]
    return {
        "run": run,
        "sel_dir": sel_dir,
        "checkpoint": str(ckpt),
        "checkpoint_sha256_frozen_in_plan": entry["primary"]["sha256_frozen"],
        "checkpoint_sha256_in_freeze": rec.get("checkpoint_sha256"),
        "sha_match": rec.get("checkpoint_sha256") == entry["primary"]["sha256_frozen"],
        "matched_fp_threshold_val": float(rec["matched_fp_threshold"]),
        "op_p90_threshold_val": float(rec["op_p90"]["threshold"]),
        "val_matched_fp_recall": rec.get("matched_fp_recall"),
        "val_op_p90_fp_per_image": rec["op_p90"].get("fp_per_image"),
        "val_map75": rec.get("unified_val_map75"),
        "val_q25_macro_recall_iou50": rec.get("q25_macro_recall_iou50"),
        "val_mean_matched_iou": rec.get("mean_matched_iou"),
        "val_taxonomy_background_fp": (rec.get("error_taxonomy_matched_fp") or {}).get("counts", {}).get("background_fp"),
        "reference_fp_per_image": sc.get("selection_rule", {}).get("reference_fp_per_image"),
    }


rows: list[dict] = []
missing: list[dict] = []

for arm, seed in ARMS:
    name = f"{arm}@{seed}"
    tag = f"{name.replace('@', '_at_')}__primary"
    d = RES / tag
    pred = d / "v/predictions.json"
    mfile = d / "unified/metrics.json"
    if not pred.exists() or not mfile.exists():
        missing.append({"arm": arm, "seed": seed, "reason": f"missing frozen input under {d}"})
        log(f"MISSING {name}")
        continue

    fr = freeze_record(arm, seed)
    if REFERENCE_FP_PER_IMAGE is None:
        REFERENCE_FP_PER_IMAGE = fr["reference_fp_per_image"]

    stored = json.loads(mfile.read_text())
    stored_metrics = stored.get("metrics", {})

    preds = normalize_prediction_rows(pred, cat_set, PRED_BASE, SCORE_FLOOR)

    # --- localization: mean_matched_iou (frozen definition, phase4c unified_for_pred)
    m50 = greedy_match(GT_BY_IMAGE, preds, 0.5, True)
    mean_iou = (sum(float(m["iou"]) for m in m50["matches"]) / len(m50["matches"])) if m50["matches"] else None

    # --- small objects: q25 macro recall @ IoU0.5 (frozen eval_size_stratified.eval_q25)
    q25 = ess.eval_q25(cfg, "test", pred, Q25_THRESHOLDS, PRED_BASE)
    q25_macro_r50 = base.macro(q25, "Q25_Recall@0.5")

    # --- recall + FP: frozen VAL thresholds applied to TEST (frozen test discipline)
    at_mfp = eop.evaluate_at_threshold(cfg, "test", pred, fr["matched_fp_threshold_val"], PRED_BASE)
    at_p90 = eop.evaluate_at_threshold(cfg, "test", pred, fr["op_p90_threshold_val"], PRED_BASE)

    # --- false positives: error taxonomy at the matched-FP threshold
    tax = classify_errors(
        GT_BY_IMAGE,
        normalize_prediction_rows(pred, cat_set, PRED_BASE, fr["matched_fp_threshold_val"]),
        FINAL_NAMES,
        0.0,
    )
    tax_counts = tax["counts"]
    bg = int(tax_counts.get("background_fp", 0))

    # --- diagnostics: test-side reselected operating points (NOT the frozen metric)
    ref_fp = float(fr["reference_fp_per_image"])
    t_p90 = base.op_p90(cfg, pred, PRED_BASE)
    t_mfp = base.matched_fp(cfg, pred, ref_fp, PRED_BASE)

    row = {
        "arm": arm,
        "seed": seed,
        "lock_result_dir": str(d),
        "checkpoint": fr["checkpoint"],
        "checkpoint_sha_matches_lock_plan": fr["sha_match"],
        "n_frozen_predictions": len(preds),
        # four-part mapped metrics
        "matched_fp_recall": at_mfp["recall"],
        "matched_fp_threshold_from_val": fr["matched_fp_threshold_val"],
        "reference_fp_per_image_C0Std": ref_fp,
        "matched_fp_test_fp_per_image": at_mfp["fp_per_image"],
        "q25_macro_recall_iou50": q25_macro_r50,
        "map75": stored_metrics.get("mAP75"),
        "mean_matched_iou": mean_iou,
        "taxonomy_background_or_spurious_count": bg,
        "taxonomy_background_or_spurious_per_image": bg / IMAGE_COUNT,
        "op_p90_fp_per_image": at_p90["fp_per_image"],
        "op_p90_threshold_from_val": fr["op_p90_threshold_val"],
        # context / provenance
        "map50_95_stored": stored_metrics.get("mAP50-95"),
        "map50_stored": stored_metrics.get("mAP50"),
        "op_p90_test_recall": at_p90["recall"],
        "op_p90_test_precision": at_p90["precision"],
        "taxonomy_true_positive": int(tax_counts.get("TP", 0)),
        "taxonomy_duplicate": int(tax_counts.get("duplicate", 0)),
        "taxonomy_class_confusion": int(tax_counts.get("class_confusion", 0)),
        "taxonomy_localization_error": int(tax_counts.get("localization_error", 0)),
        "taxonomy_partial_wrong_overlap": int(tax_counts.get("partial_wrong_overlap", 0)),
        # ALTERNATIVE literal reading of "background_or_spurious" (NOT the frozen metric):
        "alt_background_plus_partial_count": bg + int(tax_counts.get("partial_wrong_overlap", 0)),
        "alt_background_plus_partial_per_image": (bg + int(tax_counts.get("partial_wrong_overlap", 0))) / IMAGE_COUNT,
        "taxonomy_FN": int(tax_counts.get("FN", 0)),
        # diagnostics (explicitly NOT the frozen mapped definition)
        "diag_test_reselected_matched_fp_threshold": t_mfp["threshold"],
        "diag_test_reselected_matched_fp_recall": t_mfp["recall"],
        "diag_test_reselected_matched_fp_per_image": t_mfp["fp_per_image"],
        "diag_test_reselected_op_p90_threshold": t_p90["threshold"],
        "diag_test_reselected_op_p90_fp_per_image": t_p90["fp_per_image"],
        "diag_test_reselected_op_p90_reached": t_p90.get("p90_reached"),
        # validation-side frozen values, for reference only
        "val_matched_fp_recall": fr["val_matched_fp_recall"],
        "val_q25_macro_recall_iou50": fr["val_q25_macro_recall_iou50"],
        "val_map75": fr["val_map75"],
        "val_mean_matched_iou": fr["val_mean_matched_iou"],
        "val_op_p90_fp_per_image": fr["val_op_p90_fp_per_image"],
    }
    rows.append(row)
    log(f"{name:16s} mfpR={row['matched_fp_recall']:.5f} q25={q25_macro_r50:.5f} "
        f"map75={row['map75']:.5f} mIoU={mean_iou:.5f} bg={bg} p90fp={row['op_p90_fp_per_image']:.5f}")

# ---------------------------------------------------------------- sanity anchor
log("sanity anchor: recomputing COCOeval mAP50-95 for C4-M@42 primary")
anchor_dir = RES / "C4-M_at_42__primary/unified"
anchor = evaluate_coco(anchor_dir / "test_gt_coco.json", anchor_dir / "predictions_coco.json", cat_ids)
anchor_map = anchor.get("mAP50-95")
anchor_ok = anchor_map is not None and abs(round(float(anchor_map), 5) - 0.40439) < 1e-9
log(f"anchor mAP50-95 = {anchor_map} -> round5 {round(float(anchor_map),5)} ok={anchor_ok} method={anchor.get('method')}")

# ---------------------------------------------------------------- CSV
csv_path = OUT / "mapped_metrics_per_arm.csv"
fields = list(rows[0].keys())
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sorted(rows, key=lambda r: (ARMS.index((r["arm"], r["seed"])))):
        w.writerow(r)
log(f"wrote {csv_path}")

# ---------------------------------------------------------------- contrasts
index = {(r["arm"], r["seed"]): r for r in rows}


def delta(a_row, b_row, metric):
    av, bv = a_row.get(metric), b_row.get(metric)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


contrasts_out = {
    "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "what_this_is": (
        "Post-hoc re-scoring of the frozen, already-consumed Phase-4 locked-test predictions "
        "(consumption event 2026-07-20). No inference was run; the lock was not touched. "
        "Deltas are treatment_arm minus reference_arm on the PRIMARY (protocol-complete last.pt) estimand."
    ),
    "four_part_mapping": FOUR_PART,
    "background_or_spurious_resolution": (
        "'background_or_spurious' is not a literal key in the project's taxonomy. Resolved as "
        "counts['background_fp'] following tools/ccad/phase4d_synthesis.py line 106, which "
        "selects taxonomy keys containing 'background'. The alternative literal reading "
        "(background_fp + partial_wrong_overlap) is reported per contrast under "
        "'alternative_reading_background_plus_partial_wrong_overlap' and DISAGREES IN SIGN "
        "for H2 and H6a."
    ),
    "metric_direction": {
        "matched_fp_recall": "higher is better",
        "q25_macro_recall_iou50": "higher is better",
        "map75": "higher is better",
        "mean_matched_iou": "higher is better",
        "taxonomy_background_or_spurious_count": "lower is better",
        "taxonomy_background_or_spurious_per_image": "lower is better",
        "op_p90_fp_per_image": "lower is better",
    },
    "reference_fp_per_image_C0Std": REFERENCE_FP_PER_IMAGE,
    "sanity_anchor": {
        "arm": "C4-M@42 primary",
        "recomputed_map50_95": anchor_map,
        "expected_round5": 0.40439,
        "match": bool(anchor_ok),
        "evaluator": anchor.get("method"),
    },
    "contrasts": {},
}

for hid, a_arm, b_arm, seeds in CONTRASTS:
    block = {
        "contrast": f"{a_arm} minus {b_arm}",
        "seeds": seeds,
        "per_seed": {},
        "mean_delta": {},
        "arm_values": {"treatment": {}, "reference": {}},
        "missing": [],
    }
    for s in seeds:
        ar, br = index.get((a_arm, s)), index.get((b_arm, s))
        if ar is None or br is None:
            block["missing"].append({"seed": s, "treatment_present": ar is not None,
                                     "reference_present": br is not None})
            continue
        block["per_seed"][s] = {m: delta(ar, br, m) for m in MAPPED_METRICS}
        block["arm_values"]["treatment"][s] = {m: ar.get(m) for m in MAPPED_METRICS}
        block["arm_values"]["reference"][s] = {m: br.get(m) for m in MAPPED_METRICS}
    for m in MAPPED_METRICS:
        vals = [v[m] for v in block["per_seed"].values() if v.get(m) is not None]
        block["mean_delta"][m] = (sum(vals) / len(vals)) if vals else None
    alt_per_seed = {}
    for s_ in seeds:
        ar, br = index.get((a_arm, s_)), index.get((b_arm, s_))
        if ar is None or br is None:
            continue
        alt_per_seed[s_] = {m: delta(ar, br, m) for m in ALT_METRICS}
    block["alternative_reading_background_plus_partial_wrong_overlap"] = {
        "note": ("NOT the frozen metric. Literal-English alternative that folds "
                 "partial_wrong_overlap into 'spurious'. Shown because it flips the sign "
                 "of this outcome for H2 and H6a."),
        "per_seed": alt_per_seed,
        "mean_delta": {
            m: (sum(v[m] for v in alt_per_seed.values()) / len(alt_per_seed)) if alt_per_seed else None
            for m in ALT_METRICS
        },
    }
    block["four_part_summary"] = {
        part: {m: block["mean_delta"][m] for m in ms} for part, ms in FOUR_PART.items()
    }
    contrasts_out["contrasts"][hid] = block

if missing:
    contrasts_out["missing_arms"] = missing

json_path = OUT / "mapped_metrics_contrasts.json"
json_path.write_text(json.dumps(contrasts_out, indent=1), encoding="utf-8")
log(f"wrote {json_path}")

# machine-readable side-car for the README writer
(OUT / "_per_arm_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
(OUT / "_run_meta.json").write_text(json.dumps({
    "image_count_test": IMAGE_COUNT,
    "gt_annotations_test": len(GT_ANNS),
    "reference_fp_per_image_C0Std": REFERENCE_FP_PER_IMAGE,
    "q25_thresholds_path": str(Q25_THRESHOLDS_PATH),
    "q25_thresholds_sha256": sha256_file(Q25_THRESHOLDS_PATH),
    "q25_thresholds": Q25_THRESHOLDS,
    "q25_test_target_counts": {str(k): v for k, v in _q25_sets[2].items()},
    "anchor": contrasts_out["sanity_anchor"],
    "missing": missing,
}, indent=1), encoding="utf-8")
log("DONE")
