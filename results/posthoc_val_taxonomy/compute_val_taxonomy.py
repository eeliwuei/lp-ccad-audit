#!/usr/bin/env python3
"""Post-hoc VALIDATION-side error taxonomy for the 18 Phase-4 mainline arms.

Mirrors experiments/ccad_yolo26n/eval/phase4_test_lock_posthoc_mapped/compute_mapped_metrics.py
but on the VALIDATION split, so the manuscript can report the taxonomy false-positive
metric on both splits.

No inference. Only frozen validation predictions
(phase4_selection/<RUN>/candidates/last/predictions.json), frozen val GT, and the arm's own
frozen validation matched-FP threshold are read. Nothing frozen is written.
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

from ccad_core import configured_final_classes, exp_dir, load_config          # noqa: E402
from phase2b_common import (                                                  # noqa: E402
    coco_category_ids,
    load_yolo_gt,
    normalize_prediction_rows,
    split_images,
)
from analyze_error_taxonomy import classify as classify_errors                # noqa: E402

CFG = ROOT / "experiments/ccad_yolo26n/configs/ccad_project.yaml"
cfg = load_config(str(CFG))
EXP = exp_dir(cfg)
PLAN = EXP / "eval/phase4_test_lock/test_lock_plan_20260720_0218.json"
OUT = EXP / "eval/phase4_val_posthoc_taxonomy"
OUT.mkdir(parents=True, exist_ok=True)

FINAL_NAMES = {0: "knife", 1: "gun", 2: "stick"}
PRED_BASE = "auto"

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

DELTA_METRICS = [
    "taxonomy_background_or_spurious_count",
    "taxonomy_background_or_spurious_per_image",
    "alt_background_plus_partial_count",
    "alt_background_plus_partial_per_image",
]

TAX_KEYS = ["TP", "duplicate", "class_confusion", "localization_error",
            "partial_wrong_overlap", "background_fp", "FN"]


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ frozen inputs
plan = json.loads(PLAN.read_text())
cat_ids = coco_category_ids(cfg)
cat_set = set(cat_ids)
cfg_final_classes = configured_final_classes(cfg)

log("loading frozen VALIDATION GT (labels only, no inference)")
GT_IMAGES, GT_ANNS, GT_BY_IMAGE = load_yolo_gt(cfg, "val", category_base=1)
VAL_IMAGES = split_images(cfg, "val")
IMAGE_COUNT = len([p for p in VAL_IMAGES if p.exists()])
log(f"val GT: {len(GT_IMAGES)} images ({IMAGE_COUNT} present of {len(VAL_IMAGES)} listed), "
    f"{len(GT_ANNS)} annotations, categories={cat_ids}, final_classes={cfg_final_classes}")

# ------------------------------------------------------------------ C0 sanity anchor
log("SANITY: recomputing the two stored C0 baseline validation taxonomies")
sanity = {}
for model in ["C0_yolo26n_human_only_960", "C0s_yolo26s_human_only_960"]:
    stored_path = EXP / "eval/error_taxonomy" / f"{model}_val.json"
    pred_path = EXP / "eval/image_eval" / f"{model}_val" / "predictions.json"
    if not stored_path.exists() or not pred_path.exists():
        sanity[model] = {"status": "inputs_missing",
                         "stored_exists": stored_path.exists(),
                         "predictions_exists": pred_path.exists()}
        continue
    stored = json.loads(stored_path.read_text())
    conf = float(stored.get("confidence_threshold", 0.25))
    preds = normalize_prediction_rows(pred_path, cat_set, PRED_BASE, conf)
    got = classify_errors(GT_BY_IMAGE, preds, FINAL_NAMES, conf)
    counts_match = dict(got["counts"]) == dict(stored["counts"])
    fn_match = int(got["fn"]) == int(stored["fn"])
    conf_match = got["confusion_edges"] == stored["confusion_edges"]
    loc_match = got["localization_errors_by_class"] == stored["localization_errors_by_class"]
    sanity[model] = {
        "status": "ok" if (counts_match and fn_match and conf_match and loc_match) else "MISMATCH",
        "stored_file": str(stored_path),
        "predictions_file": str(pred_path),
        "confidence_threshold": conf,
        "stored_counts": stored["counts"],
        "recomputed_counts": dict(got["counts"]),
        "counts_match": counts_match,
        "fn_match": fn_match,
        "confusion_edges_match": conf_match,
        "localization_by_class_match": loc_match,
        "stored_background_fp": stored["counts"].get("background_fp"),
        "recomputed_background_fp": got["counts"].get("background_fp"),
    }
    log(f"  {model}: {sanity[model]['status']} "
        f"stored bg={stored['counts'].get('background_fp')} got bg={got['counts'].get('background_fp')} "
        f"(all-keys match={counts_match})")

SANITY_OK = all(v.get("status") == "ok" for v in sanity.values())
if not SANITY_OK:
    log("SANITY FAILED -- writing sanity report only and aborting before producing arm numbers")
    (OUT / "_SANITY_FAILURE.json").write_text(json.dumps(sanity, indent=1), encoding="utf-8")
    raise SystemExit(2)

# ------------------------------------------------------------------ per-arm
rows: list[dict] = []
missing: list[dict] = []

for arm, seed in ARMS:
    name = f"{arm}@{seed}"
    entry = plan["models"].get(name)
    if entry is None:
        missing.append({"arm": arm, "seed": seed, "reason": "arm absent from test lock plan"})
        log(f"MISSING {name}: not in lock plan")
        continue
    ckpt = Path(entry["primary"]["checkpoint"])
    run = ckpt.parent.parent.name
    sel_dir = EXP / "eval/phase4_selection" / run
    sc_path = sel_dir / "selected_checkpoint.json"
    if not sc_path.exists():
        missing.append({"arm": arm, "seed": seed, "reason": f"missing {sc_path}"})
        log(f"MISSING {name}: no selected_checkpoint.json")
        continue
    sc = json.loads(sc_path.read_text())
    last = [c for c in sc.get("candidates", []) if c.get("label") == "last"]
    if not last:
        missing.append({"arm": arm, "seed": seed, "reason": f"no candidates[label=='last'] in {sc_path}"})
        log(f"MISSING {name}: no 'last' candidate")
        continue
    rec = last[0]

    pred_path = Path(rec["predictions"])
    canonical = sel_dir / "candidates/last/predictions.json"
    if not pred_path.exists():
        missing.append({"arm": arm, "seed": seed,
                        "reason": f"frozen val predictions absent: {pred_path}"})
        log(f"MISSING {name}: predictions absent {pred_path}")
        continue

    thr = float(rec["matched_fp_threshold"])
    preds = normalize_prediction_rows(pred_path, cat_set, PRED_BASE, thr)
    tax = classify_errors(GT_BY_IMAGE, preds, FINAL_NAMES, 0.0)
    counts = tax["counts"]
    bg = int(counts.get("background_fp", 0))
    pwo = int(counts.get("partial_wrong_overlap", 0))

    frozen_tax = (rec.get("error_taxonomy_matched_fp") or {})
    frozen_counts = frozen_tax.get("counts") or {}
    reproduces_freeze = bool(frozen_counts) and dict(frozen_counts) == dict(counts)

    row = {
        "arm": arm,
        "seed": seed,
        "run": run,
        "selection_dir": str(sel_dir),
        "val_predictions_file": str(pred_path),
        "val_predictions_is_candidates_last": pred_path.resolve() == canonical.resolve(),
        "val_predictions_sha256": sha256_file(pred_path),
        "checkpoint": str(ckpt),
        "checkpoint_sha256_in_freeze": rec.get("checkpoint_sha256"),
        "checkpoint_sha_matches_lock_plan": rec.get("checkpoint_sha256") == entry["primary"]["sha256_frozen"],
        "matched_fp_threshold_from_val": thr,
        "reference_fp_per_image_C0Std": sc.get("selection_rule", {}).get("reference_fp_per_image"),
        "n_predictions_at_threshold": len(preds),
        "val_image_count": IMAGE_COUNT,
        # the mapped false-positive metric, validation side
        "taxonomy_background_or_spurious_count": bg,
        "taxonomy_background_or_spurious_per_image": bg / IMAGE_COUNT,
        # alternative literal reading
        "taxonomy_partial_wrong_overlap": pwo,
        "alt_background_plus_partial_count": bg + pwo,
        "alt_background_plus_partial_per_image": (bg + pwo) / IMAGE_COUNT,
        # full breakdown
        "taxonomy_true_positive": int(counts.get("TP", 0)),
        "taxonomy_duplicate": int(counts.get("duplicate", 0)),
        "taxonomy_class_confusion": int(counts.get("class_confusion", 0)),
        "taxonomy_localization_error": int(counts.get("localization_error", 0)),
        "taxonomy_FN": int(counts.get("FN", 0)),
        # cross-check against the value the ORIGINAL freeze stored inside selected_checkpoint.json
        "freeze_record_background_fp": frozen_counts.get("background_fp"),
        "reproduces_frozen_freeze_record": reproduces_freeze,
        # validation-side context already present in the freeze
        "val_matched_fp_recall": rec.get("matched_fp_recall"),
        "val_op_p90_fp_per_image": (rec.get("op_p90") or {}).get("fp_per_image"),
        "val_map75": rec.get("unified_val_map75"),
        "val_q25_macro_recall_iou50": rec.get("q25_macro_recall_iou50"),
        "val_mean_matched_iou": rec.get("mean_matched_iou"),
    }
    row["_confusion_edges"] = tax["confusion_edges"]
    row["_localization_errors_by_class"] = tax["localization_errors_by_class"]
    rows.append(row)
    log(f"{name:16s} thr={thr:<9} bg={bg:3d} pwo={pwo:2d} alt={bg+pwo:3d} "
        f"bg/img={bg/IMAGE_COUNT:.6f} reproduces_freeze={reproduces_freeze}")

REPRO_ALL = all(r["reproduces_frozen_freeze_record"] for r in rows)
log(f"freeze-record reproduction across arms: {sum(r['reproduces_frozen_freeze_record'] for r in rows)}/{len(rows)}")

# ------------------------------------------------------------------ CSV
csv_fields = [k for k in rows[0].keys() if not k.startswith("_")]
csv_path = OUT / "val_taxonomy_per_arm.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: ARMS.index((r["arm"], r["seed"]))):
        w.writerow(r)
log(f"wrote {csv_path}")

# ------------------------------------------------------------------ contrasts
index = {(r["arm"], r["seed"]): r for r in rows}


def delta(a, b, m):
    av, bv = a.get(m), b.get(m)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


out = {
    "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "what_this_is": (
        "Post-hoc VALIDATION-side error-taxonomy false-positive metric for the 18 Phase-4 "
        "mainline arms, computed 2026-07-26 from the frozen validation predictions. The original "
        "validation pass wrote this quantity only inside each arm's selected_checkpoint.json "
        "freeze record and produced no standalone per-arm artefact; eval/error_taxonomy/ holds "
        "only the two C0 baselines. No inference was run. Deltas are treatment minus reference "
        "on the PRIMARY (protocol-complete last.pt) estimand."
    ),
    "definition": (
        "tools/ccad/analyze_error_taxonomy.py::classify(val_gt_by_image, "
        "normalize_prediction_rows(<arm val predictions>, {1,2,3}, 'auto', matched_fp_threshold), "
        "{0:'knife',1:'gun',2:'stick'}, 0.0) -> counts['background_fp']; per-image divides by "
        f"{IMAGE_COUNT} validation images. Identical to the test-side computation in "
        "eval/phase4_test_lock_posthoc_mapped/, so the two splits are comparable."
    ),
    "threshold_provenance": (
        "each arm's own frozen validation matched-FP threshold, "
        "phase4_selection/<RUN>/selected_checkpoint.json -> candidates[label=='last'].matched_fp_threshold"
    ),
    "background_or_spurious_resolution": (
        "'background_or_spurious' is not a literal key in the project taxonomy. Resolved as "
        "counts['background_fp'] following tools/ccad/phase4d_synthesis.py line 106. The "
        "alternative literal reading (background_fp + partial_wrong_overlap) is reported "
        "alongside for every contrast."
    ),
    "metric_direction": {m: "lower is better" for m in DELTA_METRICS},
    "val_image_count": IMAGE_COUNT,
    "val_annotation_count": len(GT_ANNS),
    "sanity_check_c0_baselines": sanity,
    "reproduces_frozen_freeze_records": REPRO_ALL,
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
        block["per_seed"][s] = {m: delta(ar, br, m) for m in DELTA_METRICS}
        block["arm_values"]["treatment"][s] = {m: ar.get(m) for m in DELTA_METRICS}
        block["arm_values"]["reference"][s] = {m: br.get(m) for m in DELTA_METRICS}
    for m in DELTA_METRICS:
        vals = [v[m] for v in block["per_seed"].values() if v.get(m) is not None]
        block["mean_delta"][m] = (sum(vals) / len(vals)) if vals else None
    block["reported_reading"] = {
        "metric": "taxonomy_background_or_spurious (= background_fp)",
        "mean_delta_count": block["mean_delta"]["taxonomy_background_or_spurious_count"],
        "mean_delta_per_image": block["mean_delta"]["taxonomy_background_or_spurious_per_image"],
    }
    block["alternative_reading_background_plus_partial_wrong_overlap"] = {
        "note": ("NOT the frozen metric. Literal-English alternative that folds "
                 "partial_wrong_overlap into 'spurious'."),
        "mean_delta_count": block["mean_delta"]["alt_background_plus_partial_count"],
        "mean_delta_per_image": block["mean_delta"]["alt_background_plus_partial_per_image"],
        "per_seed": {s: {"alt_background_plus_partial_count": v["alt_background_plus_partial_count"],
                         "alt_background_plus_partial_per_image": v["alt_background_plus_partial_per_image"]}
                     for s, v in block["per_seed"].items()},
    }
    block["sign_agreement_between_readings"] = None
    a_ = block["mean_delta"]["taxonomy_background_or_spurious_count"]
    b_ = block["mean_delta"]["alt_background_plus_partial_count"]
    if a_ is not None and b_ is not None:
        def sgn(x):
            return 0 if x == 0 else (1 if x > 0 else -1)
        block["sign_agreement_between_readings"] = (sgn(a_) == sgn(b_))
    out["contrasts"][hid] = block

if missing:
    out["missing_arms"] = missing

json_path = OUT / "val_taxonomy_contrasts.json"
json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
log(f"wrote {json_path}")

(OUT / "_per_arm_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
(OUT / "_run_meta.json").write_text(json.dumps({
    "val_image_count": IMAGE_COUNT,
    "val_images_listed": len(VAL_IMAGES),
    "val_annotations": len(GT_ANNS),
    "categories": cat_ids,
    "final_classes": cfg_final_classes,
    "taxonomy_script_sha256": sha256_file(ROOT / "tools/ccad/analyze_error_taxonomy.py"),
    "freeze_script_sha256": sha256_file(ROOT / "tools/ccad/phase4_freeze_model_full.py"),
    "phase2b_common_sha256": sha256_file(ROOT / "tools/ccad/phase2b_common.py"),
    "config_sha256": sha256_file(CFG),
    "lock_plan_sha256": sha256_file(PLAN),
    "sanity": sanity,
    "reproduces_frozen_freeze_records": REPRO_ALL,
    "missing": missing,
}, indent=1), encoding="utf-8")
log("DONE")
