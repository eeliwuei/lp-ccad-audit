#!/usr/bin/env python3
"""Freeze-eval one v3 factorial arm on the PRIMARY estimand (last.pt, val,
COCOeval) using the identical unified pipeline the nine-arm study used, and
emit overall + per-class AP.

Usage: freeze_eval.py <run_name> [--force]
Writes: eval/factorial_v3_freeze/<run_name>/metrics_unified.json
Prints one JSON line suitable for assembling results12.json.

-----------------------------------------------------------------------------
RELEASE NOTE -- PATHS MUST BE ADAPTED, THIS SCRIPT IS NOT SELF-CONTAINED.
This file is released verbatim (modulo path sanitization) as the evaluation
entry point that produced results/factorial_runs.csv. It is documentation of
the frozen procedure, not a runnable demo: it needs
  * the training-project root, supplied through the environment variable
    LPCCAD_PROJECT_ROOT (originally an absolute path on the training server);
  * the project-internal helper modules `ccad_core`, `phase2b_common` and
    `phase4c_select_and_freeze_c0r`, which live in that project's
    `tools/ccad/` directory and are NOT part of this minimal release;
  * the restricted weapon-detection dataset and the trained checkpoints
    (see docs/RESTRICTED_DATA_ACCESS.md);
  * `ultralytics` (AGPL-3.0), which users must install themselves
    (see docs/THIRD_PARTY_NOTICES.md).
Re-running it therefore requires the data-use agreement and a GPU; the
released CSVs in results/ already contain every number it emitted.
-----------------------------------------------------------------------------
"""
import argparse, json, os, sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Original: a fixed absolute path on the training server. Sanitized to an
# environment variable so no host-specific path ships in the release.
ROOT = Path(os.environ.get("LPCCAD_PROJECT_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "tools/ccad"))
import hashlib                                                  # noqa: E402
from ccad_core import load_config, exp_dir, write_json          # noqa: E402
from phase2b_common import write_split_gt_coco                  # noqa: E402
import phase4c_select_and_freeze_c0r as base                    # noqa: E402

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

CFG = str(ROOT / "experiments/ccad_yolo26n/configs/ccad_project.yaml")

# run-name -> factorial cell (mono/shuf x single/mixed) and training seed
def parse_run(name):
    if name.startswith("F-MonoSingle"):
        cell = "mono_single"
    elif name.startswith("F-MonoMix"):
        cell = "mono_mixed"
    elif name.startswith("F-ShufSingle"):
        cell = "shuf_single"
    elif name.startswith("F-ShufMix"):
        cell = "shuf_mixed"
    else:
        raise SystemExit(f"cannot map run name to a factorial cell: {name}")
    seed = None
    for tok in name.split("_"):
        if tok.startswith("seed"):
            seed = int(tok[4:])
    if seed is None:
        raise SystemExit(f"no seed in run name: {name}")
    return cell, seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cell, seed = parse_run(a.run_name)
    cfg = load_config(CFG)
    exp = exp_dir(cfg)
    ckpt = exp / "models/students" / a.run_name / "weights/last.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing primary checkpoint (run not finished?): {ckpt}")

    out = exp / "eval/factorial_v3_freeze" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    gt_path = out / "val_gt_coco.json"
    gt_summary = write_split_gt_coco(cfg, "val", gt_path, category_base=1)

    pred_path = out / "predictions.json"
    if a.force or not pred_path.exists():
        data_yaml = Path(cfg["data"]["source_dataset_root"]) / "dataset.yaml"
        with (out / "yolo_val.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            from ultralytics import YOLO
            YOLO(str(ckpt)).val(data=str(data_yaml), split="val", imgsz=a.imgsz,
                                project=str(out.parent), name=out.name, exist_ok=True,
                                plots=False, save_json=True, device=a.device, verbose=False)
    if not pred_path.exists():
        raise SystemExit(f"YOLO.val produced no predictions: {pred_path}")

    unified = base.unified_for_pred(cfg, pred_path, gt_path, gt_summary, "auto")
    coco = unified["coco"]
    # per_class rows carry category_id (1-based); dataset.yaml names are 0:knife 1:gun 2:stick
    CATNAME = {1: "knife", 2: "gun", 3: "stick"}
    per_class = {CATNAME.get(int(r["category_id"]), str(r["category_id"])): r.get("AP50-95")
                 for r in (coco.get("per_class") or [])}
    per_class_ap50 = {CATNAME.get(int(r["category_id"]), str(r["category_id"])): r.get("AP50")
                      for r in (coco.get("per_class") or [])}
    rec = {
        "run": a.run_name, "cell": cell, "seed": seed,
        "mAP": coco.get("mAP50-95"), "mAP50": coco.get("mAP50"), "mAP75": coco.get("mAP75"),
        "per_class": per_class, "per_class_ap50": per_class_ap50,
        "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
    }
    write_json(out / "metrics_unified.json", rec)
    print(json.dumps(rec))

if __name__ == "__main__":
    main()
