#!/usr/bin/env python3
"""Source-clustered PAIRED bootstrap for the locked-test contrasts.

Cluster = source id (filename with trailing _<global idx>, _f<frame>, and
Roboflow .rf.<hex> tags stripped -- the same rule as the split audit).
Each replicate resamples clusters with replacement (n_clusters draws), the SAME
resample is applied to every arm (paired), and each arm's mAP50-95 is
recomputed with pycocotools on the resampled image set. B=1000, seed 20260725,
percentile 95% CIs. Read-only over the consumed locked-test artifacts.

RELEASE NOTE. The locked-test GT and per-arm prediction files this script
consumes are restricted (see docs/RESTRICTED_DATA_ACCESS.md) and are not part
of this release; results/bootstrap_summary.csv is the output it produced.
Point DATA_ROOT at a directory holding `<arm>_at_<seed>__primary/unified/`
sub-directories to re-run it. The re-numbering of image and annotation ids in
`one_rep` is the load-bearing detail -- tests/test_bootstrap_duplicate_ids.py
pins it.
"""
import json, re, os, sys, random
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

# Original: absolute paths on the training server. Sanitized to DATA_ROOT.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "eval/phase4_test_lock"))
ROOT = DATA_ROOT / "results"
OUT = DATA_ROOT / "source_boot"
OUT.mkdir(parents=True, exist_ok=True)
B = 1000
SEED = 20260725
CAT_IDS = [1, 2, 3]

ARMS = {  # arm-> {seed: dirname}
    "C0-R": {42: "C0-R_at_42__primary", 1337: "C0-R_at_1337__primary", 20260703: "C0-R_at_20260703__primary"},
    "C1-M": {42: "C1-M_at_42__primary", 1337: "C1-M_at_1337__primary", 20260703: "C1-M_at_20260703__primary"},
    "C4-M": {42: "C4-M_at_42__primary", 1337: "C4-M_at_1337__primary", 20260703: "C4-M_at_20260703__primary"},
    "C4MixFT-M": {42: "C4MixFT-M_at_42__primary", 1337: "C4MixFT-M_at_1337__primary"},
    "C4Mix-M": {42: "C4Mix-M_at_42__primary", 1337: "C4Mix-M_at_1337__primary"},
    "C4R-M": {42: "C4R-M_at_42__primary", 1337: "C4R-M_at_1337__primary"},
}

def source_id(name):
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"_\d{4,}$", "", s)
    s = re.sub(r"_f\d+$", "", s)
    s = re.sub(r"(_jpg)?\.rf\.[0-9a-f]+$", "", s)
    s = re.sub(r"_f\d+$", "", s)
    return s

# ---- load shared GT once ----
GT = json.load(open(ROOT / "C4-M_at_42__primary/unified/test_gt_coco.json"))
IMGS = GT["images"]
ANN_BY_IMG = defaultdict(list)
for a in GT["annotations"]:
    ANN_BY_IMG[a["image_id"]].append(a)
CLUSTERS = defaultdict(list)                      # source -> [image dicts]
for im in IMGS:
    CLUSTERS[source_id(im["file_name"])].append(im)
CLUST_KEYS = sorted(CLUSTERS.keys())
NC = len(CLUST_KEYS)

# ---- load every arm's predictions once ----
PRED_BY = {}                                      # (arm,seed) -> {img_id: [pred rows]}
for arm, seeds in ARMS.items():
    for sd, d in seeds.items():
        preds = json.load(open(ROOT / d / "unified/predictions_coco.json"))
        by = defaultdict(list)
        for p in preds:
            by[p["image_id"]].append(p)
        PRED_BY[(arm, sd)] = by

def eval_map(gt_dict, pred_list):
    # faster-coco-eval: the evaluator this project's frozen metrics used
    # (the venv's pycocotools has a numpy ABI mismatch).
    from contextlib import redirect_stdout
    from io import StringIO
    from faster_coco_eval import COCO, COCOeval_faster
    if not pred_list:
        return 0.0
    with redirect_stdout(StringIO()):
        cg = COCO(gt_dict)
        cd = cg.loadRes(pred_list)
        ev = COCOeval_faster(cg, cd, "bbox")
        ev.params.catIds = CAT_IDS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])

def one_rep(b):
    rng = random.Random(SEED + b)
    draw = [CLUST_KEYS[rng.randrange(NC)] for _ in range(NC)]
    # build resampled gt (unique new ids for duplicated images/annotations)
    new_imgs, ann_rows = [], []
    remap = []                                    # (new_img_id, old_img_id)
    nid = 1; aid = 1
    for inst, key in enumerate(draw):
        for im in CLUSTERS[key]:
            new_imgs.append({**im, "id": nid})
            for a in ANN_BY_IMG[im["id"]]:
                ann_rows.append({**a, "id": aid, "image_id": nid}); aid += 1
            remap.append((nid, im["id"]))
            nid += 1
    gt_dict = {"info": GT.get("info", {}), "licenses": GT.get("licenses", []),
               "images": new_imgs, "annotations": ann_rows, "categories": GT["categories"]}
    out = {}
    for key2, by in PRED_BY.items():
        plist = []
        for new_id, old_id in remap:
            for p in by.get(old_id, []):
                plist.append({**p, "image_id": new_id})
        out["%s@%s" % key2] = eval_map(gt_dict, plist)
    return b, out

def main():
    print(f"clusters={NC}, images={len(IMGS)}, arms={len(PRED_BY)}, B={B}", flush=True)
    res_file = OUT / "boot_raw.jsonl"
    done = set()
    if res_file.exists():
        for line in open(res_file):
            done.add(json.loads(line)["b"])
    todo = [b for b in range(B) if b not in done]
    with Pool(16) as pool, open(res_file, "a") as f:
        for b, row in pool.imap_unordered(one_rep, todo, chunksize=4):
            f.write(json.dumps({"b": b, **row}) + "\n"); f.flush()
            if b % 50 == 0:
                print(f"rep {b} done", flush=True)
    # ---- summarize ----
    rows = [json.loads(l) for l in open(res_file)]
    def ci(vals):
        v = sorted(vals); n = len(v)
        return v[int(0.025 * n)], v[int(0.975 * n)]
    CONTRASTS = {
        "H1": ("C4-M", "C1-M", [42, 1337, 20260703]),
        "H2": ("C4-M", "C4MixFT-M", [42, 1337]),
        "H3": ("C4MixFT-M", "C4Mix-M", [42, 1337]),
        "H4": ("C4-M", "C4R-M", [42, 1337]),
        "H5": ("C1-M", "C0-R", [42, 1337, 20260703]),
    }
    summary = {"B": len(rows), "clusters": NC, "seed": SEED,
               "cluster_rule": "source id (strip _<idx>, _f<frame>, .rf.<hex>)",
               "contrasts": {}}
    for h, (a, bnm, seeds) in CONTRASTS.items():
        per_seed = {}
        means = []
        for r in rows:
            ds = [r[f"{a}@{s}"] - r[f"{bnm}@{s}"] for s in seeds]
            means.append(sum(ds) / len(ds))
            for s, d in zip(seeds, ds):
                per_seed.setdefault(s, []).append(d)
        summary["contrasts"][h] = {
            "mean_ci95": [round(x, 5) for x in ci(means)],
            "per_seed_ci95": {str(s): [round(x, 5) for x in ci(v)] for s, v in per_seed.items()},
        }
    json.dump(summary, open(OUT / "source_boot_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("SOURCE_BOOT_DONE", flush=True)

if __name__ == "__main__":
    main()
