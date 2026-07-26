"""Why the source-clustered bootstrap must clone-and-RENUMBER, not just repeat.

scripts/source_cluster_bootstrap.py resamples source clusters with replacement.
When a cluster is drawn twice its images must be re-inserted under *fresh*
image and annotation ids, and the arm's predictions must be remapped onto those
fresh ids (see `one_rep`). A COCO index is a dict keyed by id, so appending the
same ids again does not create a second image: the duplicate row silently
merges into the original and the replicate cannot be expressed in the GT at all.

The minimal logic is reproduced here rather than imported, so the test is
self-contained and does not need the restricted locked-test artifacts.

Measured on the synthetic 4-image set below with faster-coco-eval, drawing
image 1 twice:

    baseline, every image once                      mAP = 0.25743
    clone-and-renumber image 1 (5 distinct images)  mAP = 0.40594  <- counts
    same ids, GT indexed by id (the bug)            mAP = 0.25743  <- vanished

The bug is silent in exactly the worst way: the replicate returns the baseline
number, so a bootstrap built on it reports intervals with no resampling
variance in them at all.

One subtlety worth recording, because it is the reason the structural check
below exists: a naive resampler that also duplicates the *annotation* rows
verbatim appends the duplicated GT box to the ORIGINAL image's GT list, and for
exact clones the greedy matching then happens to return the correct resample's
value (0.40594) for the wrong reason. mAP is therefore not sufficient evidence
either way - the id-level check is. Assertions here are on the relations and on
the index, not on evaluator-version-specific constants.

Skipped when faster-coco-eval (the evaluator the paper froze) is absent.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
from io import StringIO

import pytest

pytest.importorskip(
    "faster_coco_eval",
    reason="faster-coco-eval is not installed; `pip install faster-coco-eval` to run this test",
)

CATEGORIES = [{"id": 1, "name": "knife"}]


def synthetic_dataset():
    """4 images, one 20x20 GT box each; image 1 is detected perfectly, the
    other three detections are offset far enough to miss at every IoU."""
    images = [{"id": i, "file_name": f"clip_{i}.jpg", "width": 100, "height": 100}
              for i in (1, 2, 3, 4)]
    annotations = [{"id": i, "image_id": i, "category_id": 1,
                    "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
                   for i in (1, 2, 3, 4)]
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [30, 30, 20, 20], "score": 0.8},
        {"image_id": 3, "category_id": 1, "bbox": [32, 30, 20, 20], "score": 0.7},
        {"image_id": 4, "category_id": 1, "bbox": [35, 30, 20, 20], "score": 0.6},
    ]
    return images, annotations, predictions


def evaluate(images, annotations, predictions) -> float:
    """mAP50-95 with the evaluator the paper froze."""
    from faster_coco_eval import COCO, COCOeval_faster
    gt = {"images": copy.deepcopy(images), "annotations": copy.deepcopy(annotations),
          "categories": copy.deepcopy(CATEGORIES)}
    with redirect_stdout(StringIO()):
        coco_gt = COCO(gt)
        coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        ev = COCOeval_faster(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [c["id"] for c in CATEGORIES]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def coco_index_sizes(images, annotations) -> tuple[int, int]:
    """(unique images, GT rows reachable through the image index)."""
    from faster_coco_eval import COCO
    gt = {"images": copy.deepcopy(images), "annotations": copy.deepcopy(annotations),
          "categories": copy.deepcopy(CATEGORIES)}
    with redirect_stdout(StringIO()):
        coco = COCO(gt)
    return len(coco.imgs), sum(len(v) for v in coco.imgToAnns.values())


def _index(rows, key):
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(row[key], []).append(row)
    return out


def resample_with_renumbering(images, annotations, predictions, draw):
    """The minimal core of scripts/source_cluster_bootstrap.py `one_rep`:
    every drawn image is re-inserted with a fresh id, its annotations get fresh
    ids pointing at the fresh image, and the predictions are remapped."""
    ann_by_img = _index(annotations, "image_id")
    pred_by_img = _index(predictions, "image_id")
    new_images, new_annotations, new_predictions = [], [], []
    next_img_id, next_ann_id = 1, 1
    for old_id in draw:
        source = next(im for im in images if im["id"] == old_id)
        new_images.append({**source, "id": next_img_id})
        for ann in ann_by_img.get(old_id, []):
            new_annotations.append({**ann, "id": next_ann_id, "image_id": next_img_id})
            next_ann_id += 1
        for pred in pred_by_img.get(old_id, []):
            new_predictions.append({**pred, "image_id": next_img_id})
        next_img_id += 1
    return new_images, new_annotations, new_predictions


def resample_without_renumbering(images, annotations, predictions, draw):
    """The bug: rows are repeated with their original ids. The GT is keyed by
    id, so the drawn-twice image contributes its annotations once, while the
    repeated detections are scored as extra predictions on the same image."""
    ann_by_img = _index(annotations, "image_id")
    pred_by_img = _index(predictions, "image_id")
    new_images, new_predictions = [], []
    seen: set[int] = set()
    new_annotations: list[dict] = []
    for old_id in draw:
        new_images.append(dict(next(im for im in images if im["id"] == old_id)))
        new_predictions.extend(dict(p) for p in pred_by_img.get(old_id, []))
        if old_id not in seen:            # GT loaded once per unique id
            new_annotations.extend(dict(a) for a in ann_by_img.get(old_id, []))
            seen.add(old_id)
    return new_images, new_annotations, new_predictions


DRAW = [1, 1, 2, 3, 4]      # bootstrap replicate that draws image 1 twice


def test_renumbered_duplicate_changes_the_map():
    images, annotations, predictions = synthetic_dataset()
    baseline = evaluate(images, annotations, predictions)
    resampled = evaluate(*resample_with_renumbering(images, annotations, predictions, DRAW))
    assert abs(resampled - baseline) > 1e-6, (
        f"resampling image 1 twice left mAP unchanged ({resampled} vs {baseline}); "
        "the bootstrap would then carry no resampling variance at all")
    assert resampled > baseline, "counting the well-detected image twice should raise mAP"


def test_renumbering_is_what_makes_the_clone_exist():
    images, annotations, predictions = synthetic_dataset()
    r_imgs, r_anns, _ = resample_with_renumbering(images, annotations, predictions, DRAW)
    n_imgs, n_anns, _ = resample_without_renumbering(images, annotations, predictions, DRAW)
    assert len(r_imgs) == len(n_imgs) == 5, "both resamplers appended five image rows"
    assert coco_index_sizes(r_imgs, r_anns) == (5, 5), "renumbered clone must be its own image"
    assert coco_index_sizes(n_imgs, n_anns) == (4, 4), (
        "duplicate image ids no longer collapse in this evaluator; the premise of "
        "this test must be re-derived before trusting the bootstrap code")


def test_duplicate_ids_do_not_produce_a_valid_resample():
    images, annotations, predictions = synthetic_dataset()
    baseline = evaluate(images, annotations, predictions)
    correct = evaluate(*resample_with_renumbering(images, annotations, predictions, DRAW))
    naive = evaluate(*resample_without_renumbering(images, annotations, predictions, DRAW))
    assert abs(naive - correct) > 1e-6, (
        f"same-id duplication ({naive}) coincided with the correct resample ({correct}); "
        "it must never be treated as equivalent")
    assert abs(naive - baseline) < 1e-12, (
        f"expected the same-id replicate to collapse back to the baseline "
        f"({baseline}), got {naive}; the failure mode changed and the note in "
        "this module's docstring must be re-derived")


def test_drawing_every_cluster_once_reproduces_the_baseline():
    """Sanity check on the resampler itself: the identity draw is a no-op."""
    images, annotations, predictions = synthetic_dataset()
    baseline = evaluate(images, annotations, predictions)
    identity = evaluate(*resample_with_renumbering(images, annotations, predictions, [1, 2, 3, 4]))
    assert abs(identity - baseline) < 1e-12


def test_predictions_must_be_remapped_onto_the_new_ids():
    """If the GT is renumbered but the predictions are not, every clone becomes
    an unpredicted image and the replicate is silently penalised."""
    images, annotations, predictions = synthetic_dataset()
    r_imgs, r_anns, r_preds = resample_with_renumbering(images, annotations, predictions, DRAW)
    stale = [dict(p) for p in predictions] + [dict(predictions[0])]   # original ids kept
    correct = evaluate(r_imgs, r_anns, r_preds)
    broken = evaluate(r_imgs, r_anns, stale)
    assert broken < correct
