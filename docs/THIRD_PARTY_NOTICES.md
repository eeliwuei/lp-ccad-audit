# Third-party notices

No third-party source code is vendored in this repository. Every file under
`lpccad/`, `scripts/`, `tests/` and `examples/` was written by the authors.
This document records the external components the work *interfaces with* and
their terms.

## ultralytics 8.4.24 — AGPL-3.0

The student (YOLO26n) was trained and evaluated through Ultralytics
`8.4.24`. **No Ultralytics code is included or redistributed here.**

* Only `scripts/freeze_eval.py` touches it, and only through a late
  `from ultralytics import YOLO` inside a function; that script is released as
  documentation of the frozen evaluation procedure and cannot run without the
  restricted data anyway.
* Two conventions in `lpccad/evaluation_adapter.py` are *matched to*
  Ultralytics behaviour — the integer LetterBox padding rule
  (`round(dw - 0.1)`) and the batch dictionary keys (`cls`, `bboxes`,
  `batch_idx`, `im_file`, `ori_shape`, `resized_shape`). Matching an
  interface is not copying an implementation.
* `ultralytics` is deliberately **absent from `requirements.txt`**. Users who
  want to attempt the training/evaluation tier must install it themselves and
  are then responsible for their own AGPL-3.0 compliance, including the
  network-use clause if they expose a service built on it.

Upstream: <https://github.com/ultralytics/ultralytics> (AGPL-3.0).

## D-FINE — Apache-2.0

The teacher is D-FINE-X. Its weights and inference code were used to produce
the frozen teacher evidence bank; neither the code nor the weights are
redistributed here. Apache-2.0 permits use and modification with attribution
and a notice of changes; we make no modified redistribution.

Upstream: the D-FINE repository and paper (Apache-2.0).

## faster-coco-eval — Apache-2.0

The frozen evaluator for every mAP number in the paper. Used as an installed
dependency (`pip install faster-coco-eval`); not vendored. It is imported by
`scripts/source_cluster_bootstrap.py` and by
`tests/test_bootstrap_duplicate_ids.py`, which skips cleanly when it is absent.

It replaced `pycocotools` in this project only because the training
environment's `pycocotools` build had a NumPy ABI mismatch; the two agree on
the metric.

Upstream: <https://github.com/MiXaiLL76/faster_coco_eval> (Apache-2.0).

## PyTorch, NumPy, PyYAML, pytest, matplotlib

Standard scientific-Python dependencies, used as installed packages under their
own permissive licences (BSD-3-Clause / MIT). None are vendored.

## Datasets

Dataset terms are a separate matter and are documented in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md). No dataset content is included in
this repository under any licence.

## Licence of this repository

Not yet fixed — see `LICENSE_PENDING` at the repository root. The short version:
this code is standalone and interfaces with AGPL-3.0 Ultralytics only at
arm's length, so MIT, BSD-3-Clause and Apache-2.0 are all viable choices and
the decision is reserved to the authors.
