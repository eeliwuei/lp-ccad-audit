#!/usr/bin/env python3
"""Dataset/evidence adapter for matched-only LP-CCAD smoke training."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


CLASS_NAME_TO_ID = {"knife": 0, "gun": 1, "stick": 2}


def stem_aliases(value: str | Path | None) -> set[str]:
    if value is None:
        return set()
    stem = Path(str(value)).stem
    aliases = {stem}
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[-1] == parts[-2]:
        aliases.add("_".join(parts[:2]))
    return aliases


def xyxy_to_xywh_norm(box: list[float], ori_shape: tuple[int, int] | list[int], out_shape: tuple[int, int] | list[int], flip_lr: bool = False) -> list[float]:
    """Map original-image xyxy to letterboxed normalized xywh in model input coordinates."""
    h0, w0 = float(ori_shape[0]), float(ori_shape[1])
    h, w = float(out_shape[0]), float(out_shape[1])
    r = min(w / max(w0, 1.0), h / max(h0, 1.0))
    new_w, new_h = round(w0 * r), round(h0 * r)
    # Ultralytics LetterBox integer padding convention (round(dw-0.1)),
    # matched so teacher/GT boxes land in the exact frame the student
    # trains on. Was float half-padding (~0.5px common-mode offset).
    pad_x = float(int(round((w - new_w) / 2.0 - 0.1)))
    pad_y = float(int(round((h - new_h) / 2.0 - 0.1)))
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = x1 * r + pad_x
    x2 = x2 * r + pad_x
    y1 = y1 * r + pad_y
    y2 = y2 * r + pad_y
    if flip_lr:
        x1, x2 = w - x2, w - x1
    cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return [cx, cy, bw, bh]


def xywh_norm_to_xyxy_abs(box: torch.Tensor, shape: tuple[int, int] | list[int]) -> torch.Tensor:
    h, w = float(shape[0]), float(shape[1])
    cx, cy, bw, bh = box.unbind(-1)
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax = xywh_norm_to_xyxy_abs(a, (1, 1))
    bx = xywh_norm_to_xyxy_abs(b, (1, 1))
    lt = torch.maximum(ax[..., :2], bx[..., :2])
    rb = torch.minimum(ax[..., 2:], bx[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    aa = (ax[..., 2] - ax[..., 0]).clamp(min=0) * (ax[..., 3] - ax[..., 1]).clamp(min=0)
    bb = (bx[..., 2] - bx[..., 0]).clamp(min=0) * (bx[..., 3] - bx[..., 1]).clamp(min=0)
    return inter / (aa + bb - inter).clamp(min=1e-7)


class CCADEvidenceIndex:
    def __init__(self, evidence_bank: str | Path):
        self.path = Path(evidence_bank)
        self.by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.e1_box = 0
        self.e1_cls = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("evidence_type") != "E1_matched_refinement":
                    continue
                if not row.get("box_kd_allowed", row.get("e1_box_eligible", False)):
                    continue
                self.e1_box += 1
                self.e1_cls += int(bool(row.get("class_kd_allowed", row.get("e1_cls_eligible", False))))
                keys = set()
                keys.update(stem_aliases(str(row.get("image_id"))))
                for pkey in ["image_path", "image_path_resolved"]:
                    if row.get(pkey):
                        keys.update(stem_aliases(row[pkey]))
                for key in keys:
                    self.by_stem[key].append(row)

    def for_image(self, image_path: str | Path) -> list[dict[str, Any]]:
        return self.by_stem.get(Path(str(image_path)).stem, [])


def build_batch_teacher_targets(batch: dict[str, Any], evidence: CCADEvidenceIndex, device: torch.device, iou_match_thr: float = 0.999) -> dict[str, torch.Tensor]:
    """Build teacher targets aligned with batch human GT rows.

    Returns one row per human GT in the batch. Missing evidence rows are masked
    out and receive human loss only.
    """
    n = int(batch["cls"].shape[0])
    if n == 0:
        zf = torch.zeros(0, device=device)
        return {
            "has_box": zf.bool(),
            "has_cls": zf.bool(),
            "teacher_box_xyxy_abs": torch.zeros(0, 4, device=device),
            "teacher_logits": torch.zeros(0, 3, device=device),
            "teacher_probs": torch.zeros(0, 3, device=device),
            "gt_uid": [],
        }
    has_box = torch.zeros(n, dtype=torch.bool, device=device)
    has_cls = torch.zeros(n, dtype=torch.bool, device=device)
    teacher_boxes = torch.zeros(n, 4, dtype=torch.float32, device=device)
    teacher_logits = torch.zeros(n, 3, dtype=torch.float32, device=device)
    teacher_probs = torch.zeros(n, 3, dtype=torch.float32, device=device)
    gt_uid: list[str] = []
    batch_idx = batch["batch_idx"].detach().cpu().long()
    cls = batch["cls"].detach().cpu().view(-1).long()
    bboxes = batch["bboxes"].detach().cpu().float()

    used_queries: set[tuple[str, int]] = set()
    for label_i in range(n):
        bi = int(batch_idx[label_i])
        im_file = batch["im_file"][bi]
        ori_shape = batch["ori_shape"][bi]
        resized_shape = batch["resized_shape"][bi]
        target_cls = int(cls[label_i])
        candidates = [r for r in evidence.for_image(im_file) if int(r.get("matched_gt_class_id", -1)) == target_cls]
        if not candidates:
            gt_uid.append(f"{Path(im_file).stem}:{label_i}:no_teacher")
            continue
        gt_box = bboxes[label_i].unsqueeze(0)
        best = (0.0, None, None)
        for row in candidates:
            qkey = (str(row.get("image_id")), int(row.get("query_id")))
            if qkey in used_queries:
                continue
            transformed_gt = torch.tensor(xyxy_to_xywh_norm(row["matched_gt_bbox"], ori_shape, resized_shape), dtype=torch.float32).unsqueeze(0)
            iou = float(xywh_iou(gt_box, transformed_gt)[0])
            if iou > best[0]:
                best = (iou, row, transformed_gt)
        if best[1] is None or best[0] < iou_match_thr:
            gt_uid.append(f"{Path(im_file).stem}:{label_i}:unmatched")
            continue
        row = best[1]
        used_queries.add((str(row.get("image_id")), int(row.get("query_id"))))
        tbox_norm = torch.tensor(xyxy_to_xywh_norm(row["bbox_xyxy"], ori_shape, resized_shape), dtype=torch.float32, device=device)
        teacher_boxes[label_i] = xywh_norm_to_xyxy_abs(tbox_norm, resized_shape)
        teacher_logits[label_i] = torch.tensor(row["raw_class_logits"][:3], dtype=torch.float32, device=device).detach()
        teacher_probs[label_i] = torch.tensor(row["class_probabilities"][:3], dtype=torch.float32, device=device).detach()
        has_box[label_i] = bool(row.get("box_kd_allowed", True))
        has_cls[label_i] = bool(row.get("class_kd_allowed", False))
        gt_uid.append(f"{Path(im_file).stem}:{int(row.get('matched_gt_id'))}:{int(row.get('query_id'))}")
    return {
        "has_box": has_box,
        "has_cls": has_cls,
        "teacher_box_xyxy_abs": teacher_boxes,
        "teacher_logits": teacher_logits,
        "teacher_probs": teacher_probs,
        "gt_uid": gt_uid,
    }
