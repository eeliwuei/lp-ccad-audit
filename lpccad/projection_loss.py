#!/usr/bin/env python3
"""Projection losses for matched-only LP-CCAD smoke training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


CLASS_ID = {"knife": 0, "gun": 1, "stick": 2}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    active_classes: tuple[int, ...]
    loss_type: str


VIEWS = {
    "human_only": ViewSpec("human_only", tuple(), "none"),
    "full": ViewSpec("full", (0, 1, 2), "sigmoid_bce"),
    "group_knife_stick": ViewSpec("group_knife_stick", (0, 2), "sigmoid_bce"),
    "group_gun_knife": ViewSpec("group_gun_knife", (1, 0), "sigmoid_bce"),
    "binary_knife_stick": ViewSpec("binary_knife_stick", (0, 2), "pairwise_margin"),
    "binary_gun_knife": ViewSpec("binary_gun_knife", (1, 0), "pairwise_margin"),
    "single_knife": ViewSpec("single_knife", (0,), "sigmoid_bce"),
    "single_gun": ViewSpec("single_gun", (1,), "sigmoid_bce"),
    "single_stick": ViewSpec("single_stick", (2,), "sigmoid_bce"),
}


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-7)


def giou_loss_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    iou = box_iou_xyxy(a, b)
    c_lt = torch.minimum(a[:, :2], b[:, :2])
    c_rb = torch.maximum(a[:, 2:], b[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[:, 0] * c_wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    inter_wh = (rb - lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    union = area_a + area_b - inter
    giou = iou - (c_area - union) / c_area.clamp(min=1e-7)
    return 1.0 - giou


def canonicalize_xyxy(
    boxes: torch.Tensor,
    image_shape: tuple[int, int] | list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ordered finite xyxy boxes, optionally clamped to image bounds."""
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    out = torch.stack([x1, y1, x2, y2], dim=1)
    if image_shape is not None:
        scale = _box_scale_tensor(out, image_shape)
        out[:, [0, 2]] = out[:, [0, 2]].clamp(min=0.0, max=float(scale[0]))
        out[:, [1, 3]] = out[:, [1, 3]].clamp(min=0.0, max=float(scale[1]))
    return out


def _box_scale_tensor(
    boxes: torch.Tensor,
    image_shape: tuple[int, int] | list[int] | torch.Tensor | None,
) -> torch.Tensor:
    if image_shape is None:
        # Backward-compatible fallback for smoke/debug callers. Full protocol
        # passes image_shape explicitly.
        wh = boxes.detach().amax(dim=0).clamp(min=1.0)
        return torch.tensor([wh[2], wh[3], wh[2], wh[3]], device=boxes.device, dtype=boxes.dtype)
    if isinstance(image_shape, torch.Tensor):
        if image_shape.numel() == 2:
            h, w = image_shape.flatten().to(device=boxes.device, dtype=boxes.dtype)
        else:
            raise ValueError(f"image_shape tensor must have 2 values, got {tuple(image_shape.shape)}")
    else:
        h = torch.tensor(float(image_shape[0]), device=boxes.device, dtype=boxes.dtype)
        w = torch.tensor(float(image_shape[1]), device=boxes.device, dtype=boxes.dtype)
    return torch.stack([w, h, w, h]).clamp(min=1.0)


def common_box_kd(
    student_boxes_xyxy: torch.Tensor,
    teacher_boxes_xyxy: torch.Tensor,
    mask: torch.Tensor,
    image_shape: tuple[int, int] | list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    if student_boxes_xyxy.numel() == 0 or not bool(mask.any()):
        return student_boxes_xyxy.sum() * 0.0
    # AMP can make GIoU fragile for degenerate raw boxes. Canonicalize and run
    # the auxiliary KD geometry in fp32; gradients still flow to student boxes.
    s = canonicalize_xyxy(student_boxes_xyxy[mask].float(), image_shape=image_shape)
    t = canonicalize_xyxy(teacher_boxes_xyxy[mask].detach().float(), image_shape=image_shape)
    valid = (
        torch.isfinite(s).all(dim=1)
        & torch.isfinite(t).all(dim=1)
        & ((s[:, 2] - s[:, 0]) > 1e-6)
        & ((s[:, 3] - s[:, 1]) > 1e-6)
        & ((t[:, 2] - t[:, 0]) > 1e-6)
        & ((t[:, 3] - t[:, 1]) > 1e-6)
    )
    if not bool(valid.any()):
        return student_boxes_xyxy.sum() * 0.0
    s = s[valid]
    t = t[valid]
    scale = _box_scale_tensor(torch.cat([s.detach(), t.detach()], dim=0), image_shape)
    loss = F.smooth_l1_loss(s / scale, t / scale, reduction="mean") + giou_loss_xyxy(s, t).mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def projected_class_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    class_allowed: torch.Tensor,
    view_name: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute masked LP-CCAD class KD.

    Inactive classes are sliced out, not treated as negatives. Teacher tensors
    are detached at the call boundary.
    """
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    spec = VIEWS[view_name]
    if spec.loss_type == "none" or len(spec.active_classes) == 0:
        return student_logits.sum() * 0.0
    if not bool(class_allowed.any()):
        return student_logits.sum() * 0.0

    idx = torch.tensor(spec.active_classes, device=student_logits.device, dtype=torch.long)
    s = student_logits[class_allowed][:, idx]
    t_logits = teacher_logits[class_allowed][:, idx].detach()
    t_probs = teacher_probs[class_allowed][:, idx].detach()
    if s.numel() == 0:
        return student_logits.sum() * 0.0

    if spec.loss_type == "pairwise_margin":
        if s.shape[1] != 2:
            raise ValueError(f"pairwise_margin view {view_name} requires exactly 2 active classes")
        return F.smooth_l1_loss(s[:, 0] - s[:, 1], t_logits[:, 0] - t_logits[:, 1], reduction="mean")

    # Independent sigmoid BCE. Do not softmax teacher logits.
    T = float(temperature)
    target = torch.sigmoid(t_logits / T) if T != 1.0 else t_probs
    loss = F.binary_cross_entropy_with_logits(s / T, target, reduction="none") * (T * T)
    # Primary protocol uses no hand-written exposure weights. Fairness is
    # controlled by frozen view multisets and schedules, not per-view loss hacks.
    return loss.mean(dim=1).mean()
