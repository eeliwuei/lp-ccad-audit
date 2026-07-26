#!/usr/bin/env python3
"""Run the LP-CCAD projection loss on synthetic data -- no dataset, no GPU.

This is the smallest thing that shows what the method under audit actually
does: the class-distillation loss is computed through a projection onto a
time-indexed *active-class view*. Inactive classes are sliced out of the loss,
not pushed towards zero, so a single-class view still produces a gradient --
on that one class only. Everything the audit measures is a consequence of the
ORDER in which these views are visited (see configs/factorial_schedules/).

Usage (needs only PyTorch):

    python3 examples/synthetic_minimal_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover
    sys.exit("PyTorch is required: pip install torch (see requirements.txt)")

from lpccad.projection_loss import CLASS_ID, VIEWS, projected_class_kd

VIEW_ORDER = [
    ("full", "all three classes active (epoch 1 of the anneal)"),
    ("group_knife_stick", "group view: knife + stick"),
    ("group_gun_knife", "group view: gun + knife"),
    ("binary_knife_stick", "pairwise margin view: knife vs stick"),
    ("binary_gun_knife", "pairwise margin view: gun vs knife"),
    ("single_knife", "single-class view: knife only (the terminal tail)"),
    ("single_gun", "single-class view: gun only"),
    ("single_stick", "single-class view: stick only"),
    ("human_only", "no distillation at all (the C0-R baseline arm)"),
]


def make_batch(n: int = 6, seed: int = 42):
    """A tiny synthetic batch: n matched student/teacher rows over 3 classes."""
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(n, 3, generator=g, requires_grad=True)
    teacher_logits = torch.randn(n, 3, generator=g) * 1.5
    teacher_probs = torch.sigmoid(teacher_logits)
    # every row carries usable class evidence in this toy example
    class_allowed = torch.ones(n, dtype=torch.bool)
    return student_logits, teacher_logits, teacher_probs, class_allowed


def main() -> int:
    torch.manual_seed(0)
    student_logits, teacher_logits, teacher_probs, class_allowed = make_batch()

    print(f"class ids: {CLASS_ID}")
    print(f"student logits: {tuple(student_logits.shape)} (rows x classes)\n")
    print(f"{'view':<20}{'active classes [loss type]':<34}{'loss':>12}{'grad L1':>12}")
    print("-" * 78)

    losses = {}
    for view, description in VIEW_ORDER:
        spec = VIEWS[view]
        active = [name for name, idx in CLASS_ID.items() if idx in spec.active_classes]
        s = student_logits.detach().clone().requires_grad_(True)
        loss = projected_class_kd(s, teacher_logits, teacher_probs, class_allowed, view)
        loss.backward()
        grad = s.grad.detach()
        losses[view] = float(loss.detach())
        label = ",".join(active) if active else "(none)"
        print(f"{view:<20}{label + ' [' + spec.loss_type + ']':<34}"
              f"{float(loss.detach()):>12.6f}{float(grad.abs().sum()):>12.6f}")

    print("-" * 78)
    for view, description in VIEW_ORDER:
        print(f"  {view:<20} {description}")
    print()

    # ---- the property the projection is built on -------------------------
    s = student_logits.detach().clone().requires_grad_(True)
    loss = projected_class_kd(s, teacher_logits, teacher_probs, class_allowed, "single_knife")
    loss.backward()
    grad = s.grad.detach()
    knife, gun, stick = CLASS_ID["knife"], CLASS_ID["gun"], CLASS_ID["stick"]
    knife_grad = float(grad[:, knife].abs().sum())
    other_grad = float(grad[:, gun].abs().sum() + grad[:, stick].abs().sum())

    print("single-view gradient check (the projection, not a mask over negatives):")
    print(f"  |grad| on the ACTIVE class (knife)   = {knife_grad:.6f}")
    print(f"  |grad| on the INACTIVE classes       = {other_grad:.6f}")
    assert knife_grad > 0.0, "single-class view produced no gradient -- projection is broken"
    assert other_grad == 0.0, "inactive classes received gradient -- they were not sliced out"
    print("  OK: the single-class view trains knife only; gun and stick are sliced")
    print("      out of the loss entirely rather than being treated as negatives.")

    # the human_only view is the no-distillation control arm
    assert losses["human_only"] == 0.0
    print("  OK: the human_only view contributes exactly zero (the C0-R control).")
    print("\nNothing here reproduces a paper number; results/ holds the frozen runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
