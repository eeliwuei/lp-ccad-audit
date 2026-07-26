"""The dose-match invariant of the randomized 2x2 factorial.

The whole point of the v3 factorial is that ORDER and TAIL are manipulated
while the *global view multiset* is held exactly fixed: every arm sees the same
views the same number of times as the frozen C4-M anneal, so a difference
between arms cannot be a difference in exposure. This test pins that invariant
against the released schedule file, reading the C4-M reference out of the
frozen v1 protocol rather than hard-coding it.

Stdlib only: PyYAML is used when available, otherwise a minimal parser for the
(machine-generated, uniformly formatted) schedule blocks is used. When PyYAML
IS available both parsers must agree, which keeps the fallback honest.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V3 = REPO / "configs/factorial_schedules/phase4_view_schedules_v3_factorial.yaml"
V1 = REPO / "configs/frozen_protocol/phase4_view_schedules_v1.yaml"

# Which classes each view keeps active in the class-distillation loss.
VIEW_CLASSES = {
    "full": ("knife", "gun", "stick"),
    "group_knife_stick": ("knife", "stick"),
    "group_gun_knife": ("gun", "knife"),
    "binary_knife_stick": ("knife", "stick"),
    "binary_gun_knife": ("gun", "knife"),
    "single_knife": ("knife",),
    "single_gun": ("gun",),
    "single_stick": ("stick",),
    "human_only": (),
}

EXPECTED_MULTISET = {
    "full": 30,
    "group_knife_stick": 20, "group_gun_knife": 20,
    "binary_knife_stick": 20, "binary_gun_knife": 20,
    "single_knife": 14, "single_gun": 13, "single_stick": 13,
}
EXPECTED_CLASS_EPOCHS = {"knife": 124, "gun": 83, "stick": 83}
EXPECTED_ARMS = [f"{tag}-s{k}" for tag in ("F-MonoMix", "F-ShufSingle", "F-ShufMix")
                 for k in (1, 2, 3)]
EPOCHS = 150


def _parse_schedules_minimal(path: Path) -> dict[str, list[str]]:
    """Parse the `schedules:` block without PyYAML.

    The file is machine-written by lpccad/schedule.py through yaml.safe_dump,
    so the block is strictly `schedules:` / two-space key / two-space `- item`.
    """
    schedules: dict[str, list[str]] = {}
    in_block = False
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():                       # a top-level key
            in_block = line.startswith("schedules:")
            current = None
            continue
        if not in_block:
            continue
        item = re.match(r"^ {2}- (.+?)\s*$", line)
        key = re.match(r"^ {2}([^\s:]+):\s*$", line)
        if item is not None and current is not None:
            schedules[current].append(item.group(1).strip().strip("'\""))
        elif key is not None:
            current = key.group(1).strip("'\"")
            schedules[current] = []
    return schedules


def load_schedules(path: Path) -> dict[str, list[str]]:
    minimal = _parse_schedules_minimal(path)
    try:
        import yaml  # noqa: PLC0415 - optional dependency by design
    except ImportError:
        return minimal
    full = yaml.safe_load(path.read_text(encoding="utf-8"))["schedules"]
    assert {k: list(v) for k, v in full.items()} == minimal, (
        f"the fallback parser disagrees with PyYAML on {path.name}")
    return {k: list(v) for k, v in full.items()}


def class_active_epochs(seq: list[str]) -> dict[str, int]:
    counts = Counter()
    for view in seq:
        for cls in VIEW_CLASSES[view]:
            counts[cls] += 1
    return {c: counts[c] for c in ("knife", "gun", "stick")}


def test_release_files_present():
    assert V3.exists(), f"missing {V3}"
    assert V1.exists(), f"missing {V1}"


def test_c4m_reference_multiset_matches_the_expected_dose():
    """The reference dose is read from the frozen v1 protocol, not asserted blind."""
    c4m = load_schedules(V1)["C4-M"]
    assert len(c4m) == EPOCHS
    assert dict(Counter(c4m)) == EXPECTED_MULTISET
    assert class_active_epochs(c4m) == EXPECTED_CLASS_EPOCHS


def test_all_nine_randomized_arms_are_dose_matched_to_c4m():
    c4m_counts = Counter(load_schedules(V1)["C4-M"])
    schedules = load_schedules(V3)
    assert sorted(schedules) == sorted(EXPECTED_ARMS), sorted(schedules)
    for name, seq in schedules.items():
        assert len(seq) == EPOCHS, f"{name}: {len(seq)} epochs"
        assert Counter(seq) == c4m_counts, f"{name}: global view multiset differs from C4-M"
        assert class_active_epochs(seq) == EXPECTED_CLASS_EPOCHS, name


def test_head_tail_split_actually_differs_between_cells():
    """Dose is identical, so the manipulation must live in the ORDER of the
    150 epochs: the single-tail cells end on single-class views, the mixed-tail
    cells do not. Without this the factorial would manipulate nothing."""
    schedules = load_schedules(V3)
    head_len = 110
    for name, seq in schedules.items():
        tail = seq[head_len:]
        assert len(tail) == EPOCHS - head_len
        if name.startswith("F-ShufSingle"):
            assert all(v.startswith("single_") for v in tail), f"{name}: tail is not single-class"
        else:                                   # F-MonoMix / F-ShufMix
            assert not all(v.startswith("single_") for v in tail), f"{name}: tail is single-class"
            cards = {len(VIEW_CLASSES[v]) for v in tail}
            assert cards > {1}, f"{name}: mixed tail must contain non-single views"


def test_schedule_file_sha256_matches_the_recorded_digest():
    """The paper cites schedule-file SHA prefix bfdd9285; the released file must be it."""
    import hashlib
    digest = hashlib.sha256(V3.read_bytes()).hexdigest()
    recorded = (V3.parent / (V3.name + ".sha256")).read_text().split()[0]
    assert digest == recorded, "schedule file does not match its recorded sha256"
    assert digest.startswith("bfdd9285"), digest
