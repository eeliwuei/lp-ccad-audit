from __future__ import annotations

import csv
import hashlib
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

S6_SHA256 = "52a886ba6f4880bab5f203a05050cadb51bc358d340b082e3efd2601e437948d"
SCHEDULE_SHA256 = {
    "frozen_protocol/phase4_view_schedules_v1.yaml":
        "090d4829048b729c0f844075c303dd4b6f65e9e9763001551454a117d97ac414",
    "frozen_protocol/phase4_view_schedules_v2.yaml":
        "2cac6f05dd13d63d9ac352cd93246f0951d48dba328d41e903e16086f642784a",
    "factorial_schedules/phase4_view_schedules_v3_factorial.yaml":
        "bfdd92851b104eae7468b6ea83ad70812165167c364d3e816fda45d8648a10ef",
}
SCHEDULE_COUNTS = {
    "frozen_protocol/phase4_view_schedules_v1.yaml": 8,
    "frozen_protocol/phase4_view_schedules_v2.yaml": 9,
    "factorial_schedules/phase4_view_schedules_v3_factorial.yaml": 9,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_epoch_counts(path: Path) -> dict[str, int]:
    """Count epoch-list entries under the top-level schedules mapping."""

    in_schedules = False
    current: str | None = None
    counts: Counter[str] = Counter()
    schedule_name = re.compile(r"^  ([^:#][^:]*):\s*$")

    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "schedules:":
            in_schedules = True
            continue
        if in_schedules and line == "ledger:":
            break
        if not in_schedules:
            continue

        match = schedule_name.match(line)
        if match:
            current = match.group(1)
            counts.setdefault(current, 0)
        elif line.startswith("  - "):
            if current is None:
                raise AssertionError(f"Schedule entry without a name in {path}")
            counts[current] += 1

    return dict(counts)


class S6AndScheduleIntegrityTest(unittest.TestCase):
    def test_s6_byte_integrity(self) -> None:
        path = ROOT / "data" / "S6_Data.csv"
        self.assertTrue(path.is_file())
        self.assertEqual(S6_SHA256, sha256(path))

    def test_s6_structure_and_counts(self) -> None:
        path = ROOT / "data" / "S6_Data.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(2772, len(rows))
        self.assertEqual(2772, len({row["record_id"] for row in rows}))
        self.assertEqual(
            {
                "schedule_epoch": 2700,
                "source_exclusion_contrast": 21,
                "gate_audit_aggregate": 14,
                "frozen_artifact_inventory": 13,
                "run_schedule_binding": 12,
                "resampling_parameter": 5,
                "source_exclusion_aggregate": 5,
                "equivalence_parameter": 2,
            },
            dict(Counter(row["record_type"] for row in rows)),
        )
        epoch_keys = {
            (row["schedule_version"], row["schedule_id"], row["epoch"])
            for row in rows
            if row["record_type"] == "schedule_epoch"
        }
        self.assertEqual(2700, len(epoch_keys))

    def test_schedule_manifest_and_hashes(self) -> None:
        manifest = ROOT / "configs" / "SCHEDULE_SHA256SUMS"
        recorded = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            recorded[filename] = digest

        self.assertEqual(SCHEDULE_SHA256, recorded)
        for filename, expected in SCHEDULE_SHA256.items():
            self.assertEqual(expected, sha256(ROOT / "configs" / filename), filename)

    def test_schedule_registry_structure(self) -> None:
        for filename, expected_schedule_count in SCHEDULE_COUNTS.items():
            counts = schedule_epoch_counts(ROOT / "configs" / filename)
            self.assertEqual(expected_schedule_count, len(counts), filename)
            self.assertTrue(counts, filename)
            self.assertEqual({150}, set(counts.values()), filename)


if __name__ == "__main__":
    unittest.main()
