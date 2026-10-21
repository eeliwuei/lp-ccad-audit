from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "replay_results.py"
SPEC = importlib.util.spec_from_file_location("replay_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

FLOAT_SLACK = 1e-12

# Values as printed in the manuscript, keyed by (group, item) of the
# recomputation checks. The source-exclusion shift is reported in the text as
# a magnitude (0.000303); the signed value is excluded minus original.
MANUSCRIPT_PRINTED = {
    ("h2_validation", "difference seed 42"): "+0.01159",
    ("h2_validation", "difference seed 1337"): "+0.00853",
    ("h2_validation", "difference seed 20260703"): "+0.00723",
    ("h2_validation", "protocol mean (seeds 42, 1337)"): "+0.01006",
    ("h2_validation", "three-seed mean"): "+0.00912",
    ("evaluation_route_2553", "difference seed 42"): "+0.005594",
    ("evaluation_route_2553", "difference seed 1337"): "-0.023817",
    ("evaluation_route_2553", "two-seed mean"): "-0.009112",
    ("evaluation_route_2019", "difference seed 42"): "+0.005158",
    ("evaluation_route_2019", "difference seed 1337"): "-0.023987",
    ("evaluation_route_2019", "two-seed mean"): "-0.009414",
    ("evaluation_routes", "mean shift (2,019 minus 2,553)"): "-0.000303",
    ("followup_cell", "mono_mixed mean"): "0.368539",
    ("followup_cell", "mono_single mean"): "0.373885",
    ("followup_cell", "shuf_mixed mean"): "0.369252",
    ("followup_cell", "shuf_single mean"): "0.375253",
    ("followup_cell", "mono_mixed sample SD"): "0.003124",
    ("followup_cell", "mono_single sample SD"): "0.007620",
    ("followup_cell", "shuf_mixed sample SD"): "0.003484",
    ("followup_cell", "shuf_single sample SD"): "0.004003",
    ("followup_contrast", "order_axis seed 42"): "+0.00386",
    ("followup_contrast", "order_axis seed 1337"): "-0.00110",
    ("followup_contrast", "order_axis seed 20260703"): "-0.00588",
    ("followup_contrast", "order_axis mean"): "-0.00104",
    ("followup_contrast", "order_axis ci95 low"): "-0.01315",
    ("followup_contrast", "order_axis ci95 high"): "+0.01107",
    ("followup_contrast", "tail_axis seed 42"): "+0.00650",
    ("followup_contrast", "tail_axis seed 1337"): "+0.00151",
    ("followup_contrast", "tail_axis seed 20260703"): "+0.00901",
    ("followup_contrast", "tail_axis mean"): "+0.00567",
    ("followup_contrast", "tail_axis ci95 low"): "-0.00380",
    ("followup_contrast", "tail_axis ci95 high"): "+0.01515",
    ("followup_contrast", "difference_of_differences seed 42"): "+0.01077",
    ("followup_contrast", "difference_of_differences seed 1337"): "-0.00191",
    ("followup_contrast", "difference_of_differences seed 20260703"): "-0.01083",
    ("followup_contrast", "difference_of_differences mean"): "-0.00066",
    ("followup_contrast", "difference_of_differences ci95 low"): "-0.02762",
    ("followup_contrast", "difference_of_differences ci95 high"): "+0.02631",
}
N_ENDPOINT_CROSSCHECKS = 10


def run_script(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env,
                          check=False)


def write_modified_csv(source: Path, target: Path, edit) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in (edit(row) for row in reader) if row is not None]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RecomputationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = MODULE.replay()
        cls.checks = {(row["group"], row["item"]): row for row in cls.result["recomputation_checks"]}

    def test_every_expected_value_is_recomputed(self) -> None:
        self.assertEqual(len(MANUSCRIPT_PRINTED) + N_ENDPOINT_CROSSCHECKS, len(self.result["recomputation_checks"]))
        self.assertEqual(len(self.checks), len(self.result["recomputation_checks"]))
        for key in MANUSCRIPT_PRINTED:
            self.assertIn(key, self.checks)

    def test_recomputed_values_agree_with_stored_s1_values(self) -> None:
        for key, row in self.checks.items():
            with self.subTest(item=key):
                tolerance = 0.5 * 10 ** -row["printed_decimals"]
                difference = abs(row["recomputed_value"] - float(row["stored_text"]))
                self.assertLessEqual(difference, tolerance + FLOAT_SLACK,
                                     f"{key}: recomputed {row['recomputed_value']!r}, "
                                     f"stored {row['s1_record_id']}.{row['s1_field']} = {row['stored_text']}")
                self.assertTrue(row["pass"], key)
        summary = self.result["recomputation_summary"]
        self.assertTrue(summary["all_pass"])
        self.assertEqual(0, summary["n_fail"])
        MODULE.verify(self.result)

    def test_recomputed_values_agree_with_manuscript_printed_values(self) -> None:
        for key, printed in MANUSCRIPT_PRINTED.items():
            with self.subTest(item=key):
                decimals = len(printed.split(".")[1])
                self.assertEqual(decimals, self.checks[key]["printed_decimals"], key)
                difference = abs(self.checks[key]["recomputed_value"] - float(printed))
                self.assertLessEqual(difference, 0.5 * 10 ** -decimals + FLOAT_SLACK,
                                     f"{key}: recomputed {self.checks[key]['recomputed_value']!r}, printed {printed}")

    def test_paired_t_critical_value(self) -> None:
        config = json.loads((ROOT / "configs" / "replay_config.json").read_text(encoding="utf-8"))
        t_critical = config["recomputation"]["paired_t_critical_0975_df2"]
        self.assertEqual(4.302652729911275, t_critical)
        p = 0.975
        self.assertTrue(math.isclose(t_critical, (2 * p - 1) / math.sqrt(2 * p * (1 - p)), abs_tol=1e-9))

    def test_generated_output_is_current(self) -> None:
        committed = ROOT / "expected_outputs" / "replayed_results.generated.json"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "replayed.json"
            completed = run_script("--output", str(output))
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn("recomputed values agree with the stored S1 values", completed.stdout)
            self.assertEqual(committed.read_bytes(), output.read_bytes())

    def test_script_fails_loudly_on_changed_stored_value(self) -> None:
        def edit(row: dict) -> dict:
            if row["record_id"] == "FOLLOWUP_tail_axis_composite_summary":
                row["ci95_high"] = "0.01525"
            return row

        with tempfile.TemporaryDirectory() as tmp:
            s1 = Path(tmp) / "S1_Data.csv"
            write_modified_csv(ROOT / "data" / "S1_Data.csv", s1, edit)
            completed = run_script("--s1-data", str(s1), "--output", str(Path(tmp) / "out.json"))
            self.assertEqual(1, completed.returncode)
            self.assertIn("RECOMPUTATION MISMATCH: 1 of", completed.stderr)
            self.assertIn("tail_axis ci95 high", completed.stderr)

            result = MODULE.replay(s1_path=s1)
            self.assertFalse(result["recomputation_summary"]["all_pass"])
            with self.assertRaises(MODULE.RecomputationMismatch):
                MODULE.verify(result)

    def test_script_fails_loudly_on_changed_endpoint(self) -> None:
        def edit(row: dict) -> dict:
            if row["context"] == "corrected_pipeline_validation_followup" and row["arm"] == "shuf_mixed" \
                    and row["seed"] == "1337":
                row["mAP50_95"] = f"{float(row['mAP50_95']) + 0.001:.10f}"
            return row

        with tempfile.TemporaryDirectory() as tmp:
            s2 = Path(tmp) / "S2_Data.csv"
            write_modified_csv(ROOT / "data" / "S2_Data.csv", s2, edit)
            completed = run_script("--s2-data", str(s2), "--output", str(Path(tmp) / "out.json"))
            self.assertEqual(1, completed.returncode)
            self.assertIn("RECOMPUTATION MISMATCH", completed.stderr)
            self.assertIn("shuf_mixed mean", completed.stderr)

    def test_script_fails_loudly_on_missing_endpoint(self) -> None:
        def edit(row: dict) -> dict | None:
            if row["context"] == "corrected_pipeline_validation_followup" and row["arm"] == "mono_single" \
                    and row["seed"] == "20260703":
                return None
            return row

        with tempfile.TemporaryDirectory() as tmp:
            s2 = Path(tmp) / "S2_Data.csv"
            write_modified_csv(ROOT / "data" / "S2_Data.csv", s2, edit)
            completed = run_script("--s2-data", str(s2), "--output", str(Path(tmp) / "out.json"))
            self.assertEqual(1, completed.returncode)
            self.assertIn("RECOMPUTATION MISMATCH: input error", completed.stderr)
            with self.assertRaises(MODULE.RecomputationInputError):
                MODULE.replay(s2_path=s2)


if __name__ == "__main__":
    unittest.main()
