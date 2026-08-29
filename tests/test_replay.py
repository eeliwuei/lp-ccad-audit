from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("replay_results", ROOT / "analysis" / "replay_results.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReplayTest(unittest.TestCase):
    def test_canonical_results(self) -> None:
        actual = MODULE.replay()
        expected = json.loads((ROOT / "expected_outputs" / "canonical_results.json").read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            self.assertIn(key, actual)
            if isinstance(expected_value, int):
                self.assertEqual(expected_value, actual[key], key)
            else:
                self.assertTrue(math.isclose(float(actual[key]), float(expected_value), abs_tol=1e-12),
                                f"{key}: {actual[key]} != {expected_value}")

    def test_claim_boundaries(self) -> None:
        result = MODULE.replay()
        self.assertLess(abs(result["validation_protocol_mean"]), result["historical_magnitude_rule"])
        self.assertLess(result["original_evaluation_mean"], 0)
        self.assertLess(result["source_excluded_mean"], 0)
        self.assertLess(abs(result["source_exclusion_change"]), 0.001)


if __name__ == "__main__":
    unittest.main()
