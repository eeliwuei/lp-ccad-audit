#!/usr/bin/env python3
"""Replay the aggregate analytical results reported in the PLOS candidate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "S1_Data.csv"
CONFIG = ROOT / "configs" / "replay_config.json"
OUTPUT = ROOT / "expected_outputs" / "replayed_results.generated.json"


def load_records() -> dict[str, dict[str, str]]:
    with DATA.open(encoding="utf-8") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def value(records: dict[str, dict[str, str]], record_id: str) -> float:
    return float(records[record_id]["value"])


def replay() -> dict[str, float | int]:
    records = load_records()
    with CONFIG.open(encoding="utf-8") as handle:
        config = json.load(handle)

    protocol = [value(records, record_id) for record_id in config["validation_protocol_records"]]
    all_validation = [value(records, record_id) for record_id in config["validation_all_records"]]
    result = {
        "validation_protocol_mean": mean(protocol),
        "validation_three_seed_mean": mean(all_validation),
        "historical_magnitude_rule": config["historical_magnitude_rule"],
        "original_evaluation_mean": value(records, "original_full_evaluation_H2_mean"),
        "source_excluded_mean": value(records, "source_excluded_evaluation_H2_mean"),
        "source_exclusion_change": value(records, "SOURCE_EXCLUSION_H2_CHANGE"),
        "order_axis_mean": value(records, "FOLLOWUP_order_axis_composite_summary"),
        "order_axis_ci95_low": float(records["FOLLOWUP_order_axis_composite_summary"]["ci95_low"]),
        "order_axis_ci95_high": float(records["FOLLOWUP_order_axis_composite_summary"]["ci95_high"]),
        "tail_axis_mean": value(records, "FOLLOWUP_tail_axis_composite_summary"),
        "difference_of_differences_mean": value(records, "FOLLOWUP_difference_of_differences_composite_summary"),
        "gate_rows_below_0_999": int(value(records, "GATE_BELOW_0999")),
        "gate_box_eligible_rows": int(value(records, "GATE_BOX_ELIGIBLE")),
    }
    result["gate_below_fraction"] = result["gate_rows_below_0_999"] / result["gate_box_eligible_rows"]
    return result


def main() -> None:
    result = replay()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
