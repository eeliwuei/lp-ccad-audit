#!/usr/bin/env python3
"""Replay the aggregate analytical results reported in the PLOS candidate.

The replay has two parts.

1. Stored records. The keys listed in expected_outputs/canonical_results.json
   are read from the stored records in data/S1_Data.csv, as in earlier
   releases.

2. Recomputation. The following values are recomputed from endpoints and
   compared with the stored values in data/S1_Data.csv:

   - H2 validation: the per-seed differences C4-M minus C4MixFT-M at seeds
     42, 1337, and 20260703, the two-seed protocol mean, and the three-seed
     mean, from the five-decimal arm endpoints in data/S2_Data.csv;
   - evaluation routes: the per-seed differences and the two-seed means on
     the 2,019-image route (full-precision endpoints in data/S2_Data.csv) and
     on the 2,553-image route (six-decimal arm endpoints in data/S1_Data.csv,
     because S2 Data has no per-seed endpoints for that route), and the shift
     between the two route means;
   - 12-run composite follow-up: the four cell means and sample SDs, the
     per-seed order-axis, tail-axis, and difference-of-differences contrasts,
     their means over seeds, and the descriptive 95% paired-t intervals, from
     the run endpoints in data/S2_Data.csv.

   A recomputed value passes when it differs from the stored value by at most
   half a unit in the last digit printed in the manuscript. The script exits
   with status 1 and prints a RECOMPUTATION MISMATCH message if any value
   fails, or if an endpoint is missing or duplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "S1_Data.csv"
S2_DATA = ROOT / "data" / "S2_Data.csv"
CONFIG = ROOT / "configs" / "replay_config.json"
OUTPUT = ROOT / "expected_outputs" / "replayed_results.generated.json"

TOLERANCE_RULE = "abs(recomputed - stored) <= 0.5 * 10**-decimals, decimals as printed in the manuscript"
FLOAT_SLACK = 1e-12

H2_CONTEXT = "historical_validation"
ROUTE_2019_CONTEXT = "source_excluded_evaluation"
FOLLOWUP_CONTEXT = "corrected_pipeline_validation_followup"


class RecomputationInputError(ValueError):
    """An endpoint needed for the recomputation is missing or duplicated."""


class RecomputationMismatch(AssertionError):
    """A recomputed value disagrees with its stored S1 value."""


def load_records(path: Path = DATA) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def load_endpoints(path: Path = S2_DATA) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(records: dict[str, dict[str, str]], record_id: str) -> float:
    return float(records[record_id]["value"])


def stored_text(records: dict[str, dict[str, str]], record_id: str, field: str) -> str:
    """Return a stored S1 value as text; field may be 'value', 'ci95_low',
    'ci95_high', or 'sample_SD' (parsed from the interpretation field)."""

    if record_id not in records:
        raise RecomputationInputError(f"S1 record {record_id!r} is missing")
    row = records[record_id]
    if field == "sample_SD":
        match = re.search(r"sample_SD=([-+0-9.eE]+)", row["interpretation"])
        if match is None:
            raise RecomputationInputError(f"S1 record {record_id!r} has no sample_SD")
        return match.group(1)
    text = row[field]
    if text == "":
        raise RecomputationInputError(f"S1 record {record_id!r} has an empty {field!r} field")
    return text


def s2_endpoint(rows: list[dict[str, str]], context: str, arm: str,
                seed: int | None = None, estimand: str | None = None) -> float:
    matches = [
        row for row in rows
        if row["context"] == context
        and row["arm"] == arm
        and (seed is None or row["seed"] == str(seed))
        and (estimand is None or row["estimand"] == estimand)
    ]
    if len(matches) != 1:
        raise RecomputationInputError(
            f"expected one S2 endpoint for context={context!r} arm={arm!r} seed={seed!r} "
            f"estimand={estimand!r}; found {len(matches)}"
        )
    return float(matches[0]["mAP50_95"])


def exact_mean(values) -> float:
    """Arithmetic mean computed exactly and rounded once to a float."""

    values = [Fraction(v) for v in values]
    if not values:
        raise RecomputationInputError("mean of an empty set of endpoints")
    return float(sum(values) / len(values))


def sample_sd(values) -> float:
    """Sample standard deviation (n - 1 denominator). The variance is computed
    exactly and rounded once before the square root, so the result does not
    depend on the Python version."""

    values = [Fraction(v) for v in values]
    if len(values) < 2:
        raise RecomputationInputError("sample SD needs at least two endpoints")
    centre = sum(values) / len(values)
    variance = sum((v - centre) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(float(variance))


def paired_t_interval(values: list[float], t_critical: float) -> dict[str, float]:
    centre = exact_mean(values)
    sd = sample_sd(values)
    half_width = t_critical * sd / math.sqrt(len(values))
    return {
        "mean": centre,
        "sample_sd": sd,
        "ci95_low": centre - half_width,
        "ci95_high": centre + half_width,
    }


def check_t_critical(t_critical: float) -> None:
    """The 0.975 quantile of t with 2 df has the closed form
    (2p - 1) / sqrt(2p(1 - p)) = 4.3026527297...; the configured value must
    match it to within 1e-9."""

    p = 0.975
    closed_form = (2 * p - 1) / math.sqrt(2 * p * (1 - p))
    if abs(t_critical - closed_form) > 1e-9:
        raise RecomputationInputError(
            f"configured t critical value {t_critical} is not t(0.975, df=2) = {closed_form}"
        )


def recompute(records: dict[str, dict[str, str]], endpoints: list[dict[str, str]],
              config: dict) -> tuple[dict, list[tuple[str, str, float, str, str, int]]]:
    """Return the recomputed values and the list of checks
    (group, item, recomputed, S1 record_id, S1 field, printed decimals)."""

    spec = config["recomputation"]
    arm_a, arm_b = spec["contrast_arms"]
    checks: list[tuple[str, str, float, str, str, int]] = []

    # H2 validation, five-decimal arm endpoints in S2 Data.
    protocol_seeds = [int(records[rid]["seed"]) for rid in config["validation_protocol_records"]]
    all_seeds = [int(records[rid]["seed"]) for rid in config["validation_all_records"]]
    h2_by_seed: dict[str, float] = {}
    for seed in all_seeds:
        a = s2_endpoint(endpoints, H2_CONTEXT, arm_a, seed)
        b = s2_endpoint(endpoints, H2_CONTEXT, arm_b, seed)
        h2_by_seed[str(seed)] = a - b
        checks.append(("endpoint_crosscheck", f"S2 validation endpoint {arm_a} seed {seed}", a,
                       f"VAL_{seed}_{arm_a}", "value", 5))
        checks.append(("endpoint_crosscheck", f"S2 validation endpoint {arm_b} seed {seed}", b,
                       f"VAL_{seed}_{arm_b}", "value", 5))
        checks.append(("h2_validation", f"difference seed {seed}", a - b, f"VAL_H2_{seed}", "value", 5))
    h2_protocol_mean = exact_mean(h2_by_seed[str(seed)] for seed in protocol_seeds)
    h2_three_seed_mean = exact_mean(h2_by_seed[str(seed)] for seed in all_seeds)
    checks.append(("h2_validation", "protocol mean (seeds " + ", ".join(map(str, protocol_seeds)) + ")",
                   h2_protocol_mean, "VAL_H2_MEAN_PROTOCOL", "value", 5))
    checks.append(("h2_validation", "three-seed mean", h2_three_seed_mean, "VAL_H2_MEAN_THREE", "value", 5))

    # Evaluation routes.
    route_seeds = spec["evaluation_route_seeds"]
    route_2553: dict[str, float] = {}
    route_2019: dict[str, float] = {}
    for seed in route_seeds:
        a = value(records, f"original_full_evaluation_{seed}_{arm_a}")
        b = value(records, f"original_full_evaluation_{seed}_{arm_b}")
        route_2553[str(seed)] = a - b
        checks.append(("evaluation_route_2553", f"difference seed {seed}", a - b,
                       f"original_full_evaluation_H2_{seed}", "value", 6))
    for seed in route_seeds:
        estimand = spec["route_2019_estimand"]
        a = s2_endpoint(endpoints, ROUTE_2019_CONTEXT, f"{arm_a}@{seed}", None, estimand)
        b = s2_endpoint(endpoints, ROUTE_2019_CONTEXT, f"{arm_b}@{seed}", None, estimand)
        route_2019[str(seed)] = a - b
        checks.append(("endpoint_crosscheck", f"S2 2,019-image endpoint {arm_a} seed {seed}", a,
                       f"source_excluded_evaluation_{seed}_{arm_a}", "value", 6))
        checks.append(("endpoint_crosscheck", f"S2 2,019-image endpoint {arm_b} seed {seed}", b,
                       f"source_excluded_evaluation_{seed}_{arm_b}", "value", 6))
        checks.append(("evaluation_route_2019", f"difference seed {seed}", a - b,
                       f"source_excluded_evaluation_H2_{seed}", "value", 6))
    route_2553_mean = exact_mean(route_2553.values())
    route_2019_mean = exact_mean(route_2019.values())
    route_shift = route_2019_mean - route_2553_mean
    checks.append(("evaluation_route_2553", "two-seed mean", route_2553_mean,
                   "original_full_evaluation_H2_mean", "value", 6))
    checks.append(("evaluation_route_2019", "two-seed mean", route_2019_mean,
                   "source_excluded_evaluation_H2_mean", "value", 6))
    checks.append(("evaluation_routes", "mean shift (2,019 minus 2,553)", route_shift,
                   "SOURCE_EXCLUSION_H2_CHANGE", "value", 6))

    # 12-run composite follow-up, run endpoints in S2 Data.
    t_critical = float(spec["paired_t_critical_0975_df2"])
    check_t_critical(t_critical)
    followup_seeds = spec["followup_seeds"]
    cells: dict[str, dict] = {}
    for cell in spec["followup_cells"]:
        by_seed = {str(seed): s2_endpoint(endpoints, FOLLOWUP_CONTEXT, cell, seed) for seed in followup_seeds}
        followup_rows = [row for row in endpoints if row["context"] == FOLLOWUP_CONTEXT and row["arm"] == cell]
        if len(followup_rows) != len(followup_seeds):
            raise RecomputationInputError(
                f"follow-up cell {cell!r} has {len(followup_rows)} S2 rows; expected {len(followup_seeds)}"
            )
        cell_values = [by_seed[str(seed)] for seed in followup_seeds]
        cells[cell] = {"endpoint_by_seed": by_seed, "mean": exact_mean(cell_values),
                       "sample_sd": sample_sd(cell_values)}
        checks.append(("followup_cell", f"{cell} mean", cells[cell]["mean"], f"FOLLOWUP_CELL_{cell}", "value", 6))
        checks.append(("followup_cell", f"{cell} sample SD", cells[cell]["sample_sd"],
                       f"FOLLOWUP_CELL_{cell}", "sample_SD", 6))

    def run(cell: str, seed: int) -> float:
        return cells[cell]["endpoint_by_seed"][str(seed)]

    contrast_functions = {
        "order_axis": lambda s: (exact_mean([run("mono_single", s), run("mono_mixed", s)])
                                 - exact_mean([run("shuf_single", s), run("shuf_mixed", s)])),
        "tail_axis": lambda s: (exact_mean([run("mono_single", s), run("shuf_single", s)])
                                - exact_mean([run("mono_mixed", s), run("shuf_mixed", s)])),
        "difference_of_differences": lambda s: ((run("mono_single", s) - run("mono_mixed", s))
                                                - (run("shuf_single", s) - run("shuf_mixed", s))),
    }
    contrasts: dict[str, dict] = {}
    for name, function in contrast_functions.items():
        by_seed = {str(seed): function(seed) for seed in followup_seeds}
        summary = paired_t_interval([by_seed[str(seed)] for seed in followup_seeds], t_critical)
        contrasts[name] = {"by_seed": by_seed, **summary}
        prefix = f"FOLLOWUP_{name}_composite"
        for seed in followup_seeds:
            checks.append(("followup_contrast", f"{name} seed {seed}", by_seed[str(seed)],
                           f"{prefix}_{seed}", "value", 5))
        checks.append(("followup_contrast", f"{name} mean", summary["mean"], f"{prefix}_summary", "value", 5))
        checks.append(("followup_contrast", f"{name} ci95 low", summary["ci95_low"],
                       f"{prefix}_summary", "ci95_low", 5))
        checks.append(("followup_contrast", f"{name} ci95 high", summary["ci95_high"],
                       f"{prefix}_summary", "ci95_high", 5))

    recomputed = {
        "h2_validation": {
            "source": "data/S2_Data.csv, context historical_validation (five-decimal arm endpoints)",
            "difference_by_seed": h2_by_seed,
            "protocol_seeds": protocol_seeds,
            "protocol_mean": h2_protocol_mean,
            "three_seed_mean": h2_three_seed_mean,
        },
        "evaluation_routes": {
            "route_2553": {
                "source": "data/S1_Data.csv arm_endpoint rows (six decimals); S2 Data has no per-seed endpoints "
                          "for this route",
                "difference_by_seed": route_2553,
                "mean": route_2553_mean,
            },
            "route_2019": {
                "source": f"data/S2_Data.csv, context {ROUTE_2019_CONTEXT}, estimand {spec['route_2019_estimand']}",
                "difference_by_seed": route_2019,
                "mean": route_2019_mean,
            },
            "mean_shift_2019_minus_2553": route_shift,
        },
        "followup": {
            "source": f"data/S2_Data.csv, context {FOLLOWUP_CONTEXT}",
            "seeds": followup_seeds,
            "paired_t_critical_0975_df2": t_critical,
            "cells": cells,
            "contrasts": contrasts,
        },
    }
    return recomputed, checks


def compare(records: dict[str, dict[str, str]],
            checks: list[tuple[str, str, float, str, str, int]]) -> list[dict]:
    rows = []
    for group, item, recomputed, record_id, field, decimals in checks:
        text = stored_text(records, record_id, field)
        stored = float(text)
        difference = abs(recomputed - stored)
        tolerance = 0.5 * 10 ** -decimals
        rows.append({
            "group": group,
            "item": item,
            "s1_record_id": record_id,
            "s1_field": field,
            "stored_text": text,
            "stored_value": stored,
            "recomputed_value": recomputed,
            "recomputed_printed": f"{recomputed:+.{decimals}f}",
            "printed_decimals": decimals,
            "abs_difference": difference,
            "tolerance": tolerance,
            "pass": difference <= tolerance + FLOAT_SLACK,
        })
    return rows


def replay(s1_path: Path = DATA, s2_path: Path = S2_DATA, config_path: Path = CONFIG) -> dict:
    records = load_records(s1_path)
    endpoints = load_endpoints(s2_path)
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)

    # Values read from the stored S1 records (keys checked by tests/test_replay.py).
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

    # Values recomputed from endpoints and compared with the stored S1 records.
    recomputed, checks = recompute(records, endpoints, config)
    comparison = compare(records, checks)
    failures = [row for row in comparison if not row["pass"]]
    result["recomputed"] = recomputed
    result["recomputation_checks"] = comparison
    result["recomputation_summary"] = {
        "rule": TOLERANCE_RULE,
        "n_checks": len(comparison),
        "n_pass": len(comparison) - len(failures),
        "n_fail": len(failures),
        "all_pass": not failures,
    }
    return result


def verify(result: dict) -> None:
    failures = [row for row in result["recomputation_checks"] if not row["pass"]]
    if failures:
        lines = [
            f"RECOMPUTATION MISMATCH: {len(failures)} of {len(result['recomputation_checks'])} recomputed values "
            "disagree with the stored S1 values beyond printed precision.",
        ]
        for row in failures:
            lines.append(
                f"  {row['group']} / {row['item']}: recomputed {row['recomputed_value']:+.12f}, "
                f"stored {row['s1_record_id']}.{row['s1_field']} = {row['stored_text']}, "
                f"|difference| {row['abs_difference']:.3e} > tolerance {row['tolerance']:.1e}"
            )
        raise RecomputationMismatch("\n".join(lines))


def format_table(rows: list[dict]) -> str:
    header = f"{'group':<22} {'item':<48} {'stored (S1)':>16} {'recomputed':>16} {'|diff|':>9} {'tol':>7}  result"
    lines = ["Recomputed versus stored S1 values (" + TOLERANCE_RULE + ")", header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['group']:<22} {row['item']:<48} {row['stored_text']:>16} "
            f"{row['recomputed_value']:>+16.10f} {row['abs_difference']:>9.2e} {row['tolerance']:>7.0e}  "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s1-data", type=Path, default=DATA)
    parser.add_argument("--s2-data", type=Path, default=S2_DATA)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    try:
        result = replay(args.s1_data, args.s2_data, args.config)
    except RecomputationInputError as error:
        print(f"RECOMPUTATION MISMATCH: input error: {error}", file=sys.stderr)
        return 1

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print(format_table(result["recomputation_checks"]))
    try:
        verify(result)
    except RecomputationMismatch as error:
        print(str(error), file=sys.stderr)
        return 1
    summary = result["recomputation_summary"]
    print(f"All {summary['n_checks']} recomputed values agree with the stored S1 values at printed precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
