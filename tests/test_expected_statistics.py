"""The three seed-blocked contrasts must be recomputable from the raw runs.

results/factorial_runs.csv holds the twelve real runs; results/paper_metrics.csv
holds the manuscript's headline values. This test recomputes the contrasts from
the raw runs and requires them to agree with the manuscript within 1e-4, so a
silent edit to either file fails CI. Stdlib only.
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results/factorial_runs.csv"
METRICS = REPO / "results/paper_metrics.csv"
BASELINE = REPO / "results/c0r_val_baseline.csv"

SEEDS = [42, 1337, 20260703]
CELLS = ["mono_single", "mono_mixed", "shuf_single", "shuf_mixed"]
TOL = 1e-4
# The published values (numbers.tex macros FOrdEst / FTailEst / FIntEst,
# at the precision the release CSV carries).
PUBLISHED = {"order": -0.00104, "tail": +0.00567, "interaction": -0.00066}
TCRIT95_DF2 = 4.3027


def load_runs() -> dict[tuple[str, int], float]:
    with RUNS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    runs = {(r["cell"], int(r["seed"])): float(r["mAP"]) for r in rows}
    assert len(rows) == 12, f"expected 12 factorial runs, found {len(rows)}"
    assert len(runs) == 12, "duplicate (cell, seed) rows in factorial_runs.csv"
    return runs


def contrasts_within_seed(runs, seed: int) -> dict[str, float]:
    ms = runs[("mono_single", seed)]
    mm = runs[("mono_mixed", seed)]
    ss = runs[("shuf_single", seed)]
    sm = runs[("shuf_mixed", seed)]
    return {"order": ((ms + mm) - (ss + sm)) / 2.0,
            "tail": ((ms + ss) - (mm + sm)) / 2.0,
            "interaction": (ms - mm) - (ss - sm)}


def load_metrics() -> dict[str, dict]:
    with METRICS.open(newline="") as f:
        return {r["effect"]: r for r in csv.DictReader(f)}


def test_every_cell_seed_combination_is_present():
    runs = load_runs()
    for cell in CELLS:
        for seed in SEEDS:
            assert (cell, seed) in runs, f"missing run {cell}@{seed}"


def test_seed_blocked_contrasts_match_paper_metrics():
    runs = load_runs()
    metrics = load_metrics()
    for effect, published in PUBLISHED.items():
        vals = [contrasts_within_seed(runs, s)[effect] for s in SEEDS]
        recomputed = st.mean(vals)
        from_csv = float(metrics[effect]["estimate"])
        assert abs(recomputed - from_csv) < TOL, (
            f"{effect}: recomputed {recomputed:.6f} vs paper_metrics.csv {from_csv:.6f}")
        assert abs(recomputed - published) < TOL, (
            f"{effect}: recomputed {recomputed:.6f} vs published {published:.6f}")


def test_paired_t_intervals_match_paper_metrics():
    runs = load_runs()
    metrics = load_metrics()
    for effect in PUBLISHED:
        vals = [contrasts_within_seed(runs, s)[effect] for s in SEEDS]
        m, sd = st.mean(vals), st.stdev(vals)
        hw = TCRIT95_DF2 * sd / math.sqrt(len(vals))
        assert abs((m - hw) - float(metrics[effect]["t_ci95_lo"])) < TOL, effect
        assert abs((m + hw) - float(metrics[effect]["t_ci95_hi"])) < TOL, effect


def test_every_ci_contains_zero_so_no_effect_is_established():
    """The paper's conclusion: the randomized follow-up establishes no effect.
    If a future edit produced a CI excluding zero, that conclusion would have
    changed and must not slip through silently."""
    metrics = load_metrics()
    for effect in PUBLISHED:
        lo = float(metrics[effect]["t_ci95_lo"])
        hi = float(metrics[effect]["t_ci95_hi"])
        assert lo < 0.0 < hi, f"{effect}: 95% CI [{lo}, {hi}] no longer contains zero"


def test_kd_arms_do_not_beat_the_no_kd_baseline():
    """11 of the 12 paired per-seed differences from C0-R are negative."""
    runs = load_runs()
    with BASELINE.open(newline="") as f:
        c0r = {int(r["seed"]): float(r["mAP"]) for r in csv.DictReader(f)}
    diffs = [runs[(c, s)] - c0r[s] for c in CELLS for s in SEEDS]
    assert len(diffs) == 12
    assert sum(1 for d in diffs if d < 0) == 11, [round(d, 5) for d in diffs]
    positives = [(c, s) for c in CELLS for s in SEEDS if runs[(c, s)] - c0r[s] > 0]
    assert positives == [("mono_single", 42)], positives
