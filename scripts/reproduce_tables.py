#!/usr/bin/env python3
"""Reproduce the paper's factorial tables and the locked-test bootstrap summary
from the released CSVs.

Stdlib only (csv / math / statistics): no numpy, no scipy, no torch. Run from
anywhere:

    python3 scripts/reproduce_tables.py

Reads
    results/factorial_runs.csv      the 12 raw factorial runs (real, frozen)
    results/c0r_val_baseline.csv    the no-KD baseline on the same val split
    results/bootstrap_summary.csv   source-clustered bootstrap CIs (B=1000)
    results/paper_metrics.csv       the manuscript's headline values, for a
                                    live cross-check (never used as an input
                                    to the recomputation)

Prints
    Table 10  main effects + interaction, seed-blocked, paired-t CI, TOST
    Table 11  the twelve raw runs, C0-R row, paired diffs, seed-blocked contrasts
    Bootstrap the five locked-test contrasts, mean and per-seed 95% CIs

Everything in Table 10 is recomputed live from Table 11's inputs; the
manuscript values are only compared against, so a mismatch is visible.
"""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

SEEDS = [42, 1337, 20260703]
CELLS = ["mono_single", "mono_mixed", "shuf_single", "shuf_mixed"]
CELL_LABEL = {"mono_single": "mono x single", "mono_mixed": "mono x mixed",
              "shuf_single": "shuf x single", "shuf_mixed": "shuf x mixed"}
EFFECTS = ["order", "tail", "interaction"]
SESOI = 0.01
# Student-t critical values at df = 2 (three seed-blocked contrasts).
TCRIT = {0.95: 4.3027, 0.90: 2.9200}


# ----------------------------------------------------------------- utilities
def t_cdf_df2(t: float) -> float:
    """Exact Student-t CDF for df = 2: F(t) = 0.5 * (1 + t / sqrt(2 + t^2)).

    Closed form, so no scipy is needed for the TOST p-values.
    """
    return 0.5 * (1.0 + t / math.sqrt(2.0 + t * t))


def holm(pvals: dict[str, float], enforce_monotone: bool = True) -> dict[str, float]:
    """Holm step-down adjustment.

    With ``enforce_monotone`` (the correct, statsmodels-compatible behaviour)
    the adjusted sequence is forced non-decreasing in the raw ordering. The
    unenforced variant (plain ``p * (m - i)``) is exposed only so the script
    can print both when they differ.
    """
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        raw = min(1.0, p * (m - i))
        running = max(running, raw)
        out[k] = running if enforce_monotone else raw
    return out


def read_csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        sys.exit(f"missing input: {path} (run from a complete checkout)")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(x: float, nd: int = 5, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


def rule(char: str = "-", n: int = 78) -> str:
    return char * n


# --------------------------------------------------------------- computation
def load_runs() -> dict[tuple[str, int], dict]:
    runs = {}
    for r in read_csv("factorial_runs.csv"):
        runs[(r["cell"], int(r["seed"]))] = {
            "mAP": float(r["mAP"]), "knife": float(r["knife_AP"]),
            "gun": float(r["gun_AP"]), "stick": float(r["stick_AP"]),
            "sha": r["checkpoint_sha256"],
        }
    missing = [(c, s) for c in CELLS for s in SEEDS if (c, s) not in runs]
    if missing:
        sys.exit(f"factorial_runs.csv is incomplete, missing cells: {missing}")
    return runs


def effects_within_seed(runs, seed: int) -> dict[str, float]:
    ms = runs[("mono_single", seed)]["mAP"]
    mm = runs[("mono_mixed", seed)]["mAP"]
    ss = runs[("shuf_single", seed)]["mAP"]
    sm = runs[("shuf_mixed", seed)]["mAP"]
    # order > 0  <=> monotone ordering beats shuffled, averaged over tail
    # tail  > 0  <=> single-class tail beats mixed tail, averaged over order
    # inter > 0  <=> the single-tail benefit is larger under monotone ordering
    return {"order": ((ms + mm) - (ss + sm)) / 2.0,
            "tail": ((ms + ss) - (mm + sm)) / 2.0,
            "interaction": (ms - mm) - (ss - sm)}


def effect_stats(vals: list[float]) -> dict:
    m, sd, n = st.mean(vals), st.stdev(vals), len(vals)
    se = sd / math.sqrt(n)
    ci95 = (m - TCRIT[0.95] * se, m + TCRIT[0.95] * se)
    ci90 = (m - TCRIT[0.90] * se, m + TCRIT[0.90] * se)
    # TOST against +/- SESOI, df = n - 1 = 2.
    p_lower = 1.0 - t_cdf_df2((m + SESOI) / se)   # H0: mu <= -SESOI
    p_upper = t_cdf_df2((m - SESOI) / se)         # H0: mu >= +SESOI
    tost_p = max(p_lower, p_upper)
    # two-sided difference-from-zero p (a DIFFERENT null; reported separately)
    diff_p = 2.0 * (1.0 - t_cdf_df2(abs(m) / se))
    return {"mean": m, "sd": sd, "se": se, "ci95": ci95, "ci90": ci90,
            "tost_p": tost_p, "diff_p": diff_p, "per_seed": vals}


# ------------------------------------------------------------------- tables
def table10(stats: dict, paper: dict) -> None:
    print(rule("="))
    print("TABLE 10  Seed-balanced 2x2 factorial: main effects and interaction")
    print("          primary estimand = last.pt validation mAP50-95, seed-blocked")
    print(f"          (mean of {len(SEEDS)} within-seed contrasts; paired-t CI, df=2; SESOI={SESOI})")
    print(rule("="))
    head = f"{'effect':<14}{'estimate':>10}{'95% CI (t)':>26}{'TOST p':>9}{'Holm TOST p':>13}"
    print(head)
    print(rule())
    hp = holm({k: stats[k]["tost_p"] for k in EFFECTS})
    for k in EFFECTS:
        s = stats[k]
        ci = f"[{fmt(s['ci95'][0], 5, True)}, {fmt(s['ci95'][1], 5, True)}]"
        print(f"{k:<14}{fmt(s['mean'], 5, True):>10}{ci:>26}{s['tost_p']:>9.3f}{hp[k]:>13.3f}")
    print(rule())
    print("equivalence verdicts (TOST vs the |beta| >= SESOI null):")
    for k in EFFECTS:
        s = stats[k]
        equiv = s["ci90"][0] > -SESOI and s["ci90"][1] < SESOI
        verdict = "equivalent to zero at the uncorrected 5% level" if equiv else "inconclusive"
        holm_note = "" if not equiv else ("; NOT equivalent after Holm" if hp[k] >= 0.05 else "")
        print(f"  {k:<12} 90% CI [{fmt(s['ci90'][0], 5, True)}, {fmt(s['ci90'][1], 5, True)}]"
              f"  ->  {verdict}{holm_note}")
    print("note: the two-sided difference-from-zero p is a different null and is")
    print("      never a substitute for the equivalence verdict:")
    dh = holm({k: stats[k]["diff_p"] for k in EFFECTS})
    dh_raw = holm({k: stats[k]["diff_p"] for k in EFFECTS}, enforce_monotone=False)
    for k in EFFECTS:
        extra = "" if abs(dh[k] - dh_raw[k]) < 5e-3 else f"   [unenforced p*(m-i) = {dh_raw[k]:.2f}]"
        print(f"  {k:<12} diff-from-zero p = {stats[k]['diff_p']:.3f}"
              f"   Holm-adjusted = {dh[k]:.2f}{extra}")
    print()
    print("cross-check against the manuscript (results/paper_metrics.csv):")
    ok = True
    for k in EFFECTS:
        want = float(paper[k]["estimate"])
        got = stats[k]["mean"]
        good = abs(want - got) < 1e-4
        ok &= good
        print(f"  {k:<12} recomputed {fmt(got, 5, True)}  paper {fmt(want, 5, True)}"
              f"  paper-macro {paper[k]['paper_estimate']:>8}   {'MATCH' if good else 'MISMATCH'}")
    print(f"  => {'all effects reproduce within 1e-4' if ok else 'MISMATCH: investigate before citing'}")
    print()


def table11(runs: dict, stats: dict) -> None:
    c0r = {int(r["seed"]): float(r["mAP"]) for r in read_csv("c0r_val_baseline.csv")}
    print(rule("="))
    print("TABLE 11  The twelve raw factorial runs (val mAP50-95, last.pt)")
    print(rule("="))
    print(f"{'cell':<16}" + "".join(f"{'seed ' + str(s):>14}" for s in SEEDS) + f"{'mean':>12}")
    print(rule())
    for c in CELLS:
        vals = [runs[(c, s)]["mAP"] for s in SEEDS]
        print(f"{CELL_LABEL[c]:<16}" + "".join(f"{v:>14.5f}" for v in vals) + f"{st.mean(vals):>12.5f}")
    print(rule())
    vals = [c0r[s] for s in SEEDS]
    print(f"{'C0-R (no KD)':<16}" + "".join(f"{v:>14.5f}" for v in vals) + f"{st.mean(vals):>12.5f}")
    print(rule())
    print("paired per-seed difference from C0-R")
    neg = tot = 0
    for c in CELLS:
        ds = [runs[(c, s)]["mAP"] - c0r[s] for s in SEEDS]
        neg += sum(1 for d in ds if d < 0)
        tot += len(ds)
        print(f"{CELL_LABEL[c]:<16}" + "".join(f"{d:>+14.5f}" for d in ds))
    print(f"  -> {neg} of {tot} paired differences are negative"
          " (KD does not beat the no-KD baseline)")
    print(rule())
    print("seed-blocked contrasts")
    for k in EFFECTS:
        s = stats[k]
        print(f"{k:<16}" + "".join(f"{v:>+14.5f}" for v in s["per_seed"])
              + f"{s['mean']:>+12.5f}")
    print(rule())
    print("per-class AP50-95 (cell means over the three seeds)")
    print(f"{'cell':<16}{'overall':>12}{'knife':>12}{'gun':>12}{'stick':>12}")
    for c in CELLS:
        cols = [st.mean([runs[(c, s)][k] for s in SEEDS]) for k in ("mAP", "knife", "gun", "stick")]
        print(f"{CELL_LABEL[c]:<16}" + "".join(f"{v:>12.4f}" for v in cols))
    print()
    print("checkpoint SHA-256 (primary last.pt of every run)")
    for c in CELLS:
        for s in SEEDS:
            print(f"  {CELL_LABEL[c]:<16} seed {s:<10} {runs[(c, s)]['sha']}")
    print()


def table_bootstrap() -> None:
    rows = read_csv("bootstrap_summary.csv")
    print(rule("="))
    print("LOCKED-TEST SOURCE-CLUSTERED BOOTSTRAP  (B=1000, seed 20260725, 2099 clusters)")
    print("paired resampling of source clusters; every arm scored on the SAME resample")
    print(rule("="))
    print(f"{'contrast':<10}{'definition':<26}{'scope':<10}{'seed':>10}{'95% CI':>26}")
    print(rule())
    for r in rows:
        ci = f"[{float(r['ci95_lo']):+.5f}, {float(r['ci95_hi']):+.5f}]"
        print(f"{r['contrast']:<10}{r['arm_a'] + ' - ' + r['arm_b']:<26}"
              f"{r['scope']:<10}{r['seed'] or '-':>10}{ci:>26}")
    print(rule())
    mean_rows = [r for r in rows if r["scope"] == "mean"]
    excl = [r["contrast"] for r in mean_rows
            if not (float(r["ci95_lo"]) <= 0.0 <= float(r["ci95_hi"]))]
    print("intervals excluding zero: " + (", ".join(excl) if excl else "none"))
    print("H1's interval touches zero at its upper end (-0.00004); the sign is")
    print("negative, i.e. the curriculum arm does NOT beat full-view KD on the")
    print("locked test. No contrast supports the validation-side ordering signal.")
    print()


def locked_test_arms() -> None:
    rows = read_csv("locked_test_contrasts.csv")
    print(rule("="))
    print("LOCKED-TEST PRIMARIES (single consumed pass; per-arm, per-seed)")
    print(rule("="))
    print(f"{'arm':<12}{'seed':>10}{'mAP50-95':>12}{'knife':>10}{'gun':>10}{'stick':>10}")
    print(rule())
    for r in rows:
        print(f"{r['arm']:<12}{r['seed']:>10}{float(r['mAP50_95']):>12.5f}"
              f"{float(r['knife']):>10.4f}{float(r['gun']):>10.4f}{float(r['stick']):>10.4f}")
    print()


def main() -> int:
    runs = load_runs()
    stats = {k: effect_stats([effects_within_seed(runs, s)[k] for s in SEEDS]) for k in EFFECTS}
    paper = {r["effect"]: r for r in read_csv("paper_metrics.csv")}

    print()
    print("LP-CCAD audit -- table reproduction from released CSVs")
    print("source: results/*.csv (frozen real runs); no data or GPU needed")
    print()
    table10(stats, paper)
    table11(runs, stats)
    locked_test_arms()
    table_bootstrap()
    print("done. See docs/EXPECTED_OUTPUTS.md for the values this should print.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
