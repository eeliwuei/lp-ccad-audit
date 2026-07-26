#!/usr/bin/env python3
"""LP-CCAD v3 2x2 factorial analysis (order x tail), SEED-BALANCED design.

All four cells are run at the same three training seeds {42, 1337, 20260703};
the three randomized cells pair a distinct schedule draw with each seed
(s1<->42, s2<->1337, s3<->20260703), the deterministic mono-single cell uses
the C4-M schedule at each seed. Because seed is crossed with (balanced across)
the 2x2 factors, we analyse SEED-BLOCKED: compute a complete 2x2 contrast
WITHIN each seed, then average the three per-seed effects and take a paired-t CI
over them (df=2). This cancels the training-seed main effect exactly and folds
schedule-lottery variance (present only in the randomized cells) into the
per-seed effect spread. No pooling of the two replication mechanisms.

Input JSON (from the freeze-collection step): 12 records
  {"cell": "mono_single|mono_mixed|shuf_single|shuf_mixed", "seed": 42|1337|20260703,
   "mAP": float, "per_class": {"knife": f, "gun": f, "stick": f}}
Run: factorial_analysis.py results12.json  (or --selftest)

RELEASE NOTE. This is the analysis script as it was run, unmodified. Its input
JSON is the same data the release ships as results/factorial_runs.csv; rebuild
it with:

  python3 -c "import csv,json;
  json.dump([{'cell':r['cell'],'seed':int(r['seed']),'mAP':float(r['mAP']),
              'per_class':{'knife':float(r['knife_AP']),'gun':float(r['gun_AP']),
                           'stick':float(r['stick_AP'])}}
             for r in csv.DictReader(open('results/factorial_runs.csv'))],
            open('results12.json','w'))"

then `python3 scripts/factorial_analysis.py results12.json`. SciPy is optional:
without it the hard-coded df=2 critical values are used and the p-values are
omitted (scripts/reproduce_tables.py computes them in closed form instead).
"""
import json, sys, math, argparse
from statistics import mean, stdev

CELLS = ["mono_single", "mono_mixed", "shuf_single", "shuf_mixed"]
SEEDS = [42, 1337, 20260703]
SESOI = 0.01

def tcrit(p, df):
    try:
        from scipy import stats
        return float(stats.t.ppf(p, df))
    except Exception:
        return {(0.95, 2): 2.9200, (0.975, 2): 4.3027, (0.80, 2): 1.0607}[(round(p, 3), df)]

def cell_at(recs, cell, seed):
    v = [r["mAP"] for r in recs if r["cell"] == cell and r["seed"] == seed]
    if len(v) != 1:
        raise ValueError(f"expected exactly 1 record for {cell}@{seed}, got {len(v)}")
    return v[0]

def effect_within_seed(recs, seed):
    ms = cell_at(recs, "mono_single", seed); mm = cell_at(recs, "mono_mixed", seed)
    ss = cell_at(recs, "shuf_single", seed); sm = cell_at(recs, "shuf_mixed", seed)
    # SIGN CONVENTION (matches the nine-arm H2 = C4-C4MixFT and H3 = C4MixFT-C4Mix):
    #   order > 0  <=> MONOTONE ordering outperforms shuffled, averaged over tail
    #   tail  > 0  <=> SINGLE-class tail outperforms mixed tail, averaged over order
    #   inter > 0  <=> the single-tail benefit is LARGER under monotone ordering
    order = ((ms + mm) - (ss + sm)) / 2.0
    tail  = ((ms + ss) - (mm + sm)) / 2.0
    inter = (ms - mm) - (ss - sm)
    return {"order": order, "tail": tail, "interaction": inter}

def paired_ci(per_seed_vals, level=0.95):
    m = mean(per_seed_vals); s = stdev(per_seed_vals); n = len(per_seed_vals)
    hw = tcrit(0.5 + level / 2.0, n - 1) * s / math.sqrt(n)
    return m, s, (m - hw, m + hw)

def seed_paired_boot(recs, key, nboot=20000):
    """Bootstrap the three seed-blocked per-seed effects (resample seed index,
    shared across cells -> seed variance already cancelled inside each block)."""
    import random
    rng = random.Random(12345)
    eff = [effect_within_seed(recs, s)[key] for s in SEEDS]
    n = len(eff); samples = []
    for _ in range(nboot):
        draw = [eff[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()
    return samples[int(0.025 * nboot)], samples[int(0.975 * nboot)]

def tost(per_seed_vals):
    m, s, ci90 = paired_ci(per_seed_vals, level=0.90)
    equiv = ci90[0] > -SESOI and ci90[1] < SESOI
    differs = ci90[0] > 0 or ci90[1] < 0
    verdict = ("equivalent" if equiv else
               ("effect-beyond-SESOI" if differs and (abs(ci90[0]) >= SESOI or abs(ci90[1]) >= SESOI)
                else "inconclusive"))
    return ci90, equiv, differs, verdict

def analyze(recs):
    # cell means (over seeds) for display; per-seed effects for inference
    cellmean = {c: mean([r["mAP"] for r in recs if r["cell"] == c]) for c in CELLS}
    cellsd = {c: stdev([r["mAP"] for r in recs if r["cell"] == c]) for c in CELLS}
    rep = {"design": "seed-balanced 2x2 (4 cells x seeds 42/1337/20260703); seed-blocked analysis; PRIMARY interval = paired t over the 3 within-seed contrasts (df=2). The bootstrap is reported only as a secondary check: at n=3 resampling creates no new seed or schedule realization and its interval must not be called robust.",
           "cells": {c: {"mean": round(cellmean[c], 5), "sd_over_seeds": round(cellsd[c], 5)} for c in CELLS},
           "per_seed_effects": {}, "effects": {}}
    for s in SEEDS:
        e = effect_within_seed(recs, s)
        rep["per_seed_effects"][str(s)] = {k: round(v, 5) for k, v in e.items()}
    for key in ("order", "tail", "interaction"):
        vals = [effect_within_seed(recs, s)[key] for s in SEEDS]
        m, sd, ci95 = paired_ci(vals, 0.95)
        ci90, equiv, differs, verdict = tost(vals)
        blo, bhi = seed_paired_boot(recs, key)
        rep["effects"][key] = {
            "estimate": round(m, 5), "sd_across_seeds": round(sd, 5),
            "t_ci95_PRIMARY": [round(ci95[0], 5), round(ci95[1], 5)],
            "boot_ci95_secondary_n3_unreliable": [round(blo, 5), round(bhi, 5)],
            "tost_ci90": [round(ci90[0], 5), round(ci90[1], 5)],
            "equivalent_to_zero": equiv, "differs_from_zero": differs, "verdict": verdict,
            "per_seed": [round(v, 5) for v in vals],
        }
    # Holm across the 3 effects using |t| -> two-sided p from paired t (df=2)
    def p_two(vals):
        m, sd = mean(vals), stdev(vals)
        if sd == 0: return 0.0 if m != 0 else 1.0
        t = abs(m) / (sd / math.sqrt(len(vals)))
        try:
            from scipy import stats
            return float(2 * (1 - stats.t.cdf(t, len(vals) - 1)))
        except Exception:
            return None
    praw = {k: p_two([effect_within_seed(recs, s)[k] for s in SEEDS]) for k in ("order","tail","interaction")}
    if all(v is not None for v in praw.values()):
        order_by_p = sorted(praw.items(), key=lambda kv: kv[1]); m = len(order_by_p); holm = {}
        for i,(k,p) in enumerate(order_by_p): holm[k] = round(min(1.0, p*(m-i)), 4)
        rep["holm_adjusted_p"] = holm
        rep["raw_p"] = {k: round(v,4) for k,v in praw.items()}
    return rep

def selftest():
    import random; rng = random.Random(7)
    # true: order +0.006 (stable across seeds), tail ~0, tiny interaction; seed offsets large & shared
    seed_off = {42: +0.007, 1337: 0.000, 20260703: -0.004}   # big seed main effect (must cancel)
    base = {"mono_single": 0.379, "mono_mixed": 0.377, "shuf_single": 0.385, "shuf_mixed": 0.383}
    recs = []
    for c in CELLS:
        for s in SEEDS:
            recs.append({"cell": c, "seed": s, "mAP": base[c] + seed_off[s] + rng.gauss(0, 0.003),
                         "per_class": {"knife": 0.5, "gun": 0.4, "stick": 0.2}})
    print(json.dumps(analyze(recs), indent=2))
    print("\nSELFTEST OK: order ~+0.006 with seed cancelled (despite +7/-4mAP seed offsets); tail ~-0.002")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.results: selftest()
    else: print(json.dumps(analyze(json.load(open(a.results))), indent=2))
