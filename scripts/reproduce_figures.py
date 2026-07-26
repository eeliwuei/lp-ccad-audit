#!/usr/bin/env python3
"""Regenerate the figures that the released CSVs actually support.

    python3 scripts/reproduce_figures.py [--outdir figures]

Generated here (real data, no fabrication):
  1. fig_factorial_order_effect   paper Fig. 5B -- the three seed-blocked
     ORDER contrasts, their mean with the 95% paired-t interval, and the
     +/- SESOI band. Source: results/factorial_runs.csv.
  2. fig_locked_test_contrasts    the test-split half of paper Fig. 5A --
     per-seed locked-test deltas for the five pre-registered contrasts with
     their source-clustered bootstrap 95% intervals. Sources:
     results/locked_test_contrasts.csv, results/bootstrap_summary.csv.
  3. fig_locked_test_arms         per-arm, per-seed locked-test primaries;
     the test-split counterpart of paper Fig. 3 (which is drawn on the
     validation split).

NOT generated here, deliberately:
  * Fig. 1 (dataflow) and Fig. 2 (label-space evolution) are schematics drawn
    from the schedule definition, not from measurements.
  * Fig. 3 as printed is the nine-arm VALIDATION panel; the per-arm validation
    primaries of the nine-arm family are outside this minimal release (it ships
    the factorial cells and the C0-R baseline). Panel 3 above is the test-split
    counterpart, not a substitute.
  * Fig. 4 (epoch 30/60/90/120/150 trajectories) needs the intermediate
    checkpoints and a GPU re-evaluation pass.
  * Fig. 6 (dispersion) needs the per-epoch validation traces.
  None of these are approximated from data that is not here.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SEEDS = [42, 1337, 20260703]
CELLS = ["mono_single", "mono_mixed", "shuf_single", "shuf_mixed"]
SESOI = 0.01
TCRIT95_DF2 = 4.3027

NOT_REPRODUCIBLE = [
    ("Fig. 1  LP-CCAD dataflow", "schematic; no measured data"),
    ("Fig. 2  label-space evolution", "schematic drawn from configs/frozen_protocol/"),
    ("Fig. 3  nine-arm validation panel", "per-arm validation primaries are not in this release"),
    ("Fig. 4  replicate trajectories", "needs intermediate checkpoints + GPU re-evaluation"),
    ("Fig. 5A validation half", "validation-side H2 points are not in this release"),
    ("Fig. 6  dispersion panels", "needs per-epoch validation traces"),
]


def read_csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        sys.exit(f"missing input: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def order_effect_per_seed() -> list[float]:
    runs = {(r["cell"], int(r["seed"])): float(r["mAP"]) for r in read_csv("factorial_runs.csv")}
    out = []
    for s in SEEDS:
        ms, mm = runs[("mono_single", s)], runs[("mono_mixed", s)]
        ss, sm = runs[("shuf_single", s)], runs[("shuf_mixed", s)]
        out.append(((ms + mm) - (ss + sm)) / 2.0)
    return out


def locked_test_deltas() -> dict[str, dict[int, float]]:
    rows = read_csv("locked_test_contrasts.csv")
    val = {(r["arm"], int(r["seed"])): float(r["mAP50_95"]) for r in rows}
    contrasts = {"H1": ("C4-M", "C1-M"), "H2": ("C4-M", "C4MixFT-M"),
                 "H3": ("C4MixFT-M", "C4Mix-M"), "H4": ("C4-M", "C4R-M"),
                 "H5": ("C1-M", "C0-R")}
    out: dict[str, dict[int, float]] = {}
    for h, (a, b) in contrasts.items():
        per_seed = {}
        for s in SEEDS:
            if (a, s) in val and (b, s) in val:
                per_seed[s] = val[(a, s)] - val[(b, s)]
        out[h] = per_seed
    return out


def bootstrap_means() -> dict[str, tuple[float, float]]:
    return {r["contrast"]: (float(r["ci95_lo"]), float(r["ci95_hi"]))
            for r in read_csv("bootstrap_summary.csv") if r["scope"] == "mean"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "figures"))
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    effects = order_effect_per_seed()
    mean = st.mean(effects)
    hw = TCRIT95_DF2 * st.stdev(effects) / math.sqrt(len(effects))
    deltas = locked_test_deltas()
    boot = bootstrap_means()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; no figures were written.")
        print("  pip install matplotlib  then re-run, or read the values below.\n")
        print("Fig. 5B  order main effect (val mAP50-95, seed-blocked)")
        for s, e in zip(SEEDS, effects):
            print(f"  seed {s:<10} {e:+.5f}")
        print(f"  mean {mean:+.5f}   95% paired-t CI [{mean - hw:+.5f}, {mean + hw:+.5f}]"
              f"   SESOI band +/-{SESOI}")
        print("\nlocked-test per-seed deltas and bootstrap mean 95% CIs")
        for h in sorted(deltas):
            per = "  ".join(f"seed {s}: {d:+.5f}" for s, d in deltas[h].items())
            lo, hi = boot[h]
            print(f"  {h}  {per}   boot mean CI [{lo:+.5f}, {hi:+.5f}]")
        print()
        for name, why in NOT_REPRODUCIBLE:
            print(f"  not regenerable here: {name:<34} ({why})")
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # ---- 1. paper Fig. 5B -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhspan(-SESOI, SESOI, color="0.88", label=f"SESOI band $\\pm${SESOI}")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.plot(range(len(SEEDS)), effects, "o", color="#1f4e79", label="seed-blocked contrast")
    ax.errorbar([len(SEEDS)], [mean], yerr=[[hw], [hw]], fmt="s", color="#b03a2e",
                capsize=4, label="mean, 95% paired-$t$")
    ax.set_xticks(range(len(SEEDS) + 1))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS] + ["mean"], fontsize=8)
    ax.set_ylabel("ORDER main effect\n(val mAP$_{50-95}$)", fontsize=9)
    ax.set_title("Randomized factorial: ORDER main effect", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = outdir / f"fig_factorial_order_effect.{args.format}"
    fig.savefig(p); plt.close(fig); written.append(p)

    # ---- 2. locked-test contrasts + bootstrap ----------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    names = sorted(deltas)
    for i, h in enumerate(names):
        lo, hi = boot[h]
        ax.plot([lo, hi], [i, i], color="#b03a2e", lw=2.0,
                label="bootstrap 95% CI (mean)" if i == 0 else None)
        for s, d in deltas[h].items():
            ax.plot([d], [i], "o", ms=4, color="#1f4e79",
                    label="per-seed delta" if (i == 0 and s == SEEDS[0]) else None)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("locked-test delta (mAP$_{50-95}$)", fontsize=9)
    ax.set_title("Locked test: pre-registered contrasts", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = outdir / f"fig_locked_test_contrasts.{args.format}"
    fig.savefig(p); plt.close(fig); written.append(p)

    # ---- 3. per-arm locked-test primaries --------------------------------
    rows = read_csv("locked_test_contrasts.csv")
    arms: list[str] = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, arm in enumerate(arms):
        vals = [float(r["mAP50_95"]) for r in rows if r["arm"] == arm]
        ax.plot([i] * len(vals), vals, "o", ms=4, color="#1f4e79")
        ax.plot([i - 0.22, i + 0.22], [st.mean(vals)] * 2, color="#b03a2e", lw=1.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("locked-test mAP$_{50-95}$", fontsize=9)
    ax.set_title("Locked-test primaries per arm (each point = one seed)", fontsize=10)
    fig.tight_layout()
    p = outdir / f"fig_locked_test_arms.{args.format}"
    fig.savefig(p); plt.close(fig); written.append(p)

    for p in written:
        print(f"wrote {p}")
    print()
    for name, why in NOT_REPRODUCIBLE:
        print(f"not regenerable from this release: {name:<34} ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
