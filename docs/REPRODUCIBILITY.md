# What can be reproduced, and what cannot

This release is deliberately split into two tiers. Be suspicious of any claim
that the second tier is "one command away" — it is not, and pretending
otherwise is the failure mode this paper is about.

## Tier 1 — reproducible here, from the CSVs alone

No dataset, no checkpoint, no GPU, no network. Python 3.9+ standard library.

| what | command | needs |
|---|---|---|
| Table 10 (factorial effects, paired-$t$ CIs, TOST, Holm) | `python3 scripts/reproduce_tables.py` | stdlib only |
| Table 11 (twelve raw runs, C0-R row, paired diffs, seed-blocked contrasts) | same command | stdlib only |
| Locked-test primaries and the source-clustered bootstrap intervals | same command | stdlib only |
| Schedule dose-match invariant (all nine randomized arms vs C4-M) | `python3 -m pytest tests/test_schedule_multiset.py -q` | pytest (PyYAML optional) |
| The published contrasts recomputed from the raw runs | `python3 -m pytest tests/test_expected_statistics.py -q` | pytest |
| Why the bootstrap must renumber cloned ids | `python3 -m pytest tests/test_bootstrap_duplicate_ids.py -q` | pytest + faster-coco-eval (skips otherwise) |
| The projection loss on synthetic data | `python3 examples/synthetic_minimal_example.py` | torch |
| Figures 5B, locked-test contrasts, per-arm locked-test panel | `python3 scripts/reproduce_figures.py` | matplotlib (prints the numbers instead if absent) |
| Human-readable schedule ledger | `python3 scripts/verify_schedule_exposure.py` | PyYAML |

Every number printed by `reproduce_tables.py` is recomputed live from
`results/factorial_runs.csv` and `results/c0r_val_baseline.csv`; the
manuscript's own values in `results/paper_metrics.csv` are only *compared
against*, so a discrepancy is visible rather than hidden. Expected output is in
[EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md).

## Tier 2 — needs the restricted corpus and GPUs

The training and evaluation stages cannot be re-run from this repository.

| stage | what it needs | script here |
|---|---|---|
| Train one arm for 150 epochs | the restricted corpus, the D-FINE-X teacher evidence bank, `ultralytics` 8.4.24, one A100/H100-class GPU (~1 day per run; 12 factorial runs plus the nine-arm family) | not included — the training supervisor is project-internal |
| Freeze-evaluate one arm on the primary estimand | the trained `last.pt`, the val split, the project's unified COCOeval wrapper | `scripts/freeze_eval.py` (paths must be adapted; see its header) |
| Source-clustered bootstrap on the locked test | the locked-test GT and every arm's prediction JSON | `scripts/source_cluster_bootstrap.py` (set `DATA_ROOT`) |
| Rebuild the factorial schedules | nothing restricted — but note the released schedule file is the frozen one and its SHA-256 is pinned | `python3 -m lpccad.schedule --out /tmp/sched.yaml -K 3` |

Access to the restricted material is via data-use agreement through the editor:
[RESTRICTED_DATA_ACCESS.md](RESTRICTED_DATA_ACCESS.md).

### Regenerating the schedules is a check, not a rebuild

`lpccad/schedule.py` is deterministic given its seeds (base seed 20260721, see
`configs/experiment_seeds.yaml`). Re-running it reproduces
`configs/factorial_schedules/phase4_view_schedules_v3_factorial.yaml`
**byte-identically** — verified on Python 3.14 with PyYAML 6.0.3, digest
`bfdd9285...`, the same prefix the paper cites. A different PyYAML version may
emit different formatting, in which case the digest changes while the schedules
do not: compare the parsed content in that case, and treat the shipped file
plus its `.sha256` as authoritative.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, pyyaml, faster-coco-eval, pytest
```

`ultralytics` is intentionally **not** in `requirements.txt`: nothing in this
repository imports it at Tier 1, and it is AGPL-3.0. Install it yourself only
if you are attempting Tier 2 — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

`numpy` is not pinned here, but the training environment used `numpy<2` because
the evaluator wheels were built against the 1.x ABI. If `faster-coco-eval`
raises an ABI error, that is the first thing to check.

## Numerical expectations

* The three seed-blocked contrasts reproduce the published values to within
  `1e-4` (`tests/test_expected_statistics.py` enforces this).
* The paired-$t$ intervals and TOST $p$-values are recomputed with an exact
  closed-form Student-$t$ CDF at df = 2, so no SciPy is needed and the values
  are deterministic across machines.
* The bootstrap CIs in `results/bootstrap_summary.csv` are *not* recomputed
  here — they are the output of a 1000-replicate run over the restricted
  locked-test artifacts (seed 20260725). Re-running them requires Tier 2.
