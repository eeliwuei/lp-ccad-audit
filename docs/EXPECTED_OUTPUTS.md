# Expected outputs

Run these two commands from the repository root. If your numbers differ from
what is printed below, something is wrong with the checkout, not with your
machine — every value here is deterministic (no sampling, no GPU, closed-form
Student-$t$ at df = 2).

## 1. `python3 scripts/reproduce_tables.py`

Stdlib only; takes well under a second. The output has five blocks.

### Table 10 — factorial effects (recomputed live)

```
effect          estimate                95% CI (t)   TOST p  Holm TOST p
------------------------------------------------------------------------------
order           -0.00104      [-0.01315, +0.01107]    0.043        0.129
tail            +0.00567      [-0.00380, +0.01515]    0.094        0.188
interaction     -0.00066      [-0.02762, +0.02631]    0.137        0.188
```

followed by the equivalence verdicts

```
  order        90% CI [-0.00926, +0.00718]  ->  equivalent to zero at the uncorrected 5% level; NOT equivalent after Holm
  tail         90% CI [-0.00076, +0.01210]  ->  inconclusive
  interaction  90% CI [-0.01896, +0.01765]  ->  inconclusive
```

and the difference-from-zero p-values, which are a **different null** and must
never be printed next to equivalence language:

```
  order        diff-from-zero p = 0.747   Holm-adjusted = 1.00
  tail         diff-from-zero p = 0.123   Holm-adjusted = 0.37
  interaction  diff-from-zero p = 0.926   Holm-adjusted = 1.00   [unenforced p*(m-i) = 0.93]
```

then the cross-check against the manuscript, which must end with

```
  => all effects reproduce within 1e-4
```

Rounding notes, so that small differences from the printed paper do not look
like errors:

* the manuscript prints the effects at 4 decimals (`-0.0010`, `+0.0057`,
  `-0.0007`); this script prints 5 (`-0.00104`, `+0.00567`, `-0.00066`);
* the manuscript's TOST Holm values are `0.13 / 0.19 / 0.19`, i.e. the same
  `0.129 / 0.188 / 0.188` at 2 decimals;
* for the *secondary* difference-from-zero test the interaction's adjusted
  value is `1.00`. An earlier manuscript draft carried `0.93`, the unenforced
  `p x (m - i)` product; Holm is a step-down procedure whose adjusted sequence
  must be monotone, which gives `1.00` (and is what
  `statsmodels.multipletests(method='holm')` returns). The manuscript was
  corrected to `1.00`; the script prints both so the discrepancy in any older
  copy is self-explaining. This affects no verdict — the interaction is
  inconclusive either way — but the two numbers should not be confused.

### Table 11 — the twelve raw runs

```
cell                   seed 42     seed 1337 seed 20260703        mean
------------------------------------------------------------------------------
mono x single          0.38261       0.37049       0.36855     0.37388
mono x mixed           0.37073       0.36993       0.36496     0.36854
shuf x single          0.37336       0.37255       0.37985     0.37525
shuf x mixed           0.37225       0.37008       0.36543     0.36925
------------------------------------------------------------------------------
C0-R (no KD)           0.37812       0.38060       0.38211     0.38028
```

with the paired differences from C0-R summarised as

```
  -> 11 of 12 paired differences are negative (KD does not beat the no-KD baseline)
```

the single exception being `mono x single` at seed 42 (`+0.00449`), and the
seed-blocked contrasts

```
order                 +0.00386      -0.00110      -0.00588    -0.00104
tail                  +0.00650      +0.00151      +0.00901    +0.00567
interaction           +0.01077      -0.00191      -0.01083    -0.00066
```

The block ends with the twelve checkpoint SHA-256 digests, which are what a
DUA holder checks their downloaded weights against.

### Locked-test primaries

18 rows, nine arms. Spot values:

```
C0-R                42     0.40402    0.3941    0.3262    0.4917
C1-M                42     0.40952    0.3832    0.3397    0.5056
C4-M                42     0.40439    0.3856    0.3327    0.4949
C4-M              1337     0.38106    0.3712    0.3201    0.4519
C4Mix-M           1337     0.40464    0.3763    0.3348    0.5029
```

### Table 7(b) — locked-test per-seed contrasts

`reproduce_tables.py` recomputes each registered contrast from the panel-(a)
primaries in `results/locked_test_contrasts.csv` and prints:

```
H1  C4-M - C1-M                -0.00513     -0.02991     +0.00945     -0.00853  inconclusive
H2  C4-M - C4MixFT-M           +0.00559     -0.02382           --     -0.00911  inconclusive
H3  C4MixFT-M - C4Mix-M        +0.01036     +0.00024           --     +0.00530  inconclusive
H4  C4-M - C4R-M               -0.00717     -0.00298           --     -0.00508  inconclusive
H5  C1-M - C0-R                +0.00550     +0.01084     -0.01153     +0.00161  inconclusive
```

The manuscript rounds these means to `-0.0085 / -0.0091 / +0.0053 / -0.0051 /
+0.0016`, and every one is inside the frozen test noise floor
`eps_test = 0.0191` — which is the paper's central confirmatory result: the
locked test resolves nothing. `tests/test_expected_statistics.py::
test_locked_test_contrast_means_match_the_paper` pins both the values and the
floor, so a future edit cannot change that conclusion silently.

Dashes are arms the frozen plan did not replicate at that seed; the mean is
taken over the seeds that exist, exactly as in the manuscript.

### Source-clustered bootstrap

```
H1        C4-M - C1-M               mean               -      [-0.01590, -0.00004]
H2        C4-M - C4MixFT-M          mean               -      [-0.02086, +0.00245]
H3        C4MixFT-M - C4Mix-M       mean               -      [-0.00467, +0.01592]
H4        C4-M - C4R-M              mean               -      [-0.01572, +0.00596]
H5        C1-M - C0-R               mean               -      [-0.00697, +0.01018]
```

followed by `intervals excluding zero: H1` — and H1's exclusion is on the
*negative* side, i.e. the curriculum arm does not beat full-view KD.

## 2. `python3 -m pytest -q`

With every optional dependency installed:

```
................                                                         [100%]
16 passed in 0.18s
```

With only `pytest` installed (no PyYAML, no faster-coco-eval, no torch):

```
...........                                                              [100%]
SKIPPED [1] tests/test_bootstrap_duplicate_ids.py:42: faster-coco-eval is not installed; ...
11 passed, 1 skipped in 0.02s
```

`test_schedule_multiset.py` and `test_expected_statistics.py` **must pass in
both cases** — they are the two that pin the paper's claims. The schedule test
falls back to a minimal parser when PyYAML is missing, and cross-checks the two
parsers against each other when it is present.

## 3. `python3 examples/synthetic_minimal_example.py` (needs torch)

```
view                active classes [loss type]                loss     grad L1
------------------------------------------------------------------------------
full                knife,gun,stick [sigmoid_bce]         0.734533    0.279856
group_knife_stick   knife,stick [sigmoid_bce]             0.733418    0.280842
group_gun_knife     knife,gun [sigmoid_bce]               0.768156    0.283601
binary_knife_stick  knife,stick [pairwise_margin]         1.516577    1.502975
binary_gun_knife    knife,gun [pairwise_margin]           1.656357    2.000000
single_knife        knife [sigmoid_bce]                   0.799549    0.289318
single_gun          gun [sigmoid_bce]                     0.736763    0.277884
single_stick        stick [sigmoid_bce]                   0.667288    0.272367
human_only          (none) [none]                         0.000000    0.000000
```

ending with

```
  |grad| on the ACTIVE class (knife)   = 0.289318
  |grad| on the INACTIVE classes       = 0.000000
```

The exact loss values depend on the PyTorch RNG for the synthetic batch (seeded
at 42) and may differ by a few ULP across versions; the two structural facts —
a nonzero gradient on the active class and an exactly zero gradient on the
sliced-out classes — must hold exactly.

## 4. `python3 scripts/reproduce_figures.py`

With matplotlib: writes `figures/fig_factorial_order_effect.pdf`,
`figures/fig_locked_test_contrasts.pdf`, `figures/fig_locked_test_arms.pdf`.
Without it: prints the same numbers as text. Either way it ends with the
explicit list of paper figures that are **not** regenerable from this release
(Figs. 1, 2, 3, 4, 5A-validation-half, 6) and why.
