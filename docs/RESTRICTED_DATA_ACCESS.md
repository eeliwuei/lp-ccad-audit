# Restricted data access

## Short version

The training/validation/test corpus, the teacher evidence bank, the trained
checkpoints and the locked-test prediction files are **not in this repository
and will not be added to it**. They are available for *verification* under a
data-use agreement (DUA) arranged **through the journal editor**.

## Why it is restricted rather than published

The corpus is an assembly of third-party research datasets, each governed by
its own terms (see [DATA_PROVENANCE.md](DATA_PROVENANCE.md)). We hold no right
to redistribute the assembly:

* one family (`armas` / SoHas) carries a README-vs-`License.md` licence
  discrepancy that we resolve conservatively in favour of the stricter
  CC BY-SA 4.0;
* one family (UCF-Crime normal-video frames) is distributed for research from
  its project page with no formal licence text, and written confirmation of the
  research-use terms was outstanding at the time of writing;
* the material includes surveillance-style footage and stills that may contain
  identifiable persons.

Publishing a mirror of the assembled corpus would therefore be a licensing and
a privacy decision we are not entitled to make. Every constituent family is
obtainable from its own canonical source, which is named in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## How to request access

1. Contact the handling editor of the paper (Machine Learning with
   Applications) and state that you are requesting verification access to the
   LP-CCAD audit materials.
2. The editor forwards the request to the corresponding author.
3. A data-use agreement is executed between the requester's institution and the
   corresponding author's institution. The DUA limits use to **verification of
   the published analyses**, prohibits redistribution and prohibits any attempt
   to identify individuals appearing in the material.
4. On execution, the requester receives: the split manifests (file names,
   splits, source-family labels), the frozen evaluation artifacts (GT JSON and
   per-arm prediction JSON for the locked test), and the checkpoint set whose
   SHA-256 digests are printed by `scripts/reproduce_tables.py`.

Requests routed to the authors directly will be redirected to the editor, so
that access is recorded on the editorial side.

## What you can verify without any DUA

Everything in Tier 1 of [REPRODUCIBILITY.md](REPRODUCIBILITY.md): the entire
statistical chain from the twelve raw runs to the published effects, the
schedule dose-match invariant, the projection loss itself, and the correctness
property of the bootstrap resampler. The restricted material is needed only to
re-derive the raw per-run metrics, not to check what was done with them.

## Retention

Constituents are retained as research copies for five years post-publication
and then deleted. A DUA executed inside that window is honoured; after it, the
canonical third-party sources named in `DATA_PROVENANCE.md` remain the route to
reassembly.
