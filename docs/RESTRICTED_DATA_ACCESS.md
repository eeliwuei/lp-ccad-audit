# Restricted data access

## Short version

The training/validation/test corpus, the teacher evidence bank, the trained
checkpoints and the locked-test prediction files are **not in this repository
and will not be added to it**. They are available for *verification* under a
data-use agreement (DUA), requested from the corresponding author and,
once the manuscript is with a journal, routed through the handling editor.

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

The manuscript is currently unpublished and is not under review at any venue;
the intended venue is *Machine Learning with Applications*.

1. Contact the corresponding author (see `CITATION.cff`) and state that you are
   requesting verification access to the LP-CCAD audit materials. If the
   manuscript is under consideration at a journal by the time you write, send
   the request through that journal's handling editor instead, so that access
   is recorded on the editorial side.
2. A data-use agreement is executed between the requester's institution and the
   corresponding author's institution. The DUA limits use to **verification of
   the reported analyses**, prohibits redistribution and prohibits any attempt
   to identify individuals appearing in the material.
3. On execution, the requester receives: the split manifests (file names,
   splits, source-family labels), the frozen evaluation artifacts (GT JSON and
   per-arm prediction JSON for the locked test), and the checkpoint set whose
   SHA-256 digests are printed by `scripts/reproduce_tables.py`.

Access is granted for verification only. It is not a redistribution licence for
the constituent collections, which keep their own terms
([DATA_PROVENANCE.md](DATA_PROVENANCE.md)).

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
