# Data provenance

**No dataset, image, video frame or checkpoint is included in this repository.**
The corpus used in the paper aggregates third-party research datasets; the
authors collected no footage. This file records where each constituent family
comes from and under which terms, so that a reader can judge the licence
position without receiving any restricted material.

## Constituent families

| family (filename prefix) | origin | licence status |
|---|---|---|
| `armas` | OD-WeaponDetection / SoHas, ari-dasci, Univ. Granada — <https://github.com/ari-dasci/OD-WeaponDetection> | **Discrepancy disclosed.** The repository README states CC BY-SA 4.0 while the bundled `License.md` carries the CC BY 4.0 text. We comply with the **stricter CC BY-SA 4.0**: attribution is given, and no derivative dataset is redistributed, so share-alike is not triggered. |
| `hf_w_turki_ds_5` / `hf_w_turki_ds_6` | HuggingFace weapon collection (`turki`) | CC BY 4.0 per the dataset card. |
| `hf_w_turki_ds_9` (`Normal_Videos`) | UCF-Crime normal-video frames (Sultani et al., CVPR 2018) — <https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos> | Research distribution from the official project page; **no formal licence text**, and **written confirmation of the research-use terms had not been obtained at the time of writing**. The manuscript commits in print to the fallback: if confirmation cannot be obtained, this source (20.9% of the test split, hard negatives only) is removed and the locked-test analysis is re-run without it. |
| UUID clips | "Dangerous Items" 5-class VOC archive, Zenodo record **13786228** v2 (Omiotek, 2025) — <https://zenodo.org/records/13786228> | **Verified CC BY 4.0.** The archived copy matches the Zenodo record byte-for-byte: filename, size 357,834,955 bytes, and MD5 `d162fd620d33e2a330c089f3b6f0babb`. |
| `cocohn` / `cocohn_t` | COCO-2017-derived hard negatives | Annotations CC BY 4.0; the images themselves remain under individual Flickr terms. |

Content notes: `armas` stills may contain identifiable persons; the UUID family
is surveillance-style clip material (52.3% of the test split); `cocohn` are
person-co-occurrence negatives. **None of this material is in this
repository**, and none of it may be added to it.

## Split composition (source of truth for the percentages in the paper)

| family | train n | train % | val n | val % | test n | test % |
|---|---:|---:|---:|---:|---:|---:|
| `cocohn_t` | 4,000 | 26.7 | 0 | 0 | 0 | 0 |
| `cocohn` | 0 | 0 | 819 | 29.0 | 0 | 0 |
| `armas` | 2,700 | 18.0 | 0 | 0 | 0 | 0 |
| `hf_w_turki_ds_6` | 1,780 | 11.9 | 846 | 29.9 | 608 | 23.8 |
| `hf_w_turki_ds_5` | 456 | 3.0 | 149 | 5.3 | 7 | 0.3 |
| `hf_w_turki_ds_9` | 0 | 0 | 0 | 0 | 534 | 20.9 |
| UUID clips | 1,599 | 10.7 | 881 | 31.2 | 1,334 | 52.3 |
| other | 4,461 | 29.7 | 132 | 4.7 | 70 | 2.7 |
| **total** | **14,996** | | **2,827** | | **2,553** | |

The train/val/test families differ materially — this is the basis for reading
the locked test as a transport check rather than an in-distribution replication.

## Governance

* **Data controller** for the assembled corpus: the corresponding author's
  institution.
* **Retention**: constituents are kept as research copies for five years after
  publication, then deleted.
* **Redistribution**: the assembled corpus is *not* redistributed. Constituents
  remain governed by their own terms; the assembled manifests name files only.
* **Ethics**: the determination that no IRB review was required (no
  human-subjects intervention; pre-existing third-party research data) was made
  under the corresponding author's institutional research-data policy and is
  stated as such, not as an external approval.
* **Access for verification**: by data-use agreement through the journal editor
  — see [RESTRICTED_DATA_ACCESS.md](RESTRICTED_DATA_ACCESS.md).

## What this repository does contain

Frozen numeric outputs (`results/*.csv`), the frozen protocol and schedule
configs (`configs/`), and the analysis and method code (`lpccad/`, `scripts/`,
`tests/`). Nothing else. See the release manifest accompanying the submission
for the do-not-upload list.
