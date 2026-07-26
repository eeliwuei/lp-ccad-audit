#!/usr/bin/env python3
"""Constructive check that the nine randomized factorial schedules carry the
EXACT C4-M global view multiset (dose match) and to print the per-class active
epoch ledger and the head/tail cardinality mix.

Run from the repository root:  python3 scripts/verify_schedule_exposure.py
The same invariants are pinned as an automated test in
tests/test_schedule_multiset.py (which needs no PyYAML).
"""
import sys
from pathlib import Path
from collections import Counter

try:
    import yaml
except ImportError:  # pragma: no cover - dependency hint only
    sys.exit("PyYAML is required for this script: pip install pyyaml "
             "(or run tests/test_schedule_multiset.py, which does not need it)")

REPO = Path(__file__).resolve().parents[1]
V3_PATH = REPO / "configs/factorial_schedules/phase4_view_schedules_v3_factorial.yaml"
V1_PATH = REPO / "configs/frozen_protocol/phase4_view_schedules_v1.yaml"
Y=yaml.safe_load(open(V3_PATH))
V1=yaml.safe_load(open(V1_PATH))
CLS={"full":["knife","gun","stick"],"group_knife_stick":["knife","stick"],"group_gun_knife":["gun","knife"],
     "binary_knife_stick":["knife","stick"],"binary_gun_knife":["gun","knife"],
     "single_knife":["knife"],"single_gun":["gun"],"single_stick":["stick"]}
ref=Counter(V1["schedules"]["C4-M"])
print("C4-M reference multiset:",dict(sorted(ref.items())))
ok=True
for name,seq in sorted(Y["schedules"].items()):
    mc=Counter(seq); same=(mc==ref)
    ca=Counter()
    for v in seq:
        for c in CLS[v]: ca[c]+=1
    ok &= same
    k,g,s=ca["knife"],ca["gun"],ca["stick"]
    print("%-16s len=%d multiset==C4-M: %s | class epochs k/g/s = %d/%d/%d"%(name,len(seq),same,k,g,s))
print("ALL MULTISETS MATCH:",ok)
CARD={"full":3,"group_knife_stick":2,"group_gun_knife":2,"binary_knife_stick":2,"binary_gun_knife":2,"single_knife":1,"single_gun":1,"single_stick":1}
for name in ["F-MonoMix-s1","F-ShufSingle-s1","F-ShufMix-s1"]:
    seq=Y["schedules"][name]
    head,tail=seq[:110],seq[110:]
    print("%s: head card mix %s | tail card mix %s"%(name,dict(Counter(CARD[v] for v in head)),dict(Counter(CARD[v] for v in tail))))
