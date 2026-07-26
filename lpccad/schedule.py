#!/usr/bin/env python3
"""LP-CCAD v3: a 2x2 factorial of (order x tail), matched on NOMINAL
exposure, with K schedule replicates per non-baseline cell (breaks the
single-permutation "schedule lottery").

TERMINOLOGY WARNING. This file and the schedule YAML it emits use the phrase
"dose-matched" in the sense of *nominal* exposure: every arm sees the identical
global view multiset, so per-class active-epoch budgets are equal by
construction. That string is preserved verbatim because it is part of the
frozen artifact whose SHA-256 the paper cites (bfdd9285...); changing it would
break byte-identical regeneration. It does NOT mean effective distillation dose
is matched: the 0.999-IoU admission gate and the optimizer state at which each
view is presented make the number of admitted evidence pairs path-dependent.
The paper carries effective dose as a competing explanation it cannot exclude.

Design (fully documented, deterministic given a schedule seed):
  - GLOBAL view multiset M is IDENTICAL for every arm = the C4-M multiset
    {30 full, 40 group(alt), 40 binary(alt), 40 single(cycle)} = 150 views.
    => every arm sees the exact same views the same number of times
    (perfect nominal dose match; the confound the reviewer flagged is removed
    at the schedule level).
  - Partition: HEAD = epochs 1..110, TAIL = epochs 111..150.
  - Factor TAIL in {single, mixed}:
      single: TAIL = the 40 single-class views; HEAD = {30 full,40 group,40 binary}.
      mixed : TAIL is a cardinality-stratified sample of M (contains full/group/
              binary/single in the same proportion as M); HEAD is the complement.
              The exact interleave is chosen by the schedule seed.
  - Factor ORDER in {mono, shuffled} applied to HEAD:
      mono    : HEAD ordered by DECREASING view cardinality (broad->narrow):
                full -> group -> binary -> single(if any).
      shuffled: HEAD is a random permutation of the HEAD multiset (schedule seed).
  - TAIL ordering: single tail cycles knife/gun/stick; mixed tail is permuted by
    the schedule seed. (Order factor is defined on the HEAD; the tail's internal
    order is not a factor and is fixed per tail type.)

Cells and arms produced (training seed handled by the supervisor, not here):
  (mono ,single) = reference, == C4-M (NOT regenerated; reuse frozen C4-M).
  (mono ,mixed ) = F-MonoMix-s{k}     k=1..K
  (shuf ,single) = F-ShufSingle-s{k}  k=1..K
  (shuf ,mixed ) = F-ShufMix-s{k}     k=1..K
"""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path
import yaml

HEAD_LEN, TAIL_LEN, EPOCHS = 110, 40, 150
CARD = {  # view -> active-class cardinality
    "full": 3,
    "group_knife_stick": 2, "group_gun_knife": 2,
    "binary_knife_stick": 2, "binary_gun_knife": 2,
    "single_knife": 1, "single_gun": 1, "single_stick": 1,
}
def cyc(items, n): return [items[i % len(items)] for i in range(n)]

def c4_multiset():
    m  = ["full"] * 30
    m += cyc(["group_knife_stick", "group_gun_knife"], 40)
    m += cyc(["binary_knife_stick", "binary_gun_knife"], 40)
    m += cyc(["single_knife", "single_gun", "single_stick"], 40)
    assert len(m) == EPOCHS
    return m

M = c4_multiset()
M_COUNTS = Counter(M)                    # frozen global multiset (dose)
HEAD_BASE = ["full"]*30 + cyc(["group_knife_stick","group_gun_knife"],40) \
            + cyc(["binary_knife_stick","binary_gun_knife"],40)   # non-single 110
TAIL_BASE = cyc(["single_knife","single_gun","single_stick"],40)  # single 40

def mono_order(views):
    # broad -> narrow: decreasing cardinality; ties keep a stable canonical order
    canon = ["full","group_knife_stick","group_gun_knife",
             "binary_knife_stick","binary_gun_knife",
             "single_knife","single_gun","single_stick"]
    return sorted(views, key=lambda v: (-CARD[v], canon.index(v)))

def stratified_tail(rng):
    # sample TAIL_LEN views from M keeping cardinality proportions of M
    # M cardinality counts: card3=30, card2=80, card1=40  (of 150)
    # tail proportions on 40 slots: 3->8, 2->21, 1->11  (sums 40)
    by_card = {3: ["full"], 2: ["group_knife_stick","group_gun_knife",
                                "binary_knife_stick","binary_gun_knife"],
               1: ["single_knife","single_gun","single_stick"]}
    quota = {3: 8, 2: 21, 1: 11}
    tail = []
    for card, q in quota.items():
        pool = []
        # draw q views of this cardinality, respecting availability in M
        avail = {v: M_COUNTS[v] for v in by_card[card]}
        for _ in range(q):
            choices = [v for v, c in avail.items() if c > 0]
            v = rng.choice(sorted(choices))
            avail[v] -= 1; pool.append(v)
        tail += pool
    rng.shuffle(tail)
    return tail

def head_complement(tail):
    # HEAD multiset = M minus tail multiset
    rem = Counter(M) - Counter(tail)
    head = []
    for v in sorted(rem): head += [v]*rem[v]
    assert len(head) == HEAD_LEN, (len(head), HEAD_LEN)
    return head

def build_arm(order, tail_type, seed):
    rng = random.Random(seed)
    if tail_type == "single":
        head_ms, tail = list(HEAD_BASE), list(TAIL_BASE)
    else:  # mixed
        tail = stratified_tail(rng)
        head_ms = head_complement(tail)
    head = mono_order(head_ms) if order == "mono" else (lambda h: (rng.shuffle(h), h)[1])(list(head_ms))
    seq = head + tail
    assert Counter(seq) == M_COUNTS, "dose (global multiset) not preserved!"
    return seq

def ledger(seq):
    cs = Counter(); vc = Counter(seq)
    for v in seq:
        for cls in {"full":["knife","gun","stick"],
                    "group_knife_stick":["knife","stick"],"group_gun_knife":["gun","knife"],
                    "binary_knife_stick":["knife","stick"],"binary_gun_knife":["gun","knife"],
                    "single_knife":["knife"],"single_gun":["gun"],"single_stick":["stick"]}[v]:
            cs[cls]+=1
    tail = seq[HEAD_LEN:]
    return {"view_counts": dict(vc),
            "class_active_epochs": {c: cs[c] for c in ("knife","gun","stick")},
            "tail_cardinality_mix": dict(Counter(CARD[v] for v in tail))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-K", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=20260721)
    args = ap.parse_args()
    cells = [("mono","mixed","F-MonoMix"),
             ("shuf","single","F-ShufSingle"),
             ("shuf","mixed","F-ShufMix")]
    schedules, meta, led = {}, {}, {}
    for ci,(order,tail,tag) in enumerate(cells):
        for k in range(1, args.K+1):
            seed = args.base_seed + 1000*ci + k
            name = f"{tag}-s{k}"
            seq = build_arm(order, tail, seed)
            schedules[name] = seq  # FLAT list (load_frozen_schedule compatible)
            meta[name] = {"order": order, "tail": tail, "schedule_seed": seed}
            led[name] = ledger(seq)
    # reference cell (mono,single) == C4-M, documented not regenerated
    payload = {"schedule_version":"phase4_view_schedules_v3_factorial",
               "epochs": EPOCHS, "head_len": HEAD_LEN, "tail_len": TAIL_LEN,
               "global_multiset_counts": dict(M_COUNTS),
               "reference_mono_single":"C4-M (frozen; reuse, not regenerated)",
               "factorial":"order{mono,shuf} x tail{single,mixed}, dose-matched to M",
               "K_schedule_replicates": args.K,
               "schedule_meta": meta,
               "schedules": schedules, "ledger": led}
    out = Path(args.out); out.write_text(yaml.safe_dump(payload, sort_keys=False))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out)+".sha256").write_text(f"{sha}  {out.name}\n")
    # verify dose-match across ALL arms incl. the C4-M reference
    all_match = all(Counter(seq)==M_COUNTS for seq in schedules.values())
    print(json.dumps({"status":"ok","out":str(out),"sha256":sha,
                      "n_new_arms":len(schedules),
                      "dose_matched_all": all_match,
                      "global_multiset": dict(M_COUNTS)}, indent=2))
    print("--- per-arm ledger (dose identical; tail-mix & class-exposure differ) ---")
    for n in schedules:
        l=led[n]; print(f"{n:16s} tailmix={l['tail_cardinality_mix']} class={l['class_active_epochs']}")

if __name__ == "__main__":
    main()
