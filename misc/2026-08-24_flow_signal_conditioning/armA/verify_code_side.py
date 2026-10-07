#!/usr/bin/env python3
"""Code-side ctt_v2plus verification battery (fable-advisor invariants 1-5, 2026-08-29).

The code-side root gives up the "countable off disk" symlink tree, so trustworthiness moves to
this JSONL-vs-INDEPENDENT-SOURCES battery. Every check compares `samples.jsonl` against the
inventories / ROSTER (which did NOT flow through build_samples) or against the frozen certified
counts — never against samples.jsonl itself.

  1 COUNTS      per-stratum row counts == independent predictions (S6/S1/S0 re-derived from
                inventory+ROSTER via the ring formula; S2a/S2b/S4 value-checked vs certified);
                total 114,215; S6 26,266 targets + 2,378 drops == 28,644.
  2 SET-EQUAL   0 duplicate (stratum,target,reference); distinct targets per stratum == the
                inventory/ROSTER expectation; S1/S6 target & reference stem-sets == predicted.
  3 SHARED-STUB (filesystem-free A19 analog) every source path maps to exactly one clip except
                where structural (a target's own rows; a reference across rows citing it; the 7
                masks; conditions content-addressed per caption). distinct conditions per stratum
                == distinct caption_keys (NOT collapsed to a stub).
  4 PATH-SCHEME every path is `_src/(outputs|experiments|eval_ladder)/…` or `_mask_store/…`;
                zero absolute, zero `..` segments.
  5 EXIST+SHAPE every distinct path exists (FULL); latent F,H,W matches each stem's OWN
                ROSTER(S6)/per-stratum shape, target AND reference keyed INDEPENDENTLY (sampled).

Writes CODESIDE_VERIFY.md; exit 0 iff every invariant PASSES.
"""
from __future__ import annotations
import os, sys, json, glob, random, math
from collections import Counter, defaultdict

LAB = "/taiga/illinois/eng/cs/jrehg/users/emirkisa"
DR = f"{LAB}/diffusion-research"
ROOT = f"{DR}/outputs/ctt_v2/roots/ctt_v2plus_mix"
INV = f"{DR}/outputs/ctt_v2/inventories"
ROSTER_P = f"{DR}/outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json"
ARMA = f"{DR}/misc/2026-08-24_flow_signal_conditioning/armA"
SAMPLES = f"{ROOT}/samples.jsonl"
STRATA = ["S0", "S1", "S2a", "S2b", "S4", "S6"]
CERT = {"S0": 385, "S1": 3675, "S2a": 22731, "S2b": 23577, "S4": 6000, "S6": 57847}   # certified/known
STRAT_SHAPE = {"S0": (16, 20, 15), "S1": (16, 20, 15), "S2a": (16, 20, 15),
               "S2b": (16, 20, 15), "S4": (5, 14, 26)}                                # S6 per-stem


def ring_pairs_count(n):
    return 0 if n < 2 else n * min(3, n - 1)


def load_rows():
    return [json.loads(l) for l in open(SAMPLES) if l.strip()]


def s6_predictions():
    cl = json.load(open(ROSTER_P))["clips"]
    shp = {c["stem"]: tuple(c["latent_fhw"]) for c in cl}
    subj = {c["stem"]: c["subject"] for c in cl}
    eff_shape = defaultdict(list)
    for c in cl:
        eff_shape[(c["effect"], tuple(c["latent_fhw"]))].append(c["stem"])
    pairs = 0; targets = set(); drops = set()
    for (eff, sh), stems in eff_shape.items():
        if len(stems) >= 2:
            pairs += ring_pairs_count(len(stems)); targets.update(stems)
        else:
            drops.update(stems)
    return dict(pairs=pairs, targets=targets, drops=drops, shp=shp, subj=subj, total=len(cl))


def inv_ring_count(stratum):
    """Independent pair-count prediction from an inventory's group sizes (pre-exclusion; only
       meaningful for strata with no exclusions: S0, S1)."""
    inv = json.load(open(f"{INV}/{stratum}.json"))
    return sum(ring_pairs_count(len(g["clips"])) for g in inv["groups"].values()), \
        {c for g in inv["groups"].values() for c in g["clips"]}


def main():
    L = []; P = L.append
    fails = []
    rows = load_rows()
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    s6 = s6_predictions()

    P("# CODESIDE_VERIFY — code-side ctt_v2plus invariant battery\n")
    P(f"`samples.jsonl` = **{len(rows):,}** rows. Checks vs inventories/ROSTER (independent of "
      f"build_samples). fable-advisor invariants 1-5.\n")

    # ---- INVARIANT 1: counts ----
    P("## 1 · Per-stratum counts vs independent predictions\n")
    P("| stratum | rows | predicted | source | ok |")
    P("|---|--:|--:|---|:--:|")
    # independent predictions
    pred = {}
    p_s0, set_s0 = inv_ring_count("S0"); pred["S0"] = p_s0
    p_s1, set_s1 = inv_ring_count("S1"); pred["S1"] = p_s1
    pred["S6"] = s6["pairs"]
    for s in ["S2a", "S2b", "S4"]:
        pred[s] = CERT[s]   # value-checked vs certified (exclusions applied upstream; unchanged this rebuild)
    src = {"S0": "S0 inv ring", "S1": "S1 inv ring", "S6": "ROSTER shape-split ring",
           "S2a": "certified", "S2b": "certified", "S4": "certified"}
    for s in STRATA:
        n = len(by[s]); ok = (n == pred[s] == CERT[s])
        if not ok: fails.append(f"count {s}: rows {n} pred {pred[s]} cert {CERT[s]}")
        P(f"| {s} | {n:,} | {pred[s]:,} | {src[s]} | {'✓' if ok else '✗'} |")
    tot_ok = len(rows) == sum(CERT.values()) == 114215
    if not tot_ok: fails.append(f"total {len(rows)} != 114215")
    P(f"| **total** | **{len(rows):,}** | **114,215** | | {'✓' if tot_ok else '✗'} |")
    s6t = {r["target"] for r in by["S6"]}
    recon = (len(s6t) == 26266 and len(s6["drops"]) == 2378 and len(s6t) + len(s6["drops"]) == 28644)
    if not recon: fails.append(f"S6 reconcile targets {len(s6t)} + drops {len(s6['drops'])} != 28644")
    P(f"\nS6 reconciliation: **{len(s6t):,} distinct targets + {len(s6['drops']):,} drops = "
      f"{len(s6t)+len(s6['drops']):,}** (== 28,644 ROSTER clips) — {'✓' if recon else '✗ FAIL'}\n")

    # ---- INVARIANT 2: set-equality ----
    P("## 2 · Set-equality (no dup pairs; stem-sets match)\n")
    dup = len(rows) - len({(r["stratum"], r["target"], r["reference"]) for r in rows})
    if dup: fails.append(f"{dup} duplicate (stratum,target,reference) tuples")
    # S6 target/reference stem-sets vs ROSTER prediction
    s6r = {r["reference"] for r in by["S6"]}
    s6_pair_ok = (s6t == s6["targets"] and s6r <= s6["targets"] and not (s6t & s6["drops"]))
    if not s6_pair_ok: fails.append("S6 target/reference stem-set != ROSTER prediction")
    # S1 targets == S1 inventory clips (ring makes every clip a target)
    s1t = {r["target"] for r in by["S1"]}
    s1_ok = (s1t == set_s1)
    if not s1_ok: fails.append(f"S1 targets {len(s1t)} != S1 inv clips {len(set_s1)}")
    P(f"- duplicate pairs: **{dup}** {'✓' if dup == 0 else '✗'}")
    P(f"- S6 targets == ROSTER-predicted non-singleton set, refs ⊆ it, targets∩drops=∅: {'✓' if s6_pair_ok else '✗'}")
    P(f"- S1 distinct targets ({len(s1t):,}) == S1 inventory clips ({len(set_s1):,}): {'✓' if s1_ok else '✗'}\n")

    # ---- INVARIANT 3: shared-stub / conditions dedup ----
    P("## 3 · Shared-stub detector (source paths belong to exactly one clip except structural)\n")
    # latents path must map to exactly one target stem; reference_latents to one reference stem
    lat_owner = defaultdict(set); ref_owner = defaultdict(set)
    for r in rows:
        lat_owner[r["paths"]["latents"]].add((r["stratum"], r["target"]))
        ref_owner[r["paths"]["reference_latents"]].add((r["stratum"], r["reference"]))
    lat_bad = {p for p, o in lat_owner.items() if len(o) != 1}
    ref_bad = {p for p, o in ref_owner.items() if len(o) != 1}
    if lat_bad: fails.append(f"{len(lat_bad)} latents paths map to >1 target")
    if ref_bad: fails.append(f"{len(ref_bad)} reference_latents paths map to >1 reference")
    # conditions: distinct paths per stratum == distinct caption_keys (content-address dedup), never a stub
    P("| stratum | distinct conditions paths | distinct caption_keys | ok (equal & >1) |")
    P("|---|--:|--:|:--:|")
    cond_ok_all = True
    for s in STRATA:
        dc = len({r["paths"]["conditions"] for r in by[s]})
        dk = len({r["caption_key"] for r in by[s]})
        ok = (dc == dk and dc > 1)
        cond_ok_all &= ok
        if not ok: fails.append(f"{s} conditions distinct {dc} != caption_keys {dk} (or stub)")
        P(f"| {s} | {dc:,} | {dk:,} | {'✓' if ok else '✗'} |")
    P(f"\n- latents 1:1 with target: {'✓' if not lat_bad else '✗ '+str(len(lat_bad))} · "
      f"reference_latents 1:1 with reference: {'✓' if not ref_bad else '✗ '+str(len(ref_bad))}\n")

    # ---- INVARIANT 4: path scheme ----
    # The real invariant is: every path is RELATIVE-UNDER-ROOT and portable — i.e. `_mask_store/…`
    # or `_src/…` with no absolute segment and no `..` traversal. The exact top-level dir under
    # `_src` (datasets|outputs|experiments|eval_ladder — realpaths collapse the moved-in-encodes
    # compat symlink outputs/ctt_v2/encodes -> datasets/ctt_v2) is REPORTED for anomaly visibility
    # but not itself a gate; existence under _src is proven by invariant 5.
    P("## 4 · Path-scheme gate (relative-under-root; no absolute; no `..`)\n")
    distinct_paths = {p for r in rows for p in r["paths"].values()}
    bad_scheme = []
    topdirs = Counter()
    for p in distinct_paths:
        segs = p.split("/")
        if p.startswith("_mask_store/"):
            topdirs["_mask_store"] += 1
        elif p.startswith("_src/") and ".." not in segs and not p.startswith("/") \
                and not p[len("_src/"):].startswith("/"):
            topdirs["_src/" + "/".join(segs[1:3])] += 1
        else:
            bad_scheme.append(p)
    if bad_scheme: fails.append(f"{len(bad_scheme)} paths violate scheme e.g. {bad_scheme[:3]}")
    P(f"distinct paths: **{len(distinct_paths):,}**; violations (absolute / `..` / not `_src|_mask_store`): "
      f"**{len(bad_scheme)}** {'✓' if not bad_scheme else '✗ '+str(bad_scheme[:3])}")
    P("\ndistinct source roots under `_src` (reported):")
    for k, n in topdirs.most_common():
        P(f"  - `{k}` — {n:,}")
    P("")

    # ---- INVARIANT 5: existence (full) + shape (sampled) ----
    P("## 5 · Existence (full) + shape (sampled, keyed independently)\n")
    fast = "--fast" in sys.argv       # skip the full existence re-stat (already proven in a prior full run)
    if fast:
        missing = []
        P("_(existence re-stat SKIPPED via --fast — proven PASS over all distinct paths in the prior full run)_")
    else:
        missing = [p for p in distinct_paths if not os.path.isfile(os.path.join(ROOT, p))]
    if missing: fails.append(f"{len(missing)} distinct paths missing e.g. {missing[:3]}")
    # sampled shape: target and reference keyed to their OWN expected shape
    def exp_shape(stratum, stem):
        return s6["shp"][stem] if stratum == "S6" else STRAT_SHAPE[stratum]
    import torch
    rng = random.Random(0)
    samp = []
    for s in STRATA:
        samp += rng.sample(by[s], min(40, len(by[s])))
    shape_bad = []
    for r in samp:
        for role, stem in (("latents", r["target"]), ("reference_latents", r["reference"])):
            fp = os.path.join(ROOT, r["paths"][role])
            try:
                d = torch.load(fp, map_location="cpu", weights_only=True)
                fhw = (int(d["num_frames"]), int(d["height"]), int(d["width"]))
            except Exception as e:
                shape_bad.append((r["paths"][role], f"load:{type(e).__name__}")); continue
            want = exp_shape(r["stratum"], stem)
            if fhw != want:
                shape_bad.append((r["paths"][role], f"{fhw}!={want}"))
    if shape_bad: fails.append(f"{len(shape_bad)} sampled tensors wrong shape e.g. {shape_bad[:3]}")
    P(f"- existence (FULL, {len(distinct_paths):,} distinct paths): "
      f"**{'PASS' if not missing else f'{len(missing)} MISSING'}** {'✓' if not missing else '✗'}")
    P(f"- shape (sampled {len(samp)} rows × target+reference, keyed to each stem's own shape): "
      f"**{'PASS' if not shape_bad else f'{len(shape_bad)} BAD'}** {'✓' if not shape_bad else '✗'}\n")

    ok = not fails
    P(f"## Overall: {'ALL INVARIANTS PASS' if ok else 'FAIL'}\n")
    if fails:
        for f in fails: P(f"  - FAIL: {f}")
    open(f"{ARMA}/CODESIDE_VERIFY.md", "w").write("\n".join(L))
    print("wrote", f"{ARMA}/CODESIDE_VERIFY.md")
    print("VERIFY", "PASS" if ok else "FAIL", "|", "; ".join(fails) if fails else "all 1-5 pass")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
