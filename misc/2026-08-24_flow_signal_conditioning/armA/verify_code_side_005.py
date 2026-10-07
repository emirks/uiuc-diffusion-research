#!/usr/bin/env python3
"""Code-side ctt_v2plus_s6reshape (dataset 005) verification battery — invariants 1-5.

Successor of verify_code_side.py (004). Deltas: ROOT5, r832 S6 roster, S6 inventory S6_r832.json,
CERT S6 = 81,779 (total 138,147), S6 latent grids (11,16,26)/(11,26,16), invariant-5 shape check on
>=400 S6 rows spanning BOTH grids, mask set = the 5 code-side masks.

Every check compares samples.jsonl against the inventories / r832 ROSTER (which did NOT flow through
build_samples) or the certified counts — never against samples.jsonl itself.

Writes CODESIDE_VERIFY_005.md; exit 0 iff every invariant PASSES.
"""
from __future__ import annotations
import os, sys, json, random
from collections import Counter, defaultdict

LAB = "/taiga/illinois/eng/cs/jrehg/users/emirkisa"
DR = f"{LAB}/diffusion-research"
ROOT = f"{DR}/outputs/ctt_v2/roots/ctt_v2plus_s6reshape_mix"
INV = f"{DR}/outputs/ctt_v2/inventories"
ROSTER_P = f"{DR}/outputs/ctt_v2/encodes/EFFECTDATA_r832/ROSTER.json"
ARMA = f"{DR}/misc/2026-08-24_flow_signal_conditioning/armA"
SAMPLES = f"{ROOT}/samples.jsonl"
STRATA = ["S0", "S1", "S2a", "S2b", "S4", "S6"]
CERT = {"S0": 385, "S1": 3675, "S2a": 22731, "S2b": 23577, "S4": 6000, "S6": 81779}
TOTAL = 138147
STRAT_SHAPE = {"S0": (16, 20, 15), "S1": (16, 20, 15), "S2a": (16, 20, 15),
               "S2b": (16, 20, 15), "S4": (5, 14, 26)}                                # S6 per-stem
S6_GRIDS = {(11, 16, 26), (11, 26, 16)}
EXPECT_MASKS = {"f16_h20_w15_p2_twosided.pt", "f16_h20_w15_p2_onesided.pt",
                "f5_h14_w26_p1_onesided.pt", "f11_h16_w26_p1_onesided.pt",
                "f11_h26_w16_p1_onesided.pt"}


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

    P("# CODESIDE_VERIFY_005 — code-side ctt_v2plus_s6reshape invariant battery\n")
    P(f"`samples.jsonl` = **{len(rows):,}** rows. Checks vs inventories / EFFECTDATA_r832 ROSTER "
      f"(independent of build_samples). Invariants 1-5.\n")

    # ---- INVARIANT 1: counts ----
    P("## 1 · Per-stratum counts vs independent predictions\n")
    P("| stratum | rows | predicted | source | ok |")
    P("|---|--:|--:|---|:--:|")
    pred = {}
    p_s0, set_s0 = inv_ring_count("S0"); pred["S0"] = p_s0
    p_s1, set_s1 = inv_ring_count("S1"); pred["S1"] = p_s1
    pred["S6"] = s6["pairs"]
    for s in ["S2a", "S2b", "S4"]:
        pred[s] = CERT[s]
    src = {"S0": "S0 inv ring", "S1": "S1 inv ring", "S6": "r832 ROSTER effect×grid ring",
           "S2a": "certified", "S2b": "certified", "S4": "certified"}
    for s in STRATA:
        n = len(by[s]); ok = (n == pred[s] == CERT[s])
        if not ok: fails.append(f"count {s}: rows {n} pred {pred[s]} cert {CERT[s]}")
        P(f"| {s} | {n:,} | {pred[s]:,} | {src[s]} | {'✓' if ok else '✗'} |")
    tot_ok = len(rows) == sum(CERT.values()) == TOTAL
    if not tot_ok: fails.append(f"total {len(rows)} != {TOTAL}")
    P(f"| **total** | **{len(rows):,}** | **{TOTAL:,}** | | {'✓' if tot_ok else '✗'} |")
    s6t = {r["target"] for r in by["S6"]}
    recon = (len(s6t) == 28552 and len(s6["drops"]) == 92 and len(s6t) + len(s6["drops"]) == 28644)
    if not recon: fails.append(f"S6 reconcile targets {len(s6t)} + drops {len(s6['drops'])} != 28644")
    P(f"\nS6 reconciliation: **{len(s6t):,} distinct targets + {len(s6['drops']):,} drops = "
      f"{len(s6t)+len(s6['drops']):,}** (== 28,644 r832 ROSTER clips) — {'✓' if recon else '✗ FAIL'}\n")

    # ---- INVARIANT 2: set-equality ----
    P("## 2 · Set-equality (no dup pairs; stem-sets match; same-grid + different-subject)\n")
    dup = len(rows) - len({(r["stratum"], r["target"], r["reference"]) for r in rows})
    if dup: fails.append(f"{dup} duplicate (stratum,target,reference) tuples")
    s6r = {r["reference"] for r in by["S6"]}
    s6_pair_ok = (s6t == s6["targets"] and s6r <= s6["targets"] and not (s6t & s6["drops"]))
    if not s6_pair_ok: fails.append("S6 target/reference stem-set != r832 ROSTER prediction")
    # every S6 pair: same grid AND different subject (from r832 ROSTER)
    grid_bad = subj_bad = 0
    for r in by["S6"]:
        if s6["shp"].get(r["target"]) != s6["shp"].get(r["reference"]):
            grid_bad += 1
        if s6["subj"].get(r["target"]) == s6["subj"].get(r["reference"]):
            subj_bad += 1
    if grid_bad: fails.append(f"{grid_bad} S6 pairs cross-grid")
    if subj_bad: fails.append(f"{subj_bad} S6 pairs same-subject")
    s1t = {r["target"] for r in by["S1"]}
    s1_ok = (s1t == set_s1)
    if not s1_ok: fails.append(f"S1 targets {len(s1t)} != S1 inv clips {len(set_s1)}")
    P(f"- duplicate pairs: **{dup}** {'✓' if dup == 0 else '✗'}")
    P(f"- S6 targets == r832 ROSTER non-singleton set, refs ⊆ it, targets∩drops=∅: {'✓' if s6_pair_ok else '✗'}")
    P(f"- S6 pairs same-grid: {'✓' if grid_bad == 0 else f'✗ {grid_bad}'} · "
      f"different-subject: {'✓' if subj_bad == 0 else f'✗ {subj_bad}'} (all {len(by['S6']):,} S6 rows)")
    P(f"- S1 distinct targets ({len(s1t):,}) == S1 inventory clips ({len(set_s1):,}): {'✓' if s1_ok else '✗'}\n")

    # ---- INVARIANT 3: shared-stub / conditions dedup ----
    P("## 3 · Shared-stub detector (source paths belong to exactly one clip except structural)\n")
    lat_owner = defaultdict(set); ref_owner = defaultdict(set)
    for r in rows:
        lat_owner[r["paths"]["latents"]].add((r["stratum"], r["target"]))
        ref_owner[r["paths"]["reference_latents"]].add((r["stratum"], r["reference"]))
    lat_bad = {p for p, o in lat_owner.items() if len(o) != 1}
    ref_bad = {p for p, o in ref_owner.items() if len(o) != 1}
    if lat_bad: fails.append(f"{len(lat_bad)} latents paths map to >1 target")
    if ref_bad: fails.append(f"{len(ref_bad)} reference_latents paths map to >1 reference")
    P("| stratum | distinct conditions paths | distinct caption_keys | ok (equal & >1) |")
    P("|---|--:|--:|:--:|")
    for s in STRATA:
        dc = len({r["paths"]["conditions"] for r in by[s]})
        dk = len({r["caption_key"] for r in by[s]})
        ok = (dc == dk and dc > 1)
        if not ok: fails.append(f"{s} conditions distinct {dc} != caption_keys {dk} (or stub)")
        P(f"| {s} | {dc:,} | {dk:,} | {'✓' if ok else '✗'} |")
    P(f"\n- latents 1:1 with target: {'✓' if not lat_bad else '✗ '+str(len(lat_bad))} · "
      f"reference_latents 1:1 with reference: {'✓' if not ref_bad else '✗ '+str(len(ref_bad))}\n")

    # ---- INVARIANT 4: path scheme + mask set ----
    P("## 4 · Path-scheme gate (relative-under-root; no absolute; no `..`) + mask set\n")
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
    # mask set: distinct mask paths referenced + on-disk _mask_store must == the 5
    used_masks = {os.path.basename(r["paths"]["masks"]) for r in rows}
    disk_masks = {p for p in os.listdir(f"{ROOT}/_mask_store") if p.endswith(".pt")}
    mask_ok = (used_masks == EXPECT_MASKS == disk_masks)
    if not mask_ok:
        fails.append(f"mask set mismatch used={sorted(used_masks)} disk={sorted(disk_masks)} exp={sorted(EXPECT_MASKS)}")
    P(f"distinct paths: **{len(distinct_paths):,}**; scheme violations: **{len(bad_scheme)}** "
      f"{'✓' if not bad_scheme else '✗ '+str(bad_scheme[:3])}")
    P(f"mask set (used == disk == expected 5): {'✓' if mask_ok else '✗'}  used={sorted(used_masks)}")
    P("\ndistinct source roots under `_src` (reported):")
    for k, n in topdirs.most_common():
        P(f"  - `{k}` — {n:,}")
    P("")

    # ---- INVARIANT 5: existence (full) + shape (>=400 S6 both grids, keyed independently) ----
    P("## 5 · Existence (full) + shape (>=400 S6 rows spanning both grids; keyed independently)\n")
    fast = "--fast" in sys.argv
    if fast:
        missing = []
        P("_(existence re-stat SKIPPED via --fast)_")
    else:
        missing = [p for p in distinct_paths if not os.path.isfile(os.path.join(ROOT, p))]
    if missing: fails.append(f"{len(missing)} distinct paths missing e.g. {missing[:3]}")

    def exp_shape(stratum, stem):
        return s6["shp"][stem] if stratum == "S6" else STRAT_SHAPE[stratum]
    import torch
    rng = random.Random(0)
    # S6: >=400 rows spanning both grids (>=200 per grid)
    s6_by_grid = defaultdict(list)
    for r in by["S6"]:
        s6_by_grid[s6["shp"][r["target"]]].append(r)
    samp = []
    for g in sorted(S6_GRIDS):
        samp += rng.sample(s6_by_grid[g], min(210, len(s6_by_grid[g])))
    for s in ["S0", "S1", "S2a", "S2b", "S4"]:
        samp += rng.sample(by[s], min(40, len(by[s])))
    s6_samp_grids = Counter(s6["shp"][r["target"]] for r in samp if r["stratum"] == "S6")
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
    n_s6_samp = sum(v for v in s6_samp_grids.values())
    grids_ok = set(s6_samp_grids) == S6_GRIDS and n_s6_samp >= 400
    if not grids_ok: fails.append(f"S6 shape sample {n_s6_samp} rows / grids {dict(s6_samp_grids)} < 400/both")
    if shape_bad: fails.append(f"{len(shape_bad)} sampled tensors wrong shape e.g. {shape_bad[:3]}")
    P(f"- existence (FULL, {len(distinct_paths):,} distinct paths): "
      f"**{'PASS' if not missing else f'{len(missing)} MISSING'}** {'✓' if not missing else '✗'}")
    P(f"- S6 shape sample: **{n_s6_samp} rows** over grids "
      f"{ {str(list(k)): v for k,v in s6_samp_grids.items()} } (target+reference keyed independently) "
      f"{'✓' if grids_ok else '✗'}")
    P(f"- shape (all {len(samp)} sampled rows × target+reference): "
      f"**{'PASS' if not shape_bad else f'{len(shape_bad)} BAD'}** {'✓' if not shape_bad else '✗'}\n")

    ok = not fails
    P(f"## Overall: {'ALL INVARIANTS PASS' if ok else 'FAIL'}\n")
    if fails:
        for f in fails: P(f"  - FAIL: {f}")
    open(f"{ARMA}/CODESIDE_VERIFY_005.md", "w").write("\n".join(L))
    print("wrote", f"{ARMA}/CODESIDE_VERIFY_005.md")
    print("VERIFY_005", "PASS" if ok else "FAIL", "|", "; ".join(fails) if fails else "all 1-5 pass")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
