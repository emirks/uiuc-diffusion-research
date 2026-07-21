"""exp_072 — local exact-kernel pool fill for arms the harness lane hasn't covered.

Computes v4 pool-mean app_ref (S3 kernel, worktree code + reference_v4 artifact)
from CACHED DINOv2 features only (CPU, read-only; features are instrument-shared).
Covers: constructed pools for r1k / r1k_ext / ic2_r4 / ic2_r5 / r2_ckpt250 /
r3_ckpt250 (never planned in the harness lane) + the queued fill manifests
eval_pool_x0/x1.json (r0, r3x, ic3_x — harness jobs will confirm these later).

Validates first against existing harness pool rows (must match < 1e-4), then
writes a single viewer-facing index:
    outputs/eval/exp_072_pool_v4/local_fill/pool_index.json
      { "ceilings": {class: ceiling}, "matrix": "m1a_S3",
        "items": {src_item_id: {arm, style, pool, n, src}},   # src: harness|local
        "missing": {arm: n_pairs_without_cached_features} }

Run with the diffusion env python (numpy):
    $LAB/envs/diffusion/bin/python experiments/exp_072_pool_reference_rescore/local_pool_fill.py
"""
import collections
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
WT = REPO / ".claude/worktrees/eval-v4-cert"
sys.path.insert(0, str(WT / "src"))
from diffusion.transition_eval.features import file_key, feature_cache_path  # noqa: E402
from diffusion.transition_eval.m1_transfer import appearance_s3  # noqa: E402
from diffusion.transition_eval.morph import morph_profile  # noqa: E402
from diffusion.transition_eval.reference_stats import load_reference  # noqa: E402
from diffusion.transition_eval.s_structure import core_mask_v3  # noqa: E402

DS66 = REPO / "experiments/exp_066_ladder_v3_scoring/dataset"
DS72 = REPO / "experiments/exp_072_pool_reference_rescore/dataset"
OUT = REPO / "outputs/eval/exp_072_pool_v4/local_fill"
MODEL, SHORT = "facebook/dinov2-base", "256"
MAX_REFS = 8
# caches searched in order; features are stat-keyed so any dir that has the file wins
CACHES = ([REPO / "outputs/eval/cache", WT / "outputs/eval/cache"]
          + sorted(REPO.glob("outputs/eval/cache_v4_*"))
          + sorted(REPO.glob("outputs/eval/cache_alt*")))
CONSTRUCT_ARMS = {"r1k", "r1k_ext", "ic2_r4", "ic2_r5", "r2_ckpt250", "r3_ckpt250"}
FILL_MANIFESTS = ["eval_pool_x0.json", "eval_pool_x1.json"]  # r0, r3x, ic3_x (queued)

corpus = json.load(open(REPO / "data/processed/transitions_std121/corpus_manifest.json"))
SIDE = {c: v["sidedness"] for c, v in corpus["classes"].items()}
REF_STATS = load_reference()
_cache = {}


def bundle_of(path, style):
    path = pathlib.Path(path if str(path).startswith("/") else REPO / path)
    ck = (path, style)
    if ck in _cache:
        return _cache[ck]
    key = file_key(path, MODEL, SHORT)
    feats = None
    for cdir in CACHES:
        f = feature_cache_path(key, cdir)
        if f.exists():
            feats = np.load(f)["feats"]
            break
    if feats is None:
        _cache[ck] = None
        return None
    prof = morph_profile(feats)
    core, _ = core_mask_v3(prof, SIDE[style])
    out = (feats, core, prof["n_prefix"], prof["n_suffix"])
    _cache[ck] = out
    return out


def app_v4(gen_path, ref_path, style):
    g, r = bundle_of(gen_path, style), bundle_of(ref_path, style)
    if g is None or r is None:
        return None
    return float(appearance_s3(g[0], g[1], g[2], g[3], r[0], r[1], r[2], r[3],
                               REF_STATS)["app_ref"])


def pool_rows(it):
    """Same rules as build_manifests.pool_rows: same-class pool, minus own clip,
    minus the item's original reference, deterministic first-8 by clip name."""
    byc = pool_rows.byc
    style = it["style"]
    own_clip = it["item_id"].split("__")[2]
    orig_ref = pathlib.Path(it.get("reference_video") or "").stem
    rows = []
    for key in byc.get(style, []):
        stem = pathlib.Path(key).stem
        if stem in (own_clip, orig_ref):
            continue
        rows.append((f"data/processed/transitions_std121/{key}", stem))
        if len(rows) >= MAX_REFS:
            break
    return rows


pool_rows.byc = {}
for key, meta in corpus["clips"].items():
    pool_rows.byc.setdefault(meta["class"], []).append(key)
for v in pool_rows.byc.values():
    v.sort()


def main():
    # ---- 0) validation vs harness pool rows -------------------------------
    print("== validation vs harness pool rows (exp_072_pool_v4/pool_c*) ==")
    val, bad = 0, 0
    seen_arms = set()
    for f in sorted(glob.glob(str(REPO / "outputs/eval/exp_072_pool_v4/pool_c*/items.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["arm"].startswith("control") or r.get("app_ref") is None:
                continue
            if r["arm"] in seen_arms:
                continue
            seen_arms.add(r["arm"])
            gen = r.get("generated_video")
            ref = r.get("reference_video")
            if not gen or not ref:  # items.jsonl may not carry paths; fall back to manifest
                continue
            mine = app_v4(gen, ref, r["style"])
            if mine is None:
                continue
            ok = abs(mine - r["app_ref"]) < 1e-4
            bad += (not ok)
            val += 1
            print(f"  {r['item_id'][:58]:60s} harness {r['app_ref']:.6f} local {mine:.6f} "
                  f"{'OK' if ok else 'MISMATCH'}")
    if val == 0:  # rows don't embed paths -> validate via the chunk manifests
        harness = {}
        for f in sorted(glob.glob(str(REPO / "outputs/eval/exp_072_pool_v4/pool_c*/items.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                if not r["arm"].startswith("control") and r.get("app_ref") is not None:
                    harness[r["item_id"]] = (r["arm"], r["style"], r["app_ref"])
        per_arm = collections.Counter()
        for cf in sorted(DS72.glob("eval_pool_c*.json")):
            for it in json.load(open(cf)):
                iid = it["item_id"]
                if iid not in harness or per_arm[harness[iid][0]] >= 2:
                    continue
                mine = app_v4(it["generated_video"], it["reference_video"], it["style"])
                if mine is None:
                    continue
                ok = abs(mine - harness[iid][2]) < 1e-4
                bad += (not ok)
                val += 1
                per_arm[harness[iid][0]] += 1
                print(f"  {iid[:58]:60s} harness {harness[iid][2]:.6f} local {mine:.6f} "
                      f"{'OK' if ok else 'MISMATCH'}")
    print(f"  ({val} rows checked, {bad} mismatches)")
    if bad:
        sys.exit("VALIDATION FAILED — do not trust local fill")

    # ---- 1) harness per-item pool means -----------------------------------
    items_out = {}
    per_item = collections.defaultdict(list)
    meta_of = {}
    for f in sorted(glob.glob(str(REPO / "outputs/eval/exp_072_pool_v4/pool_c*/items.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["arm"].startswith("control") or r.get("app_ref") is None:
                continue
            src = r["item_id"].split("__ref_")[0]
            per_item[src].append(r["app_ref"])
            meta_of[src] = (r["arm"], r["style"])
    for src, vals in per_item.items():
        arm, style = meta_of[src]
        items_out[src] = {"arm": arm, "style": style, "pool": st.mean(vals),
                          "n": len(vals), "src": "harness"}
    print(f"[harness] {len(items_out)} src items from pool_c*")

    # ---- 2) local pairs: constructed + queued fill manifests ---------------
    pairs = []  # (src_item_id, arm, style, gen, ref_path)
    seen_ids = set()
    for f in sorted(DS66.glob("eval_*.json")):
        for it in json.load(open(f)):
            if it["arm"] not in CONSTRUCT_ARMS or it["item_id"] in seen_ids:
                continue
            seen_ids.add(it["item_id"])
            for ref_path, _stem in pool_rows(it):
                pairs.append((it["item_id"], it["arm"], it["style"],
                              it["generated_video"], ref_path))
    for fn in FILL_MANIFESTS:
        p = DS72 / fn
        if not p.exists():
            continue
        for it in json.load(open(p)):
            src = it["item_id"].split("__ref_")[0]
            pairs.append((src, it["arm"], it["style"],
                          it["generated_video"], it["reference_video"]))
    print(f"[local] {len(pairs)} pairs to score")

    loc = collections.defaultdict(list)
    loc_meta = {}
    missing = collections.Counter()
    for i, (src, arm, style, gen, ref) in enumerate(pairs):
        if i and i % 500 == 0:
            print(f"  ... {i}/{len(pairs)}")
        v = app_v4(gen, ref, style)
        if v is None:
            missing[arm] += 1
            continue
        loc[src].append(v)
        loc_meta[src] = (arm, style)
    for src, vals in loc.items():
        if src in items_out:  # harness wins
            continue
        arm, style = loc_meta[src]
        items_out[src] = {"arm": arm, "style": style, "pool": st.mean(vals),
                          "n": len(vals), "src": "local"}
    print(f"[local] scored items: {len(loc)}; pairs without cached features: {dict(missing)}")

    # ---- 3) ceilings (v4 certified pairwise matrix) ------------------------
    z = np.load(WT / "outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz",
                allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    S = 1.0 - z["m1a_S3"]
    byc = collections.defaultdict(list)
    for i, c in enumerate(k.split("/")[0] for k in keys):
        byc[c].append(i)
    ceil = {c: float(S[np.ix_(np.array(ix), np.array(ix))][~np.eye(len(ix), dtype=bool)].mean())
            for c, ix in byc.items() if len(ix) >= 2}

    OUT.mkdir(parents=True, exist_ok=True)
    out = {"matrix": "m1a_S3", "instrument": "transition-eval/4.0.0",
           "ceilings": ceil, "items": items_out,
           "missing_pairs": dict(missing)}
    json.dump(out, open(OUT / "pool_index.json", "w"), indent=1)
    arms = collections.Counter(v["arm"] for v in items_out.values())
    print(f"[done] {OUT/'pool_index.json'}  items={len(items_out)}  ceilings={len(ceil)}")
    print("       items per arm:", dict(sorted(arms.items())))


if __name__ == "__main__":
    main()
