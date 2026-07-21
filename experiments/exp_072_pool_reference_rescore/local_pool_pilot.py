"""Local read-only pool-lane pilot: exact v3 kernel (worktree code) on cached DINO
features. No cache writes, no GPU, no model — rows with cache misses are skipped.
Validates itself first by reproducing certified app_ref values on original rows.
"""
import collections
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
WT = REPO / ".claude/worktrees/eval-v3.0.0"
sys.path.insert(0, str(WT / "src"))
from diffusion.transition_eval.features import file_key, feature_cache_path  # noqa: E402
from diffusion.transition_eval.m1_transfer import appearance_ref  # noqa: E402
from diffusion.transition_eval.morph import morph_profile  # noqa: E402
from diffusion.transition_eval.s_structure import core_mask_v3  # noqa: E402

CACHE = REPO / "outputs/eval/cache"
MODEL, SHORT = "facebook/dinov2-base", "256"
corpus = json.load(open(REPO / "data/processed/transitions_std121/corpus_manifest.json"))
SIDE = {c: v["sidedness"] for c, v in corpus["classes"].items()}

_feat_cache = {}


def feats_of(path):
    path = pathlib.Path(path if str(path).startswith("/") else REPO / path)
    if path in _feat_cache:
        return _feat_cache[path]
    f = feature_cache_path(file_key(path, MODEL, SHORT), CACHE)
    out = np.load(f)["feats"] if f.exists() else None
    _feat_cache[path] = out
    return out


def app_ref(gen_path, ref_path, style):
    gf, rf = feats_of(gen_path), feats_of(ref_path)
    if gf is None or rf is None:
        return None
    side = SIDE[style]
    gcore, _ = core_mask_v3(morph_profile(gf), side)
    rcore, _ = core_mask_v3(morph_profile(rf), side)
    return float(appearance_ref(gf, gcore, rf, rcore))


# ---- 1. validation: reproduce certified app_ref on original rows -------------
print("== validation vs certified rows ==")
checked = 0
for lbl, mani in [("ic3_abc_c0", "eval_ic3_abc_c0.json"), ("r2r3_c0", "eval_r2r3_c0.json"),
                  ("base_c0", "eval_base_c0.json")]:
    cert = {}
    for line in open(REPO / f"outputs/eval/ladder_v3/{lbl}/items.jsonl"):
        r = json.loads(line)
        if not r["arm"].startswith("control"):
            cert[r["item_id"]] = r.get("app_ref")
    items = json.load(open(REPO / "experiments/exp_066_ladder_v3_scoring/dataset" / mani))
    for it in items[:4]:
        mine = app_ref(it["generated_video"], it["reference_video"], it["style"])
        ref = cert.get(it["item_id"])
        if mine is None or ref is None:
            continue
        print(f"  {it['item_id'][:52]:54s} certified {ref:.6f}  local {mine:.6f}  "
              f"{'OK' if abs(mine-ref) < 1e-4 else 'MISMATCH'}")
        checked += 1
print(f"  ({checked} rows checked)")

# ---- 2. pilot: pool rows ------------------------------------------------------
rows = json.load(open(REPO / "experiments/exp_072_pool_reference_rescore/dataset/eval_pool_pilot.json"))
per_item = collections.defaultdict(list)
miss = 0
for r in rows:
    v = app_ref(r["generated_video"], r["reference_video"], r["style"])
    if v is None:
        miss += 1
        continue
    per_item[(r["arm"], r["style"], r["item_id"].split("__ref_")[0])].append(v)

z = np.load(REPO / "outputs/eval/certification/3.0.0-draft.8/analysis/distance_matrices.npz",
            allow_pickle=True)
keys = [str(k) for k in z["keys"]]
S = 1.0 - z["m1a__v3_sided"]
cls = [k.split("/")[0] for k in keys]
byc = collections.defaultdict(list)
for i, c in enumerate(cls):
    byc[c].append(i)
ceil = {c: S[np.ix_(np.array(i), np.array(i))][~np.eye(len(i), dtype=bool)].mean()
        for c, i in byc.items() if len(i) >= 2}

per_cls = collections.defaultdict(list)
for (arm, style, _src), vals in per_item.items():
    per_cls[(arm, style)].append(st.mean(vals))
ORDER = ["r1", "r2_ckpt2000", "r3_ckpt2000", "ic3_a", "ic3_b"]
NAME = {"r1": "base+endpoints", "r2_ckpt2000": "specialist SEEN", "r3_ckpt2000": "specialist UNSEEN",
        "ic3_a": "ic3 held-in", "ic3_b": "ic3 unseen"}
classes = sorted({s for (_a, s) in per_cls})
print(f"\n== PILOT: pool-mean app_ref (% of GT ceiling), {len(rows)-miss}/{len(rows)} pairs, cache-miss {miss} ==")
print(f"{'':18s}" + "".join(f"{NAME[a]:>19s}" for a in ORDER))
for c in classes:
    line = f"{c:18s}"
    for a in ORDER:
        v = per_cls.get((a, c))
        line += f"{st.mean(v):>7.3f} ({st.mean(v)/ceil[c]:>4.0%})" if v else f"{'—':>19s}"
    print(line + f"   ceiling {ceil[c]:.3f}")
print("\nmean achieved-% across pilot classes:")
for a in ORDER:
    fr = [st.mean(v) / ceil[c] for (arm, c), v in per_cls.items() if arm == a]
    if fr:
        print(f"  {NAME[a]:18s} {st.mean(fr):>5.0%}")
