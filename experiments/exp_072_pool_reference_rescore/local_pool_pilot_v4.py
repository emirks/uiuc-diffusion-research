"""Local read-only pool-lane pilot under the v4.0.0 instrument (S3 kernel +
reference_v4 artifact), imported from the eval-v4-cert worktree. Cache-read only.
Validates against existing ladder_v4h rows first.
"""
import collections
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

CACHE = REPO / "outputs/eval/cache"
MODEL, SHORT = "facebook/dinov2-base", "256"
corpus = json.load(open(REPO / "data/processed/transitions_std121/corpus_manifest.json"))
SIDE = {c: v["sidedness"] for c, v in corpus["classes"].items()}
REF_STATS = load_reference()

_cache = {}


def bundle_of(path, style):
    path = pathlib.Path(path if str(path).startswith("/") else REPO / path)
    if path in _cache:
        return _cache[path]
    f = feature_cache_path(file_key(path, MODEL, SHORT), CACHE)
    if not f.exists():
        _cache[path] = None
        return None
    feats = np.load(f)["feats"]
    prof = morph_profile(feats)
    core, _ = core_mask_v3(prof, SIDE[style])
    out = (feats, core, prof["n_prefix"], prof["n_suffix"])
    _cache[path] = out
    return out


def app_v4(gen_path, ref_path, style):
    g, r = bundle_of(gen_path, style), bundle_of(ref_path, style)
    if g is None or r is None:
        return None
    return float(appearance_s3(g[0], g[1], g[2], g[3], r[0], r[1], r[2], r[3],
                               REF_STATS)["app_ref"])


print("== validation vs existing v4 sweep rows (ladder_v4h) ==")
checked = 0
for lbl, mani in [("ic3_abc_c0", "eval_ic3_abc_c0.json"), ("r2r3_c0", "eval_r2r3_c0.json")]:
    cert = {}
    p = REPO / f"outputs/eval/ladder_v4h/{lbl}/items.jsonl"
    if not p.exists():
        continue
    for line in open(p):
        r = json.loads(line)
        if not r["arm"].startswith("control"):
            cert[r["item_id"]] = r.get("app_ref")
    items = json.load(open(REPO / "experiments/exp_066_ladder_v3_scoring/dataset" / mani))
    for it in items[:4]:
        mine = app_v4(it["generated_video"], it["reference_video"], it["style"])
        ref = cert.get(it["item_id"])
        if mine is None or ref is None:
            continue
        ok = "OK" if abs(mine - ref) < 1e-4 else "MISMATCH"
        print(f"  {it['item_id'][:50]:52s} v4-sweep {ref:.6f}  local {mine:.6f}  {ok}")
        checked += 1
print(f"  ({checked} rows checked)")

rows = json.load(open(REPO / "experiments/exp_072_pool_reference_rescore/dataset/eval_pool_pilot.json"))
per_item = collections.defaultdict(list)
miss = 0
for r in rows:
    v = app_v4(r["generated_video"], r["reference_video"], r["style"])
    if v is None:
        miss += 1
        continue
    per_item[(r["arm"], r["style"], r["item_id"].split("__ref_")[0])].append(v)

z = np.load(WT / "outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz",
            allow_pickle=True)
keys = [str(k) for k in z["keys"]]
S = 1.0 - z["m1a_S3"]
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
print(f"\n== v4 PILOT: pool-mean app_ref (% of v4 GT ceiling), {len(rows)-miss}/{len(rows)} pairs ==")
print(f"{'':18s}" + "".join(f"{NAME[a]:>19s}" for a in ORDER))
for c in sorted({s for (_a, s) in per_cls}):
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
