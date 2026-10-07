#!/usr/bin/env python
"""metrics v5 blend library — the three-channel transport blend (appearance Mu /
semantic transport D / pixel transport PX), reproducing the construction of
misc/2026-09-02_temporal_dynamics_metric/blend_grid.py from the pinned reference
artifact (scripts/metrics_v5/reference_v5.npz). blend_grid.py stays UNTOUCHED; the
math (RS.population / RS.ecdf_rank_matrix / P.ecdf_vec / dist_concat) is imported,
never reimplemented.

Two entry points:
  build_from_reference(ref)      -> (pop, fpop, ceil)   corpus populations + class ceilings
  score_cache(cache, grid_map, pop, fpop, ceil)  -> (pair_rows, per_gen_rows)   one arm's blend

WEIGHTS is copied verbatim from blend_grid.py; the numbers are validated end-to-end
against blend_grid's BLEND_GRID_{cells,teg}_pergen.jsonl (metrics v5 acceptance A3).
"""
from __future__ import annotations
import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
_MISC = REPO / "misc" / "2026-09-02_temporal_dynamics_metric"
sys.path.insert(0, str(_MISC))
OUT = REPO / "outputs/eval/temporal_dynamics"
STD = REPO / "data/processed/transitions_std121"
PAIR = OUT / "mf_pair_cache"
DIRS_CACHE = OUT / "mf_dirs_cache"

from diffusion.transition_eval import reference_stats as RS   # noqa: E402
import pillars as P                                           # noqa: E402  (P.ecdf_vec)
from run_motion_descriptors import dist_concat                # noqa: E402  (PX distance)

# --- the weight grid, copied verbatim from blend_grid.py (w_Mu, w_D, w_PX) ---
WEIGHTS = collections.OrderedDict([
    ("Look_u (.50/.50/0)", (0.5, 0.5, 0.0)),
    (".33/.33/.33", (1 / 3, 1 / 3, 1 / 3)),
    (".25/.25/.50", (0.25, 0.25, 0.5)),
    (".20/.40/.40", (0.2, 0.4, 0.4)),
    (".20/.30/.50", (0.2, 0.3, 0.5)),
    (".10/.45/.45", (0.1, 0.45, 0.45)),
    ("0/.50/.50", (0.0, 0.5, 0.5)),
    ("0/.60/.40", (0.0, 0.6, 0.4)),
    ("0/1/0 (D only)", (0.0, 1.0, 0.0)),
    ("0/0/1 (PX only)", (0.0, 0.0, 1.0)),
])
HEADLINE = ".33/.33/.33"          # -> the paper's Transport column (bl_333)


# --- corpus populations + ceilings (the construction of blend_grid.main) ------
def fused_sims(raw: dict, pop: dict, fpop: dict) -> dict:
    """blend_grid.main.fused_sims verbatim: rank-similarities of the three raw
    distances vs the corpus populations, weighted sum, re-ranked against the
    same-weight fused corpus population."""
    s = {k: 1.0 - P.ecdf_vec(pop[k], raw[k]) for k in ("Mu", "D", "PX")}
    for k in s:
        s[k][~np.isfinite(s[k])] = 0.0
    return {name: 1.0 - P.ecdf_vec(fpop[name], 1.0 - (a * s["Mu"] + b * s["D"] + c * s["PX"]))
            for name, (a, b, c) in WEIGHTS.items()}


def build_populations(Mu, D, DPX, labels222, nc_Mu, nc_D, nc_PX, nc_pair_cls):
    """(pop, fpop, ceil) exactly as blend_grid.main builds them. ``Mu``/``D`` are
    the 222x222 pillar matrices; ``DPX`` the 222x222 pixel-transport matrix;
    ``nc_*`` the new-class pair distances (corpus_newclass_pairs.npz) with
    ``nc_pair_cls`` the class of each pair (= labels[pairs[p][0]])."""
    M = {"Mu": np.asarray(Mu, float), "D": np.asarray(D, float), "PX": np.asarray(DPX, float)}
    pop = {k: RS.population(v) for k, v in M.items()}
    rank = {k: RS.ecdf_rank_matrix(v) for k, v in M.items()}
    for v in rank.values():
        np.fill_diagonal(v, 0.0)
    by = collections.defaultdict(list)
    for i, c in enumerate(labels222):
        by[c].append(i)
    fpop, ceil = {}, {}
    for name, (a, b, c) in WEIGHTS.items():
        dist = 1.0 - (a * (1 - rank["Mu"]) + b * (1 - rank["D"]) + c * (1 - rank["PX"]))
        np.fill_diagonal(dist, 0.0)
        fpop[name] = RS.population(dist)
        r = RS.ecdf_rank_matrix(dist)
        np.fill_diagonal(r, 0.0)
        ceil[name] = {cl: float((1.0 - r)[np.ix_(ix, ix)][~np.eye(len(ix), dtype=bool)].mean())
                      for cl, ix in by.items() if len(ix) >= 2}
    # new-class ceilings from the precomputed pair distances (no OT re-solve)
    raw_new = {"Mu": np.asarray(nc_Mu, float), "D": np.asarray(nc_D, float), "PX": np.asarray(nc_PX, float)}
    fz_new = fused_sims(raw_new, pop, fpop)
    for name in WEIGHTS:
        acc = collections.defaultdict(list)
        for p_, cls in enumerate(nc_pair_cls):
            acc[cls].append(fz_new[name][p_])
        ceil[name].update({c: float(np.mean(v)) for c, v in acc.items()})
    return pop, fpop, ceil


# --- inputs: original files (for the reference build + A2) and the artifact ----
def _sha16(video) -> str:
    import hashlib
    return hashlib.sha1(str(pathlib.Path(video).resolve()).encode()).hexdigest()[:16]


def _px_raw(video) -> np.ndarray:
    return np.load(DIRS_CACHE / f"px_{_sha16(video)}.npz")["px"]


def _dpx_from_px222(px222_stack: np.ndarray, px_raw_all: dict, keys222: list) -> np.ndarray:
    """The 222x222 PX distance matrix, EXACTLY as blend_grid.main builds DPX:
    z-score every 677-corpus descriptor by the 222-subset mean/std, then
    dist_concat over the 222."""
    P222 = np.stack(px222_stack)
    mu_, sd_ = P222.reshape(-1, 18).mean(0), P222.reshape(-1, 18).std(0) + 1e-9
    zpx = {k: (v - mu_) / sd_ for k, v in px_raw_all.items()}
    n = len(keys222)
    DPX = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            DPX[i, j] = DPX[j, i] = dist_concat(zpx[keys222[i]], zpx[keys222[j]])
    return DPX


def original_inputs() -> dict:
    """The reference inputs read straight from the original files (pillars npz +
    mf_dirs_cache px + corpus_newclass_pairs.npz + corpus manifest)."""
    corpus = json.loads((STD / "corpus_manifest.json").read_text())
    keys677 = sorted(corpus["clips"])
    labels677 = [corpus["clips"][k]["class"] for k in keys677]
    z = np.load(OUT / "pillars_corpus_222__grid.npz", allow_pickle=True)
    keys222 = [str(k) for k in z["keys"]]
    labels222 = [corpus["clips"][k]["class"] for k in keys222]
    px_raw_all = {k: _px_raw(STD / k) for k in keys677}
    px222 = np.stack([px_raw_all[k] for k in keys222]).astype(np.float32)
    DPX = _dpx_from_px222([px_raw_all[k] for k in keys222], px_raw_all, keys222)
    cc = np.load(PAIR / "corpus_newclass_pairs.npz")
    nc_pairs = cc["pairs"]
    nc_pair_cls = [labels677[int(i)] for (i, j) in nc_pairs]
    return dict(keys222=np.array(keys222), labels222=np.array(labels222),
                mu=z["mu"], T=z["T"], Tshape=z["Tshape"], E=z["E"], Mw=z["Mw"],
                Mu=z["Mu"], D=z["D"], px222=px222, DPX=DPX,
                keys677=np.array(keys677), labels677=np.array(labels677),
                nc_pairs=nc_pairs, nc_Mu=cc["Mu"], nc_D=cc["D"], nc_PX=cc["PX"],
                nc_pair_cls=np.array(nc_pair_cls))


def load_reference(path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_from_reference(ref: dict):
    return build_populations(ref["Mu"], ref["D"], ref["DPX"],
                             [str(x) for x in ref["labels222"]],
                             ref["nc_Mu"], ref["nc_D"], ref["nc_PX"],
                             [str(x) for x in ref["nc_pair_cls"]])


# --- pair caches (blend_grid.load_pair_cache verbatim) -----------------------
def load_pair_cache(mode: str, label: str):
    """Newest cache for (mode, arm); a shared/teg request falls back to the arm's
    cells-mode cache (a superset). Returns the npz dict or None."""
    for m in ((mode, "cells") if mode in ("shared", "teg") else (mode,)):
        fs = sorted(PAIR.glob(f"{m}_{label.replace(' ', '_')}_*.npz"), key=lambda p: p.stat().st_mtime)
        if fs:
            z = np.load(fs[-1])
            return {k: z[k] for k in z.files}, fs[-1].name
    return None, None


# --- score one arm (pairs + per-gen) -----------------------------------------
def score_cache(cache: dict, grid_map: dict, pop, fpop, ceil, label: str,
                pairs_mode: str) -> tuple[list, list]:
    """(pair_rows, per_gen_rows) for one arm's pair cache. ``pairs_mode`` names
    the cache mode used (cells|teg) for provenance only; per-gen values are
    order-independent (fused_sims is per-pair) so grouping the cache's own pairs
    by generation reproduces blend_grid exactly (verified: the cache's gen set
    equals blend_grid's post-filter row set)."""
    gens = [str(x) for x in cache["gens"]]
    refs = [str(x) for x in cache["refs"]]
    raw = {"Mu": np.asarray(cache["Mu"], float), "D": np.asarray(cache["D"], float),
           "PX": np.asarray(cache["pxd"], float)}
    fz = fused_sims(raw, pop, fpop)

    def meta_of(gen: str) -> dict:
        item_id = gen.rpartition("__s")[0]
        seed = int(gen.rpartition("__s")[2])
        g = grid_map[item_id]
        return dict(item_id=item_id, seed=seed, harness_arm=item_id.split("__")[1],
                    cls=g["gt_pool_class"], cell=g["cell"], sided=g.get("sided", "one"),
                    tier=g.get("ref_novelty"), ed=bool(g["endpoint"].startswith("ed.")))

    pair_rows = []
    for i, gen in enumerate(gens):
        m = meta_of(gen)
        pair_rows.append(dict(item_id=m["item_id"], seed=m["seed"], harness_arm=m["harness_arm"],
                              ref=refs[i], cls=m["cls"], cell=m["cell"], sided=m["sided"],
                              tier=m["tier"], T=float(cache["T"][i]), Tshape=float(cache["Tshape"][i]),
                              E=float(cache["E"][i]), Mw=float(cache["Mw"][i]), Mu=float(cache["Mu"][i]),
                              D=float(cache["D"][i]), fid=float(cache["fid"][i]), pxd=float(cache["pxd"][i])))

    byg = collections.defaultdict(list)
    for i, gen in enumerate(gens):
        byg[gen].append(i)
    per_gen = []
    for gen, ix in byg.items():
        m = meta_of(gen)
        rec = dict(item_id=m["item_id"], seed=m["seed"], harness_arm=m["harness_arm"], label=label,
                   cls=m["cls"], cell=m["cell"], ed=m["ed"], n_pairs=len(ix))
        ixa = np.array(ix)
        for name in WEIGHTS:
            sv = fz[name][ixa]
            v = float(np.nanmean(sv)) if np.isfinite(sv).any() else None
            c = ceil[name].get(m["cls"])
            rec[name] = (100.0 * min(v / c, 1.0)) if (v is not None and c) else None
        # headline alias + uncapped for the paper table (bl_333)
        vh = float(np.nanmean(fz[HEADLINE][ixa])) if np.isfinite(fz[HEADLINE][ixa]).any() else None
        ch = ceil[HEADLINE].get(m["cls"])
        rec["bl_333"] = rec[HEADLINE]
        rec["bl_333_uncapped"] = (100.0 * vh / ch) if (vh is not None and ch) else None
        per_gen.append(rec)
    return pair_rows, per_gen
