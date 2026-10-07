#!/usr/bin/env python
"""metrics v5 driver — from the store to the paper's metric table (CPU only).

Subcommands (each idempotent / skip-if-done):
  import-trackdesc  D1  fill trackdesc@cotracker3-s64-v1 for the roster gens + corpus + conds
                        (import the mf_dirs_cache dirs/px, else compute from the stored tracks)
  reference         D2  build scripts/metrics_v5/reference_v5.npz + versioning.py; verify (A2)
  blend             D3  eval 047 transport v5 (pairs + per-gen) from the imported pair caches
  pairs             D8  compute Mu/D/PX for roster arms WITHOUT a pair cache (sweep arms + neutral
                        twins) from stored features and ADD them to eval 047 (never re-imports)
  seam              D4  eval 048 seam z from the stored temporal LPIPS at the physical windows
  flow-action       R4  eval 049 Flow MSE + Action KL from the two whole-video namespaces
                        (flow_u32@raft-r256-g24x32 + action@swin3db-k400-u32) vs the reference clip
  pixel-endpoints   R6  eval 050 PSNR/SSIM/LPIPS of the given endpoint frames (pixels, PyAV rgb24);
                        shardable/resumable (--shard i/n, --device cuda|cpu, per-arm locks), --wait/--finalize
  finalize          D5  re-run the four draft evals with numbered ids 043-046 (--no-index)
  tables            D6  rebuild the metric-family tables (roster-driven, Transport = the blend)
  paper-table       D10 render the paper-style main table (tab_main.tex + preview.pdf) from the
                        same family_tables levels; A10 = paper_table_check.py
  paper-tables      D3(R3) fill the paper's five result tables (tab_main / tab_ablation /
                        tab_ablationtiers / tab_weight / tab_isolation); --apply writes them in + builds
  status                fast per-roster-arm namespace coverage + per-eval (043-048) presence (D9)

LAB=/taiga/.../emirkisa; PY=$LAB/envs-aarch64/ltx2/bin/python; PYTHONPATH=<repo>/src;
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PILLAR_JOBS=1; HF_HUB_OFFLINE=1. ONE CPU-heavy process at a time.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util as ilu
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
HERE = REPO / "scripts" / "metrics_v5"
ROSTER = HERE / "roster.json"
REFERENCE = HERE / "reference_v5.npz"
VERSIONING = HERE / "versioning.py"
LOGDIR = REPO / "misc/2026-09-23_metrics_v5/logs"
OUT = REPO / "outputs/eval/temporal_dynamics"
DIRS_CACHE = OUT / "mf_dirs_cache"
STD = REPO / "data/processed/transitions_std121"
CONDS = REPO / "eval_ladder/conds"
CONDS16 = REPO / "misc/2026-09-20_teg_baselines/conds_16fps"
TRACKDESC = "trackdesc@cotracker3-s64-v1"
COTRACKER = "cotracker3@g20-m384-v2"


def _load_feature_store():
    path = REPO / "src" / "diffusion" / "feature_store.py"
    spec = ilu.spec_from_file_location("v5_feature_store", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FS_MOD = _load_feature_store()
FeatureStore = FS_MOD.FeatureStore


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def roster() -> dict:
    return json.loads(ROSTER.read_text())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _sha16(video) -> str:
    return hashlib.sha1(str(Path(video).resolve()).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- D1 import-trackdesc
def _import_targets():
    """[(label, video_dir, variant_dir|None, videos)] over roster gens + corpus + conds + conds16."""
    r = roster()
    targets = []
    seen = set()
    for arm in r["arms"]:
        for g in arm["gens"]:
            if g in seen:
                continue
            seen.add(g)
            vdir = REPO / g
            vids = sorted((vdir / "videos").glob("*.mp4"))
            targets.append((g, vdir / "videos", vdir, vids))
    for cdir in sorted({p.parent for p in STD.glob("*/*.mp4")}):
        targets.append((f"corpus/{cdir.name}", cdir, None, sorted(cdir.glob("*.mp4"))))
    targets.append(("conds", CONDS, None, sorted(CONDS.glob("*.mp4"))))
    targets.append(("conds_16fps", CONDS16, None, sorted(CONDS16.glob("*.mp4"))))
    return targets


def cmd_import_trackdesc(args):
    fs = FeatureStore(REPO)
    sha = git_sha()
    extractor = None
    tot = dict(covered=0, present=0, imported=0, computed=0, skip_no_ct=0, failures=0)
    bad = []
    for label, vdir, variant_dir, vids in _import_targets():
        n_imp = n_cmp = n_skip = n_pres = 0
        for v in vids:
            if fs.has(v, TRACKDESC):
                n_pres += 1
                continue
            k = _sha16(v)
            dp, pp = DIRS_CACHE / f"dirs_{k}.npz", DIRS_CACHE / f"px_{k}.npz"
            vsha = fs.sha_from_sums(v)
            if vsha is None and fs.has(v, COTRACKER):
                try:
                    vsha = fs.read_meta(v, COTRACKER).get("video_sha256")
                except Exception:
                    vsha = None
            try:
                if dp.exists() and pp.exists():
                    arrays = {"dirs": np.load(dp)["dirs"].astype(np.float32),
                              "px": np.load(pp)["px"].astype(np.float32)}
                    fs.put(v, TRACKDESC, arrays,
                           {"host": None, "code_sha": sha, "video_sha256": vsha,
                            "origin": f"migrated:mf_dirs_cache/dirs_{k}.npz+px_{k}.npz",
                            "source_ns": COTRACKER})
                    n_imp += 1
                elif fs.has(v, COTRACKER):
                    if extractor is None:
                        from diffusion.feature_extractors import REGISTRY
                        extractor = REGISTRY[TRACKDESC]("cpu")
                    arrays = extractor.extract(v)
                    fs.put(v, TRACKDESC, arrays,
                           {"host": None, "code_sha": sha, "video_sha256": vsha,
                            "origin": "extracted", "source_ns": COTRACKER})
                    n_cmp += 1
                else:
                    n_skip += 1
            except Exception as e:  # noqa: BLE001
                tot["failures"] += 1
                bad.append(f"{v}: {type(e).__name__}: {e}")
        touched = n_imp + n_cmp
        if touched:
            fs.rebuild_manifest(vdir)
            if variant_dir is not None and (variant_dir / "meta.yaml").exists():
                fs.write_meta_block(variant_dir)
        tot["covered"] += len(vids); tot["present"] += n_pres; tot["imported"] += n_imp
        tot["computed"] += n_cmp; tot["skip_no_ct"] += n_skip
        log(f"[import] {label:<48} videos={len(vids)} present={n_pres} imported={n_imp} computed={n_cmp} skip(no cotracker3)={n_skip}")
    log(f"[import DONE] {tot}")
    for b in bad[:40]:
        log("  BAD:", b)
    return 0


# ---------------------------------------------------------------- D2 reference
def cmd_reference(args):
    import blend
    log("building reference inputs from the original files ...")
    inp = blend.original_inputs()
    if REFERENCE.exists() and not args.force:
        log(f"[reference] {REFERENCE} exists; verifying only (use --force to rebuild)")
    else:
        payload = {k: inp[k] for k in ("keys222", "labels222", "mu", "T", "Tshape", "E", "Mw",
                                       "Mu", "D", "px222", "DPX", "keys677", "labels677",
                                       "nc_pairs", "nc_Mu", "nc_D", "nc_PX", "nc_pair_cls")}
        tmp = REFERENCE.with_suffix(f".tmp{time.time_ns()}.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(REFERENCE)
        log(f"[reference] wrote {REFERENCE} ({REFERENCE.stat().st_size} bytes)")
    sha256 = FS_MOD.sha256_file(REFERENCE)
    VERSIONING.write_text(
        '"""metrics v5 — pinned reference artifact hash (scripts/metrics_v5/reference_v5.npz).\n'
        'Bundled by scripts/metrics_v5/v5.py reference; recorded in every v5 eval meta.\n"""\n'
        f'REFERENCE_V5_SHA256 = "{sha256}"\n')
    log(f"[reference] REFERENCE_V5_SHA256 = {sha256}")
    # ---- A2: build pop/fpop/ceil from the original files and from the artifact; compare
    log("A2: build_populations over original files ...")
    po, fo, co = blend.build_populations(inp["Mu"], inp["D"], inp["DPX"],
                                         [str(x) for x in inp["labels222"]],
                                         inp["nc_Mu"], inp["nc_D"], inp["nc_PX"],
                                         [str(x) for x in inp["nc_pair_cls"]])
    log("A2: build_populations over the artifact ...")
    pa, fa, ca = blend.build_from_reference(blend.load_reference(REFERENCE))
    pop_diff = {k: float(np.max(np.abs(po[k] - pa[k]))) for k in po}
    fpop_diff = max(float(np.max(np.abs(fo[n] - fa[n]))) for n in fo)
    ceil_diff = 0.0
    for n in co:
        for cl in co[n]:
            ceil_diff = max(ceil_diff, abs(co[n][cl] - ca[n][cl]))
    log(f"[A2] pop max|diff| Mu={pop_diff['Mu']:.3e} D={pop_diff['D']:.3e} PX={pop_diff['PX']:.3e}")
    log(f"[A2] fpop max|diff|={fpop_diff:.3e}  ceil max|diff|={ceil_diff:.3e}")
    log(f"[A2] n classes with ceilings (.33/.33/.33) = {len(co['.33/.33/.33'])}")
    ok = max(pop_diff.values()) == 0.0 and ceil_diff < 1e-12 and fpop_diff < 1e-12
    log(f"[A2] PASS={ok}")
    return 0 if ok else 1


# ---------------------------------------------------------------- D3 blend (eval 047)
def _grid_map(gens_rel):
    """item_id -> grid row, over a list of gens variant dirs."""
    from store_eval_common import grid_of
    m = {}
    for g in gens_rel:
        vdir = REPO / g
        if (vdir / "grid.jsonl").exists():
            m.update(grid_of(vdir))
    return m


def cmd_blend(args):
    import blend
    from store_eval_common import atomic_write
    if not REFERENCE.exists():
        log("reference_v5.npz missing; run `reference` first"); return 2
    import score_v3_cells as CELLS   # authoritative label -> variant-dir parts (imported, not modified)
    pop, fpop, ceil = blend.build_from_reference(blend.load_reference(REFERENCE))
    from versioning import REFERENCE_V5_SHA256
    log(f"reference sha {REFERENCE_V5_SHA256}; ceilings built")

    # own + author-native from the cells caches; TEG from the teg (cells_*_TEG_*) caches
    jobs = []  # (label, cache_mode, parts)
    for grp in ("ours_neutral", "ours_effect", "ext"):
        for label, parts in CELLS.ARMS[grp]:
            jobs.append((label, "cells", parts))
    for label, parts in CELLS.ARMS["teg"]:
        jobs.append((label, "teg", parts))

    eval_id = args.eval_id
    ed = REPO / "store" / "evals" / eval_id
    arms_out = {}   # harness_arm -> {pairs, per_gen, gen}
    labels_done, labels_missing = [], []
    for label, mode, parts in jobs:
        cache, fname = blend.load_pair_cache(mode, label)
        if cache is None:
            labels_missing.append(label); log(f"[blend] {label}: NO pair cache — skipped"); continue
        gm = _grid_map([f"store/gens/{subrel}" for subrel, _ in parts])
        pair_rows, per_gen = blend.score_cache(cache, gm, pop, fpop, ceil, label, mode)
        # split by harness_arm (a cells cache of one label can hold v3 + ED gens)
        by_arm_p, by_arm_g = {}, {}
        for r in pair_rows:
            by_arm_p.setdefault(r["harness_arm"], []).append(r)
        for r in per_gen:
            by_arm_g.setdefault(r["harness_arm"], []).append(r)
        for ha in by_arm_g:
            gen = _gen_for_harness(parts, ha)
            arms_out[ha] = dict(pairs=by_arm_p.get(ha, []), per_gen=by_arm_g[ha], gen=gen,
                                label=label, cache=fname, mode=mode)
        labels_done.append(label)
        log(f"[blend] {label:<26} ({mode}, {fname}) -> {len(by_arm_g)} harness arms, "
            f"{len(pair_rows)} pairs, {len(per_gen)} gens")

    ed.mkdir(parents=True, exist_ok=True)
    for ha, d in arms_out.items():
        (ed / ha).mkdir(exist_ok=True)
        atomic_write(ed / ha / "pairs.jsonl", "".join(json.dumps(r) + "\n" for r in d["pairs"]))
        atomic_write(ed / ha / "per_gen.jsonl", "".join(json.dumps(r) + "\n" for r in d["per_gen"]))
    _write_blend_meta(ed, eval_id, arms_out, labels_missing, REFERENCE_V5_SHA256, args.date)
    log(f"[blend] wrote {ed}  ({len(arms_out)} harness arms)")
    return 0


def _gen_for_harness(parts, harness_arm):
    for subrel, stamp in parts:
        if stamp == harness_arm:
            return f"store/gens/{subrel}"
    return f"store/gens/{parts[0][0]}"


def _write_blend_meta(ed, eval_id, arms_out, labels_missing, ref_sha, created):
    from store_eval_common import atomic_write
    seq = int(eval_id.split("_", 1)[0])
    defin = [
        "Transport v5 = the three-channel blend, the Look_u rule generalised: per (gen, pool-ref) pair the rank-similarities s_X = 1 - ecdf_pop_X(raw_X) of appearance Mu, semantic transport D and pixel transport PX vs the 222-corpus populations;",
        "fused sim = 1 - ecdf_fusedpop( 1 - (w_Mu*s_Mu + w_D*s_D + w_PX*s_PX) ) re-ranked against the same-weight fused corpus population; within-class ceilings under the same weights (222 pin + new-class over the 677 corpus).",
        "Per generation: capped pooled-% = 100*min(mean_over_pairs(fused_sim)/ceiling[class], 1). bl_333 = the .33/.33/.33 weights (headline Transport); the other 9 WEIGHTS columns are named as in blend_grid.py.",
        "Raw per-pair distances (Mu, D, pxd, plus the 6 pillar distances T/Tshape/E/Mw and the MF fidelity fid) are IMPORTED from outputs/eval/temporal_dynamics/mf_pair_cache (score_v3_mf.py); no recomputation. TEG arms take their two-sided zero-shot rows (teg mode); own arms take every cell (cells mode).",
    ]
    L = [f"id: {eval_id}", f"seq: {seq}", "shelf: evals", f"created: '{created}'",
         "machine: dai (login CPU, numpy over stored features + imported pair caches)",
         f"instrument: scripts/metrics_v5/v5.py @ {git_sha()}",
         f"reference_v5_sha256: {ref_sha}",
         'headline: {column: bl_333, weights: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]}',
         "definition:"]
    L += [f"  - {json.dumps(d)}" for d in defin]
    L += [f'why: {json.dumps("The paper Transport column = the equal-thirds three-channel blend (appearance / semantic transport / pixel transport), % of the class ceiling; replaces the v4 transport_pct.")}',
          f'caveat: {json.dumps("Frame-count caveat: ours 121/81 frames vs the prior works 33-49 frames; pools = evals/028s 677-corpus references and 222-based ceilings. Round-1 arms only (arms with a pair cache today); the sweep arms and the six neutral twins have no pair cache yet (arms_pending).")}',
          f"arms_pending: {json.dumps(['dualforce_dcg_w1p5_neutral_v3','dualforce_dcg_w1p5_neutral_v3ed81','dualforce_dcg_w3_neutral_v3','dualforce_dcg_w3_neutral_v3ed81','vap_neutral_v3','vfxmaster_neutral_v3','refvfx_neutral_v3','refvfx_neutral_v3_teg','wan_flf2v_neutral_v3','wan_vace_neutral_v3'])}",
          f"labels_missing_cache: {json.dumps(labels_missing)}",
          "arms_scored:"]
    for ha in sorted(arms_out):
        d = arms_out[ha]
        pg = d["per_gen"]
        vals = [r["bl_333"] for r in pg if r.get("bl_333") is not None]
        head = round(float(np.mean(vals)), 2) if vals else None
        L += [f"  {ha}:", f"    gen: {d['gen']}", f"    label: {json.dumps(d['label'])}",
              f"    cache: {d['cache']}", f"    rows: {len(d['pairs'])}", f"    gens: {len(pg)}",
              f"    headline_bl_333_mean: {head}"]
    atomic_write(ed / "meta.yaml", "\n".join(L) + "\n")


# ---------------------------------------------------------------- D8 pairs (compute Mu/D/PX for roster arms without a cache)
# Reuses score_v3_mf's channel code (pillars raw_pair, fid, dist_concat) but the rows come from the
# ROSTER: sweep arms via score_v3_cells.load_rows (+ ic_gen completion, exactly like dualforce_dcg_w6);
# neutral twins inherit their effect sibling's (endpoint, reference, seed) -> pool-refs map. dirs/px come
# from the store namespace trackdesc@cotracker3-s64-v1 (never mf_dirs_cache); gen DINO from the store
# namespace dino_cls@dinov2b-r256 (never the legacy caches); PX z-stats from reference_v5's px222.
_MISC = REPO / "misc" / "2026-09-02_temporal_dynamics_metric"
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))


def _stamp_of(gen_rel):
    """harness_arm stamp of a gens variant = the first video stem's __-field 1."""
    for v in (REPO / gen_rel / "videos").glob("*.mp4"):
        return v.stem.split("__")[1]
    return None


def _parts_of(gens):
    """score_v3_cells-style parts [(subrel, stamp)] from roster gens (subrel drops 'store/gens/')."""
    return [(g[len("store/gens/"):], _stamp_of(g)) for g in gens]


def _pairs_context():
    """Corpus manifest, mu, PX z-stats (from reference_v5), and score_v3_* modules — built once."""
    import blend
    import breakdown as BD
    import score_v3_cells as CELLS
    corpus = json.loads((STD / "corpus_manifest.json").read_text())
    keys = sorted(corpus["clips"])
    labels_of = {k: corpus["clips"][k]["class"] for k in keys}
    side = {c: v["sidedness"] for c, v in corpus["classes"].items()}
    stem2key = {Path(k).stem: k for k in keys}
    z = np.load(OUT / "pillars_corpus_222__grid.npz", allow_pickle=True)
    mu = z["mu"]
    ref5 = blend.load_reference(REFERENCE)
    px222 = np.asarray(ref5["px222"])              # [222, 31, 18] float32 (same stack score_v3_mf uses)
    PX_MU = px222.reshape(-1, 18).mean(0)
    PX_SD = px222.reshape(-1, 18).std(0) + 1e-9
    zpx = lambda d: (np.asarray(d, np.float64) - PX_MU) / PX_SD   # score_v3_mf.zpx verbatim
    r = roster()
    by_id = {a["id"]: a for a in r["arms"]}
    labels = _mf_label_map(CELLS)
    return dict(keys=keys, labels_of=labels_of, side=side, stem2key=stem2key, mu=mu, STD=STD,
                zpx=zpx, CELLS=CELLS, BD=BD, by_id=by_id, labels=labels, roster=r)


def _mf_label_map(CELLS):
    """roster arm id -> the score_v3_mf per-gen label (cosmetic; the tables key by harness_arm)."""
    stamp2label = {}
    for grp in ("ours_neutral", "ours_effect", "teg", "ext"):
        for label, parts in CELLS.ARMS[grp]:
            for subrel, stamp in parts:
                stamp2label[stamp] = label
    out = {}
    for arm in roster()["arms"]:
        stamp = _stamp_of(arm["gens"][0])
        out[arm["id"]] = stamp2label.get(stamp, arm["label"])
    return out


def _pairs_rows(arm, ctx):
    """(rows, subs, label, mode) for a roster arm needing computed pairs."""
    CELLS = ctx["CELLS"]
    stem2key = ctx["stem2key"]
    label = ctx["labels"].get(arm["id"], arm["label"])
    if arm.get("twin"):
        return _twin_rows(arm, ctx, label)
    parts = _parts_of(arm["gens"])
    if arm["role"] in ("prior", "prior_teg"):                # effect / author-native: no completion
        rows, subs = CELLS.arm_rows(parts, stem2key)
        return rows, subs, label, ("teg" if arm["role"] == "prior_teg" else "cells")
    # own (sweep): every cell, completed to every generation via ic_gen's pool map (as dualforce_dcg_w6)
    tmpl = dict(CELLS.ARMS["ours_neutral"])["ic_gen neutral"]
    tmpl_rows = CELLS.arm_rows(tmpl, stem2key)[0]
    rows, subs = CELLS.arm_rows(parts, stem2key)
    CELLS.complete(rows, subs, parts, tmpl_rows)
    return rows, subs, label, "cells"


def _twin_rows(arm, ctx, label):
    """A neutral twin: inherit the effect sibling's (trip, seed) -> pool-refs map, apply to the twin's own gens."""
    import collections
    CELLS = ctx["CELLS"]
    stem2key = ctx["stem2key"]
    eff = ctx["by_id"][arm["twin"]]
    eff_rows, _ = CELLS.arm_rows(_parts_of(eff["gens"]), stem2key)
    pool_ts, pool_t = collections.defaultdict(set), collections.defaultdict(set)
    for r in eff_rows:
        seed = r["_gen"].rpartition("__s")[2]
        pool_ts[(r["_trip"], seed)].add(r["_refkey"])
        pool_t[r["_trip"]].add(r["_refkey"])
    rows, subs = [], {}
    for subrel, stamp in _parts_of(arm["gens"]):
        sub = REPO / "store/gens" / subrel
        grid = CELLS.grid_of(sub)
        for gid, gi in grid.items():
            trip = (gi["endpoint"], gi["reference"], gi["cell"])
            for seed in ("42", "43"):
                v = sub / "videos" / f"{gid}__s{seed}.mp4"
                pool = pool_ts.get((trip, seed)) or pool_t.get(trip)
                if not pool or not v.exists():
                    continue
                for rk in pool:
                    rows.append({"app_ref": float("nan"), "_gen": f"{gid}__s{seed}", "_refkey": rk,
                                 "_cls": gi["gt_pool_class"], "_cell": gi["cell"], "_trip": trip,
                                 "_ed": gi["endpoint"].startswith("ed.")})
                subs[f"{gid}__s{seed}"] = sub
    mode = "teg" if arm["role"] == "prior_teg" else "cells"
    return rows, subs, label, mode


def _gen_video(g, subs):
    sub = subs[g]
    for cand in (sub / "videos" / f"{g}.mp4", sub / "videos" / f"{g.replace('__s', '__seed')}.mp4"):
        if cand.exists():
            return cand
    raise FileNotFoundError(g)


def _rows_ready(rows, subs, ctx, fs):
    """True when every gen has store dino_cls + trackdesc and every ref has trackdesc (else skip an arm
    whose features are still being extracted)."""
    for g in sorted({r["_gen"] for r in rows}):
        v = _gen_video(g, subs)
        if not fs.has(v, "dino_cls@dinov2b-r256"):
            return False, f"gen dino_cls: {v.name}"
        if not fs.has(v, TRACKDESC):
            return False, f"gen trackdesc: {v.name}"
    for k in sorted({r["_refkey"] for r in rows}):
        if not fs.has(ctx["STD"] / k, TRACKDESC):
            return False, f"ref trackdesc: {k}"
    return True, ""


def _build_pair_cache(rows, subs, ctx, fs, gen_dino="store"):
    """The mf_pair_cache-shape dict (gens, refs, T, Tshape, E, Mw, Mu, D, fid, pxd) for one arm's rows,
    computed from stored features (gen dino_cls + gen/ref trackdesc + corpus legacy dino), z-stats from
    reference_v5. Byte-for-byte the score_v3_mf channel recipe.

    ``gen_dino`` selects the gen DINO source: ``store`` (the brief default) reads
    dino_cls@dinov2b-r256; ``harness`` uses score_v3_mf's own loader feats_or_extract (legacy cache
    that mirrors the store, store fallback). For every Round-2 arm the two are IDENTICAL (own sweep
    gens have no legacy cache -> store, or legacy == store; twins have no legacy -> store); they differ
    only on the refvfx-TEG A8 arm, whose imported 047 cache was built from a 09-21 re-extraction that
    drifted ~4e-4 from the 09-20 store DINO -- so ``harness`` reproduces that cache exactly (A8)."""
    import pillars as P
    import score_v3_mf as MF
    from run_motion_descriptors import dist_concat
    BD = ctx["BD"]
    side, mu, zpx, STD_ = ctx["side"], ctx["mu"], ctx["zpx"], ctx["STD"]

    def gen_video(g):
        return _gen_video(g, subs)

    glist = sorted({r["_gen"] for r in rows})
    gcls = {r["_gen"]: r["_cls"] for r in rows}
    gvid = {g: gen_video(g) for g in glist}
    gfeats = []
    for g in glist:
        v = gvid[g]
        if gen_dino == "harness":
            gfeats.append(np.asarray(BD.feats_or_extract(v)[0], np.float32))
        else:
            if not fs.has(v, "dino_cls@dinov2b-r256"):
                raise RuntimeError(f"gen missing store dino_cls@dinov2b-r256: {v}")
            gfeats.append(np.asarray(fs.get(v, "dino_cls@dinov2b-r256")["feats"], np.float32))
    gp, _ = P.build_packs(gfeats, [side[gcls[g]] for g in glist], "grid", mu=mu)
    gidx = {g: i for i, g in enumerate(glist)}

    reflist = sorted({r["_refkey"] for r in rows})
    rfeats = []
    for k in reflist:
        f = BD.find_feats(BD.file_key(STD_ / k, BD.DINO, BD.SHORT))
        if f is None:
            raise RuntimeError(f"corpus ref missing legacy dino: {k}")
        rfeats.append(f)
    rp, _ = P.build_packs(rfeats, [side[ctx["labels_of"][k]] for k in reflist], "grid", mu=mu)
    ridx = {k: i for i, k in enumerate(reflist)}

    pairs = [(gidx[r["_gen"]], ridx[r["_refkey"]]) for r in rows]
    raw = {k: np.array([d[k] for d in P.pairs_values(P.raw_pair, gp, rp, pairs, lam=P.LAM)]) for k in P.PILLARS}

    gdirs = {g: np.asarray(fs.get(gvid[g], TRACKDESC)["dirs"], np.float32) for g in glist}
    gpx = {g: zpx(fs.get(gvid[g], TRACKDESC)["px"]) for g in glist}
    cdirs = {k: np.asarray(fs.get(STD_ / k, TRACKDESC)["dirs"], np.float32) for k in reflist}
    cpx = {k: zpx(fs.get(STD_ / k, TRACKDESC)["px"]) for k in reflist}
    fids = np.array([MF.fid(gdirs[r["_gen"]], cdirs[r["_refkey"]]) for r in rows])
    pxd = np.array([dist_concat(gpx[r["_gen"]], cpx[r["_refkey"]]) for r in rows])
    cache = {"gens": np.array([r["_gen"] for r in rows]), "refs": np.array([r["_refkey"] for r in rows]),
             "fid": fids, "pxd": pxd, **raw}
    return cache


def _pending_harness(ed):
    """roster harness arms not yet present as a scored dir in eval 047."""
    present = {d.name for d in ed.iterdir() if d.is_dir()}
    out = []
    for arm in roster()["arms"]:
        for g in arm["gens"]:
            ha = _stamp_of(g)
            if ha and ha not in present and ha not in out:
                out.append(ha)
    return out


def cmd_pairs(args):
    import blend
    from store_eval_common import atomic_write
    if not REFERENCE.exists():
        log("reference_v5.npz missing; run `reference` first"); return 2
    ctx = _pairs_context()
    fs = FeatureStore(REPO)
    pop, fpop, ceil = blend.build_from_reference(blend.load_reference(REFERENCE))
    from versioning import REFERENCE_V5_SHA256
    log(f"reference sha {REFERENCE_V5_SHA256}; ceilings built")

    ed = REPO / "store" / "evals" / args.eval_id
    scratch = Path(args.scratch) if args.scratch else None
    dest = scratch if scratch else ed
    dest.mkdir(parents=True, exist_ok=True)

    # which roster arms to compute pairs for
    if args.arm:
        want = [ctx["by_id"][args.arm]]
    else:
        pending = set(_pending_harness(ed))
        want = [a for a in ctx["roster"]["arms"]
                if any(_stamp_of(g) in pending for g in a["gens"])]
    grid_map_all = None
    new_meta = {}
    for arm in want:
        # skip-if-done (store dest only): every harness arm of this roster arm already present
        harnesses = [_stamp_of(g) for g in arm["gens"]]
        if not scratch and not getattr(args, "force", False) and all((ed / ha).is_dir() for ha in harnesses):
            log(f"[pairs] {arm['id']}: already scored ({harnesses}); skip"); continue
        t0 = time.time()
        rows, subs, label, mode = _pairs_rows(arm, ctx)
        if not rows:
            log(f"[pairs] {arm['id']}: 0 rows; skip"); continue
        ready, why = _rows_ready(rows, subs, ctx, fs)
        if not ready and args.gen_dino == "store":
            log(f"[pairs] {arm['id']}: features not ready ({why}); skip"); continue
        cache = _build_pair_cache(rows, subs, ctx, fs, gen_dino=args.gen_dino)
        gm = _grid_map([f"store/gens/{subrel}" for subrel, _ in _parts_of(arm["gens"])])
        pair_rows, per_gen = blend.score_cache(cache, gm, pop, fpop, ceil, label, mode)
        by_arm_p, by_arm_g = {}, {}
        for rr in pair_rows:
            by_arm_p.setdefault(rr["harness_arm"], []).append(rr)
        for rr in per_gen:
            by_arm_g.setdefault(rr["harness_arm"], []).append(rr)
        for ha in by_arm_g:
            (dest / ha).mkdir(exist_ok=True)
            atomic_write(dest / ha / "pairs.jsonl", "".join(json.dumps(r) + "\n" for r in by_arm_p.get(ha, [])))
            atomic_write(dest / ha / "per_gen.jsonl", "".join(json.dumps(r) + "\n" for r in by_arm_g[ha]))
            vals = [r["bl_333"] for r in by_arm_g[ha] if r.get("bl_333") is not None]
            gen = next((g for g in arm["gens"] if _stamp_of(g) == ha), arm["gens"][0])
            new_meta[ha] = dict(gen=gen, label=label,
                                cache=f"computed:v5.py pairs ({arm['id']}, {mode} mode, stored features)",
                                rows=len(by_arm_p.get(ha, [])), gens=len(by_arm_g[ha]),
                                headline_bl_333_mean=(round(float(np.mean(vals)), 2) if vals else None))
        log(f"[pairs] {arm['id']:<24} {label:<26} ({mode}) -> {len(by_arm_g)} harness arms, "
            f"{len(pair_rows)} pairs, {len(per_gen)} gens in {time.time()-t0:.0f}s")

    if scratch:
        log(f"[pairs] wrote scratch {dest} ({len(new_meta)} harness arms; store 047 untouched)")
        return 0
    if new_meta:
        _update_047_meta(ed, new_meta)
        log(f"[pairs] 047 updated (+{len(new_meta)} harness arms); meta refreshed")
    else:
        log("[pairs] nothing new to score")
    return 0


def _update_047_meta(ed, new_meta):
    """Merge new harness arms into 047 meta.yaml, refresh arms_pending — never drop existing arms."""
    import yaml
    from store_eval_common import atomic_write
    old = yaml.safe_load((ed / "meta.yaml").read_text())
    old_scored = old.get("arms_scored", {}) or {}
    scored = dict(old_scored)
    scored.update(new_meta)
    pending = _pending_harness(ed)
    L = [f"id: {old['id']}", f"seq: {old['seq']}", f"shelf: {old['shelf']}",
         f"created: '{old['created']}'", f"machine: {old['machine']}",
         f"instrument: {old['instrument']}", f"reference_v5_sha256: {old['reference_v5_sha256']}",
         'headline: {column: bl_333, weights: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]}',
         "definition:"]
    L += [f"  - {json.dumps(d)}" for d in old["definition"]]
    caveat = ("Frame-count caveat: ours 121/81 frames vs the prior works 33-49 frames; pools = evals/028s "
              "677-corpus references and 222-based ceilings. Round-1 arms imported from mf_pair_cache; "
              "Round-2 arms (the guidance sweep w=1.5/w=3 and the neutral twins) computed by scripts/metrics_v5/v5.py "
              "pairs -- score_v3_mf's channel code over stored features (gen dino_cls@dinov2b-r256, gen/ref "
              "trackdesc@cotracker3-s64-v1, PX z-stats from reference_v5), reproducing the imported caches (A8).")
    L += [f"why: {json.dumps(old['why'])}", f"caveat: {json.dumps(caveat)}",
          f"arms_pending: {json.dumps(pending)}",
          f"labels_missing_cache: {json.dumps(old.get('labels_missing_cache', []))}",
          "arms_scored:"]
    for ha in sorted(scored):
        d = scored[ha]
        L += [f"  {ha}:", f"    gen: {d['gen']}", f"    label: {json.dumps(d['label'])}",
              f"    cache: {d['cache']}", f"    rows: {d['rows']}", f"    gens: {d['gens']}",
              f"    headline_bl_333_mean: {d['headline_bl_333_mean']}"]
    atomic_write(ed / "meta.yaml", "\n".join(L) + "\n")


# ---------------------------------------------------------------- D4 seam (eval 048)
def cmd_seam(args):
    import seam_eval
    return seam_eval.main(["--eval-id", args.eval_id, "--date", args.date, "--no-index"])


# ---------------------------------------------------------------- R6 pixel-endpoints (eval 050)
def cmd_pixel_endpoints(args):
    """PSNR/SSIM/LPIPS of the given endpoint frames (eval 050). Shardable/resumable; cooperates via per-arm locks."""
    import pixel_endpoint_eval
    argv = ["--eval-id", args.eval_id, "--date", args.date, "--shard", args.shard,
            "--device", args.device, "--workers", str(args.workers), "--no-index"]
    if args.wait:
        argv.append("--wait")
    if args.finalize:
        argv.append("--finalize")
    if args.self_check:
        argv.append("--self-check")
    return pixel_endpoint_eval.main(argv)


# ---------------------------------------------------------------- R4 flow-action (eval 049)
def cmd_flow_action(args):
    """Flow MSE + Action KL (eval 049) from the two whole-video namespaces (flow_u32 + action)."""
    import flow_action_eval
    return flow_action_eval.main(["--eval-id", args.eval_id, "--date", args.date, "--no-index"])


# ---------------------------------------------------------------- D5 finalize (043-046)
FINALIZE = [
    ("scripts/endpoint_identity.py", "043_endpoint_identity_gridv3__dai", "--no-index"),
    ("scripts/endpoint_motion.py", "044_endpoint_motion_gridv3__dai", "--no-index"),
    ("scripts/smooth_matched.py", "045_smooth_matched_gridv3__dai", "--no-index"),
    ("scripts/viclip_text.py", "046_viclip_text_gridv3__dai", None),   # no --no-index flag; passes index_line=None already
]


def cmd_finalize(args):
    for script, base, flag in FINALIZE:
        eval_id = f"{base}__{args.date}"
        cmd = [sys.executable, str(REPO / script), "--eval-id", eval_id]
        if flag:
            cmd.append(flag)
        if script.endswith("endpoint_motion.py"):
            cmd += ["--out-dir", str(LOGDIR / "motion_gridv3_preview")]  # keep the default preview folder untouched
        log("[finalize] " + " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            log(f"[finalize] FAILED: {script} rc={r.returncode}"); return r.returncode
    return 0


# ---------------------------------------------------------------- D6 tables
def cmd_tables(args):
    ft = str(REPO / "scripts/family_tables.py")
    fams = ["metrics_v2_gridv3"]
    runs = [[], ["--clean"], ["--sweep"], ["--sweep", "--clean"]]
    if args.all_families:
        runs = [[]]; fams = None
    for extra in runs:
        cmd = [sys.executable, ft] + (["--families"] + fams if fams else []) + extra
        log("[tables] " + " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            log(f"[tables] FAILED rc={r.returncode}")
    return 0


# ---------------------------------------------------------------- D10 paper-table (placeholder; filled below)
def cmd_paper_table(args):
    return _paper_table_impl(args)


# ---------------------------------------------------------------- D3 (R3) paper-tables
def cmd_paper_tables(args):
    """Render + optionally apply the paper's five result tables (scripts/metrics_v5/paper_tables.py)."""
    import paper_tables
    argv = []
    if args.apply:
        argv.append("--apply")
    if args.no_preview:
        argv.append("--no-preview")
    return paper_tables.main(argv)


# The paper-style main table (papers_drafts/ctt_iclr2027/tables/tab_main.tex): 12 metric columns in the
# paper's order + groups, two blocks (TEG first), each prior work as an effect/author-native row + its
# neutral twin, then our rows. Values are the zero-shot-tier levels family_tables.collect() builds
# (Table 3 shared two-sided set for TEG, Table 2 shared one-sided set for transfer); Transport = bl_333,
# Seam-free from 048. Reuse family_tables' collect + agg; render only.
PAPER_COLS = [                      # (metrics_v2 key, bold direction for the paper)
    ("transport", "max"), ("motfid", "max"), ("vp_ref", "max"), ("seam_free", "max"),
    ("ep_id_A", "max"), ("ep_id_B", "max"), ("ep_mot_A", "max"), ("ep_mot_B", "max"),
    ("text_own", "min"),            # Text consistency: LOWER is better (owner 2026-09-19)
    ("smooth_native", "max"), ("dyn_pxs", "none"), ("aesthetic", "max"),
]
PAPER_HEADER = "\n".join([
    r"        & \multicolumn{4}{c}{Transition fidelity\,$\uparrow$} & \multicolumn{4}{c}{Endpoint fidelity\,$\uparrow$} & & \multicolumn{3}{c}{Quality} \\",
    r"        \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){11-13}",
    r"        & \shortstack{Trans-\\port} & \shortstack{Motion\\fid.} & \shortstack{Ref\\sim.} & \shortstack{Seam-\\free\,\%}",
    r"        & \shortstack{ID\\start} & \shortstack{ID\\end}",
    r"        & \shortstack{Motion\\start} & \shortstack{Motion\\end} & \shortstack{Text\\cons.\,$\downarrow$}",
    r"        & \shortstack{Smooth-\\ness\,$\uparrow$} & \shortstack{Dyn.\\(px/s)}",
    r"        & \shortstack{Aesth.\,$\uparrow$} \\",
])
PAPER_ROW_TEX = {   # roster id -> the table row macro/label (D10)
    "base_cond_effect": r"Base \ltx{}", "ic_gen": "Plain LoRA",
    "dualforce_control": r"\segue{} w/o NRG", "dualforce_dcg_w6": r"\segue{}",
    "vap_author_native": r"\vap{}", "vap_neutral": r"\vap{} (neutral prompt)",
    "vfxmaster_author_native": r"\vfxmaster{}", "vfxmaster_neutral": r"\vfxmaster{} (neutral prompt)",
    "refvfx_author_native": r"\refvfx{}", "refvfx_neutral": r"\refvfx{} (neutral prompt)",
    "refvfx_teg": r"\refvfx{}", "refvfx_teg_neutral": r"\refvfx{} (neutral prompt)",
    "wan_flf2v": "Wan2.1 FLF2V", "wan_flf2v_neutral": "Wan2.1 FLF2V (neutral prompt)",
    "wan_vace": "VACE", "wan_vace_neutral": "VACE (neutral prompt)",
}
# TEG block = base LTX-2, prior works (each two rows), our rows; transfer block = prior works (each two rows), our rows.
PAPER_TEG_ROWS = ["base_cond_effect", "wan_vace", "wan_vace_neutral", "refvfx_teg", "refvfx_teg_neutral", "wan_flf2v", "wan_flf2v_neutral",
                  "__addlinespace__", "dualforce_control", "dualforce_dcg_w6"]          # the paper's TEG order; no Plain LoRA row (as in tab_main.tex)
PAPER_VET_ROWS = ["vap_author_native", "vap_neutral", "vfxmaster_author_native", "vfxmaster_neutral",
                  "refvfx_author_native", "refvfx_neutral", "__addlinespace__", "dualforce_control", "dualforce_dcg_w6"]


def _paper_table_impl(args):
    import family_tables as FT
    FT.CLEAN = True; FT.SWEEP = False; FT.WITH_BASE = False; FT.PRIOR_BASELINE = False; FT.SUFFIX = ""
    recs = FT.collect(args.handoff_motion)
    id2label = {a["id"]: a["label"] for a in roster()["arms"]}
    cols = [k for k, _ in PAPER_COLS]

    def _shared(arm_ids, sided):
        """The shared zero-shot key set over these roster arms (family_tables' Table 2 / Table 3 logic)."""
        per = {}
        for aid in arm_ids:
            lbl = id2label[aid]
            per[aid] = {(r["cell"], r["endpoint"], r["reference"], k[2]): r for k, r in recs.items()
                        if r["label"] == lbl and r["tier"] == "zero_shot" and r["sided"] == sided}
        shared = None
        for aid in arm_ids:
            if not per[aid]:
                continue
            ks = set(per[aid])
            shared = ks if shared is None else (shared & ks)
        return (shared or set()), per

    def _cells(label, shared, sided):
        per = {(r["cell"], r["endpoint"], r["reference"], k[2]): r for k, r in recs.items()
               if r["label"] == label and r["tier"] == "zero_shot" and r["sided"] == sided}
        return FT.cells_for([per[key] for key in shared if key in per], cols), sum(1 for key in shared if key in per)

    # Table 3 / Table 2 shared sets, computed over the SAME arm lists family_tables uses (so the shared
    # rows are numerically identical to metrics_v2_gridv3_clean).
    arms3 = [aid for _, aid in FT.OWN_BASE + FT.EXT_TEG + FT.own_arms()]
    shared3, _ = _shared(arms3, "two")
    arms2 = [aid for _, aid in FT.EXT + FT.own_arms()[1:]]
    shared2, _ = _shared(arms2, "one")

    def _render_block(row_ids, shared, sided):
        cells = {aid: _cells(id2label[aid], shared, sided)[0] for aid in row_ids if aid != "__addlinespace__"}
        best = []
        for ci, (key, direction) in enumerate(PAPER_COLS):
            vals = [cells[aid][ci][0] for aid in cells if cells[aid][ci][0] is not None]
            best.append(None if (direction == "none" or not vals) else (max(vals) if direction == "max" else min(vals)))
        lines = []
        for aid in row_ids:
            if aid == "__addlinespace__":
                lines.append(r"        \addlinespace[2pt]"); continue
            parts = []
            for ci, (key, direction) in enumerate(PAPER_COLS):
                m = cells[aid][ci][0]
                if m is None:
                    parts.append("--"); continue
                s = f"{m:.{FT.CAT[key][2]}f}"
                if best[ci] is not None and abs(m - best[ci]) <= 1e-9:
                    s = r"\textbf{" + s + "}"
                parts.append(s)
            lines.append("        " + PAPER_ROW_TEX[aid] + " & " + " & ".join(parts) + r" \\")
        return "\n".join(lines)

    teg = _render_block(PAPER_TEG_ROWS, shared3, "two")
    vet = _render_block(PAPER_VET_ROWS, shared2, "one")
    caption = (r"\textbf{Comparison with previous approaches and adapted baselines}, zero-shot tier. "
               r"(neutral prompt): the same system with the effect description removed from its text. --: not applicable.")
    tex = "\n".join([
        r"% GENERATED by scripts/metrics_v5/v5.py paper-table -- the paper-style main table.",
        r"% Values = zero-shot-tier levels from scripts/family_tables.py collect(): Table 3 (both endpoints given,"
        r" n=" + str(len(shared3)) + r") for the TEG block, Table 2 (start endpoint given, n=" + str(len(shared2)) + r") for the transfer block.",
        r"% Transport = bl_333 (eval 047); Seam-free from eval 048 (physical windows). Text cons. lower is better.",
        r"\begin{table}[t]", r"    \centering",
        r"    \caption{" + caption + "}", r"    \label{tab:main}",
        r"    \footnotesize", r"    \setlength{\tabcolsep}{2pt}",
        r"    \resizebox{\linewidth}{!}{\begin{tabular}{lcccccccccccc}", r"        \toprule",
        PAPER_HEADER, r"        \midrule",
        r"        \multicolumn{13}{l}{\emph{Transition effect generation: both endpoints given}} \\",
        teg, r"        \midrule",
        r"        \multicolumn{13}{l}{\emph{Visual effect transfer: start endpoint given}} \\",
        vet, r"        \bottomrule", r"    \end{tabular}}", r"\end{table}", ""])

    out = Path(args.out_root) / "paper_table_v5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_main.tex").write_text(tex)
    _paper_table_preview(out, tex)
    log(f"[paper-table] wrote {out/'tab_main.tex'} (TEG n={len(shared3)}, transfer n={len(shared2)})")
    return 0


def _paper_table_preview(out, tab_tex):
    """preview.tex -> preview.pdf with the paper preamble (family_tables' latexmk recipe)."""
    PAPER = REPO / "papers_drafts" / "ctt_iclr2027"
    TEXBIN = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/texlive/bin/aarch64-linux"
    preview = "\n".join([
        r"% GENERATED by scripts/metrics_v5/v5.py paper-table", r"\documentclass{article}",
        r"\usepackage{natbib}", r"\input{preamble}", r"\usepackage{geometry}",
        r"\geometry{landscape, margin=1.5cm}", r"\usepackage{booktabs}", r"\usepackage{graphicx}",
        r"\hypersetup{hidelinks}", r"\begin{document}", r"\begin{center}",
        r"{\Large\bfseries \segue{} -- paper main table (grid v3 preview)}\\[3pt]",
        r"\small " + __import__("datetime").date.today().isoformat() + r". Zero-shot tier; levels, two seeds per row, no confidence intervals.",
        r"\end{center}", r"\vspace{0.4em}", r"\input{tab_main}", r"\end{document}", ""])
    (out / "preview.tex").write_text(preview)
    env = dict(os.environ); env["PATH"] = TEXBIN + ":" + env.get("PATH", "")
    env["TEXINPUTS"] = f".:{PAPER}:{PAPER}//:"; env["BIBINPUTS"] = env["TEXINPUTS"]; env["BSTINPUTS"] = env["TEXINPUTS"]
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "preview.tex"],
                       cwd=out, env=env, capture_output=True, text=True, timeout=600)
    subprocess.run(["latexmk", "-c", "preview.tex"], cwd=out, env=env, capture_output=True, text=True)
    print(f"[paper-table] build {'OK' if r.returncode == 0 else 'FAILED'} -> {out/'preview.pdf'}")
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-25:]))


# ---------------------------------------------------------------- status
# Fast status (D9): count feature files with os.scandir (one scandir per item
# folder), never FeatureStore.coverage over the world; per roster arm the
# per-namespace have/of and per eval 043-048 whether the arm dir exists.
STATUS_NS = [("dino_cls@dinov2b-r256", "dn"), ("cotracker3@g20-m384-v2", "ct"),
             (TRACKDESC, "td"), ("lpips_t@alex-r256", "lp"), ("clip_b32@r256", "cb"),
             ("videoprism@f16r288", "vp"), ("raft_mag@r256", "rm"), ("clip_l14@r224", "cl"),
             ("raft_flow_win@r256", "rf"), ("viclip@l14-f8", "vc"),
             ("flow_u32@raft-r256-g24x32", "fu"), ("action@swin3db-k400-u32", "ac")]
STATUS_EVALS = [("043", "043_endpoint_identity_gridv3"), ("044", "044_endpoint_motion_gridv3"),
                ("045", "045_smooth_matched_gridv3"), ("046", "046_viclip_text_gridv3"),
                ("047", "047_transport_v5_gridv3"), ("048", "048_seam_gridv3"),
                ("049", "049_transition_flow_action_gridv3"), ("050", "050_endpoint_pixel_gridv3")]


def _variant_scan(vdir):
    """(of, {ns: have}, harness_arm) for one gens variant, via os.scandir only."""
    of = 0
    stem0 = None
    try:
        with os.scandir(vdir / "videos") as it:
            for e in it:
                if e.name.endswith(".mp4"):
                    of += 1
                    if stem0 is None:
                        stem0 = e.name[:-4]
    except FileNotFoundError:
        pass
    harness = stem0.split("__")[1] if stem0 and "__" in stem0 else None
    have = {ns: 0 for ns, _ in STATUS_NS}
    try:
        with os.scandir(vdir / "features") as it:
            item_dirs = [e.path for e in it if e.is_dir()]
    except FileNotFoundError:
        item_dirs = []
    for d in item_dirs:
        try:
            names = set(os.listdir(d))
        except OSError:
            continue
        for ns, _ in STATUS_NS:
            if f"{ns}.npz" in names and f"{ns}.json" in names:
                have[ns] += 1
    return of, have, harness


def _eval_dirs():
    """{short: newest eval path} for 043-048 (or None)."""
    out = {}
    for short, prefix in STATUS_EVALS:
        c = sorted((REPO / "store" / "evals").glob(prefix + "*"))
        out[short] = c[-1] if c else None
    return out


def cmd_status(args):
    r = roster()
    evals = _eval_dirs()
    abbr = " ".join(f"{a:>7}" for _, a in STATUS_NS)
    print(f"{'arm':<26}{'role':<10} {'of':>5}  {abbr}  | " + " ".join(short for short, _ in STATUS_EVALS))
    for arm in r["arms"]:
        of_tot = 0
        have_tot = {ns: 0 for ns, _ in STATUS_NS}
        harnesses = []
        for g in arm["gens"]:
            of, have, harness = _variant_scan(REPO / g)
            of_tot += of
            for ns, _ in STATUS_NS:
                have_tot[ns] += have[ns]
            if harness:
                harnesses.append(harness)
        cells = []
        for ns, _ in STATUS_NS:
            h = have_tot[ns]
            cells.append(f"{h:>7}" if h == of_tot else f"{h}/{of_tot}".rjust(7))
        # eval-dir presence: any of this arm's harness arms scored in the eval
        flags = []
        for short, _ in STATUS_EVALS:
            ed = evals[short]
            present = bool(ed) and any((ed / ha).is_dir() for ha in harnesses)
            flags.append(" Y " if present else " . ")
        print(f"{arm['id']:<26}{arm['role']:<10} {of_tot:>5}  {' '.join(cells)}  | " + "".join(flags))
    print("\nns: " + ", ".join(f"{a}={ns}" for ns, a in STATUS_NS))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import-trackdesc")
    p = sub.add_parser("reference"); p.add_argument("--force", action="store_true")
    from datetime import date
    p = sub.add_parser("blend"); p.add_argument("--eval-id", default="047_transport_v5_gridv3__dai__2026-09-23"); p.add_argument("--date", default="2026-09-23")
    p = sub.add_parser("pairs"); p.add_argument("--eval-id", default="047_transport_v5_gridv3__dai__2026-09-23")
    p.add_argument("--arm", default=None, help="a single roster arm id (else every pending roster arm whose features are ready)")
    p.add_argument("--scratch", default=None, help="write to this dir instead of the store 047 (A8; no meta update)")
    p.add_argument("--force", action="store_true", help="recompute an arm already present in 047 (overwrites its pairs/per_gen; meta arm entry refreshed)")
    p.add_argument("--gen-dino", dest="gen_dino", choices=["store", "harness"], default="store",
                   help="gen DINO source: store=dino_cls@dinov2b-r256 (brief default); harness=score_v3_mf's feats_or_extract (A8 reproduction of the imported cache)")
    p.add_argument("--date", default="2026-09-23")
    p = sub.add_parser("seam"); p.add_argument("--eval-id", default="048_seam_gridv3__dai__2026-09-23"); p.add_argument("--date", default="2026-09-23")
    p = sub.add_parser("flow-action"); p.add_argument("--eval-id", default="049_transition_flow_action_gridv3__dai__2026-09-23"); p.add_argument("--date", default="2026-09-23")
    p = sub.add_parser("pixel-endpoints")   # R6: eval 050 PSNR/SSIM/LPIPS of the given endpoint frames
    p.add_argument("--eval-id", default="050_endpoint_pixel_gridv3__dai__2026-09-24")
    p.add_argument("--date", default="2026-09-24")
    p.add_argument("--shard", default="0/1", help="i/n: slice the sorted harness arms (locks let shards + the login race cooperate)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu", help="LPIPS device (decode/PSNR/SSIM stay on the CPU pool)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--wait", action="store_true", help="loop until every arm is complete (the login guarantor)")
    p.add_argument("--finalize", action="store_true", help="write meta.yaml (all arms must be complete)")
    p.add_argument("--self-check", dest="self_check", action="store_true", help="run the three CPU self-checks and exit")
    p = sub.add_parser("finalize"); p.add_argument("--date", default="2026-09-23")
    p = sub.add_parser("tables"); p.add_argument("--all-families", action="store_true")
    p = sub.add_parser("paper-table")
    p.add_argument("--out-root", default=str(REPO / "papers_drafts/_preview"))
    p.add_argument("--handoff-motion", default="none")
    p = sub.add_parser("paper-tables")   # D3 (R3): the paper's five result tables
    p.add_argument("--apply", action="store_true", help="back up + copy the rendered tables into the paper and build main.tex")
    p.add_argument("--no-preview", action="store_true", help="skip the preview.pdf build")
    sub.add_parser("status")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    fn = {"import-trackdesc": cmd_import_trackdesc, "reference": cmd_reference, "blend": cmd_blend,
          "pairs": cmd_pairs, "seam": cmd_seam, "flow-action": cmd_flow_action, "pixel-endpoints": cmd_pixel_endpoints,
          "finalize": cmd_finalize, "tables": cmd_tables, "paper-table": cmd_paper_table, "paper-tables": cmd_paper_tables,
          "status": cmd_status}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
