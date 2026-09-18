#!/usr/bin/env python
"""lens_pass_gridv3 — competitor-lens scoring over the grid-v3 population, via ``score_batch --store``.

Drives ``misc/2026-08-13_baseline_metric_table/their_metrics/score_batch.py`` (the FROZEN
competitor-metric scorer, impl_sha stamped into every row) over the 19 grid-v3 gen variants named in
``misc/2026-09-17_feature_store/population_gridv3.json``. The per-video lens arrays
(``clip_b32@r256``, ``videoprism@f16r288``, ``raft_mag@r256``, ``cotracker3@g20-m384-v2``) are read
from / written to the contract-v2 feature store BY VIDEO PATH; the input start frame is embedded
fresh. This wrapper imports ``score_batch``'s ``score_one`` / ``Models`` / ``StoreLenses`` /
``impl_sha`` so the metric arithmetic and impl_sha are IDENTICAL to the frozen scorer (single source),
loads the backbones once, and lays the per-gen JSON out per harness arm.

Two phases (same CLI):

  score (default)  --shard i/n : score this shard's slice of the population, writing one JSON per gen
                   at ``store/evals/<eval-id>/<harness_arm>/rows/<gen stem>.json``. Resumable
                   (skip-if-exists), so a job array of N shards fills the roster and reruns are cheap.

  --collect        : merge the per-gen JSON into ``<harness_arm>/rows.jsonl`` (one row per gen, keyed
                   by the GRID (item_id, seed) — joined to Op-2's per_gen.jsonl by gen path, NOT by
                   score_batch's own filename-stem item_id), write ``meta.yaml``, append one
                   ``store/INDEX.md`` line. Idempotent.

  --plan           : enumerate arms / gen counts / this shard's slice and sample store coverage; loads
                   no models, scores nothing (CPU, safe on the login node).

Rows whose input-frame png is missing still score every other lens (a warning is recorded); the pass
never crashes on a single gen. GPU only when a per-video lens namespace is missing from the store
(then score_batch fills it via the pinned REGISTRY extractor); with the roster fully extracted the only
model loads are the fresh input-frame CLIP/VideoPrism embeds.

Read first: ``store/FEATURES.md`` (namespaces), ``misc/2026-08-13_baseline_metric_table/their_metrics/
{score_batch.py,SCORERS.json}``, ``store/README.md`` (eval-entry contract),
``store/evals/038_handoff_gridv3__dai__2026-09-18/meta.yaml`` (the shape copied here).
"""

from __future__ import annotations

import argparse
import datetime
import glob as globmod
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THEIR_DIR = REPO_ROOT / "misc" / "2026-08-13_baseline_metric_table" / "their_metrics"
POPULATION = REPO_ROOT / "misc" / "2026-09-17_feature_store" / "population_gridv3.json"
EVAL_DATE = "2026-09-18"
DEFAULT_EVAL_ID = f"039_lenses_gridv3__dai__{EVAL_DATE}"

# The lens columns carried into rows.jsonl (subset of score_batch's per-gen JSON keys).
LENS_KEYS = ["motion_smoothness", "videoprism_sim_ref", "videoprism_sim_input",
             "clip_sim_ref", "clip_sim_input", "det_motion_fidelity", "dynamic_degree_mean_mag"]

# Per-video lens namespaces the store must hold for a gen (for coverage reporting / --plan).
LENS_NS = ("clip_b32@r256", "videoprism@f16r288", "raft_mag@r256", "cotracker3@g20-m384-v2")

# v4 evals that carry Op-2's per_gen.jsonl (internal paper arms + externals).
PERGEN_EVALS = ("028_grid_v3_paper_arms__dai__2026-09-07",
                "030_external_zs_authornative__dai__2026-09-12")


# --- score_batch import (single source of the metric arithmetic + impl_sha) --
def _import_score_batch():
    if str(THEIR_DIR) not in sys.path:
        sys.path.insert(0, str(THEIR_DIR))
    import score_batch as sb  # noqa: E402
    return sb


# --- small helpers ----------------------------------------------------------
def _now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp-{os.getpid()}"
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_meta(path: Path) -> dict:
    """Tiny top-level ``key: value`` reader for a gen meta.yaml (no yaml dep)."""
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        if ln[:1] in (" ", "\t", "#") or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        v = v.split("  #", 1)[0].strip().strip("'\"")
        out[k.strip()] = v
    return out


def _relpath(p) -> str:
    """Repo-relative POSIX path for a video that lives under the repo root."""
    return Path(p).resolve().relative_to(REPO_ROOT).as_posix()


# --- population / arm / gen enumeration -------------------------------------
def load_population(path: Path = POPULATION) -> dict:
    return json.loads(Path(path).read_text())


def variant_harness_arm(variant_rel: str) -> str:
    """The frozen harness_arm that joins keep (from the variant's own meta.yaml; the same field
    Op-2/Op-3 stamp into their item_ids). Falls back to the strip rule if the field is absent."""
    meta_p = REPO_ROOT / variant_rel / "meta.yaml"
    if meta_p.exists():
        m = _read_meta(meta_p)
        ha = m.get("harness_arm") or (
            f"{m['arm']}_{m['variant']}" if m.get("arm") and m.get("variant") else None)
        if ha:
            return ha
    # strip rule: <group w/o NNN_>_<variant w/o KK_ and __machine>
    import re
    parts = variant_rel.rstrip("/").split("/")
    group = re.sub(r"^\d+_", "", parts[-2])
    variant = re.sub(r"^\d+_", "", parts[-1])
    variant = variant.rsplit("__", 1)[0] if "__" in variant else variant
    return f"{group}_{variant}"


def enumerate_gens(variant_rel: str) -> list[Path]:
    vdir = REPO_ROOT / variant_rel / "videos"
    return sorted(vdir.glob("*.mp4"))


def all_items(pop: dict) -> list[dict]:
    """One entry per gen across the 19 variants, globally sorted for stable sharding."""
    items: list[dict] = []
    for variant_rel in pop["gen_variants"]:
        arm = variant_harness_arm(variant_rel)
        for gen in enumerate_gens(variant_rel):
            items.append({"variant_rel": variant_rel, "arm": arm, "gen": gen})
    items.sort(key=lambda d: str(d["gen"]))
    return items


def shard_slice(items: list, shard: int, num_shards: int) -> list:
    return items[shard::num_shards] if num_shards > 1 else items


# --- per_gen.jsonl join index (grid item_id + seed, keyed by gen path) ------
def build_pergen_index(arm: str) -> dict[str, tuple[str, int]]:
    """{gen_video repo-rel path -> (grid item_id, seed)} from Op-2's per_gen.jsonl for this arm."""
    idx: dict[str, tuple[str, int]] = {}
    for ev in PERGEN_EVALS:
        for per_gen in (REPO_ROOT / "store" / "evals").glob(f"{ev.split('__')[0]}*/{arm}/per_gen.jsonl"):
            for ln in per_gen.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                gv = r.get("gen_video")
                if gv is None:
                    continue
                gv = Path(gv).as_posix()  # already repo-relative in Op-2 rows
                idx[gv] = (r.get("item_id"), r.get("seed"))
    return idx


def _grid_ids_from_stem(stem: str, sb_seed) -> tuple[str, int]:
    """Fallback when a gen has no per_gen match: strip the trailing __s<seed> token."""
    import re
    m = re.search(r"__s(\d+)$", stem)
    if m:
        return stem[: m.start()], int(m.group(1))
    return stem, (sb_seed if sb_seed is not None else -1)


def merge_row(scored: dict, pergen_index: dict, arm: str) -> tuple[dict, str]:
    """One rows.jsonl row from a score_batch per-gen JSON, with the GRID (item_id, seed) resolved
    by joining on gen path. Returns (row, join_status) where join_status in {pergen, fallback}."""
    gen_rel = _relpath(scored["gen"])
    warnings = list(scored.get("warnings") or [])
    if gen_rel in pergen_index:
        item_id, seed = pergen_index[gen_rel]
        status = "pergen"
    else:
        item_id, seed = _grid_ids_from_stem(scored.get("item_id", Path(gen_rel).stem),
                                             scored.get("seed"))
        warnings = warnings + ["no_pergen_join"]
        status = "fallback"
    row = {"item_id": item_id, "seed": seed, "arm": arm, "gen": gen_rel}
    for k in LENS_KEYS:
        row[k] = scored.get(k)
    row["impl_sha"] = scored.get("impl_sha")
    row["warnings"] = warnings
    return row, status


# --- scoring phase ----------------------------------------------------------
def score_shard(pop: dict, eval_dir: Path, shard: int, num_shards: int,
                device: str, store_root: Path, limit: int, overwrite: bool) -> dict:
    sb = _import_score_batch()
    sha = sb.impl_sha()
    items = shard_slice(all_items(pop), shard, num_shards)
    if limit:
        items = items[:limit]
    print(f"[lens] impl_sha={sha} shard={shard}/{num_shards} this_shard={len(items)} "
          f"eval_dir={eval_dir} store_root={store_root} device={device}", flush=True)

    store = sb.StoreLenses(store_root, device)
    M = sb.Models(device, THEIR_DIR / ".track_cache", store=store)
    done = skip = fail = 0
    for i, it in enumerate(items):
        gen = it["gen"]
        rows_dir = eval_dir / it["arm"] / "rows"
        of = rows_dir / f"{gen.stem}.json"
        if of.exists() and not overwrite:
            skip += 1
            continue
        try:
            row = sb.score_one(gen, M, sha)
            _atomic_write(of, json.dumps(row))
            done += 1
            if done % 50 == 1 or i + 1 == len(items):
                print(f"[{i+1}/{len(items)}] {it['arm']}/{gen.stem} "
                      f"clip_ref={row.get('clip_sim_ref')} vp_ref={row.get('videoprism_sim_ref')} "
                      f"smooth={row.get('motion_smoothness')} detMF={row.get('det_motion_fidelity')} "
                      f"dyn={row.get('dynamic_degree_mean_mag')} ({row.get('elapsed_sec')}s)", flush=True)
        except Exception as e:  # never let one gen kill the shard
            fail += 1
            print(f"[{i+1}/{len(items)}] FAIL {it['arm']}/{gen.stem}: {type(e).__name__}: {e}", flush=True)
    print(f"[lens] shard done={done} skip={skip} fail={fail} impl_sha={sha}", flush=True)
    return {"done": done, "skip": skip, "fail": fail, "impl_sha": sha}


# --- collect phase ----------------------------------------------------------
def collect_arm(eval_dir: Path, variant_rel: str, arm: str,
                pergen_index: dict | None = None) -> dict:
    """Merge <arm>/rows/*.json into <arm>/rows.jsonl; return per-arm stats.
    ``pergen_index`` (gen relpath -> (grid item_id, seed)) is built from Op-2's per_gen.jsonl
    when None; tests inject it directly."""
    rows_dir = eval_dir / arm / "rows"
    scored_files = sorted(rows_dir.glob("*.json")) if rows_dir.is_dir() else []
    if pergen_index is None:
        pergen_index = build_pergen_index(arm)
    merged: list[dict] = []
    n_pergen = n_fallback = 0
    n_missing_frame = 0
    lens_finite = {k: 0 for k in LENS_KEYS}
    impl = None
    for f in scored_files:
        scored = json.loads(f.read_text())
        row, status = merge_row(scored, pergen_index, arm)
        if status == "pergen":
            n_pergen += 1
        else:
            n_fallback += 1
        if any(w.startswith("missing input_frame") for w in (scored.get("warnings") or [])):
            n_missing_frame += 1
        for k in LENS_KEYS:
            v = row.get(k)
            if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v)):
                lens_finite[k] += 1
        impl = row.get("impl_sha") or impl
        merged.append(row)
    merged.sort(key=lambda r: (str(r["item_id"]), r["seed"] if r["seed"] is not None else -1))
    out = eval_dir / arm / "rows.jsonl"
    _atomic_write(out, "".join(json.dumps(r) + "\n" for r in merged))
    return {"arm": arm, "gen": variant_rel, "n": len(merged),
            "pergen_join": n_pergen, "fallback_join": n_fallback,
            "missing_input_frame": n_missing_frame, "lens_finite": lens_finite,
            "impl_sha": impl, "pergen_index_size": len(pergen_index)}


def write_meta(eval_dir: Path, eval_id: str, seq: int, created: str,
               impl_sha: str, store_root: Path, results: list[dict]) -> Path:
    lines = [
        f"id: {eval_id}",
        f"seq: {seq}",
        "shelf: evals",
        f"created: '{created}'",
        "machine: dai (login node drives GPU shards; competitor lenses via score_batch --store)",
        f"instrument: scripts/lens_pass_gridv3.py @ {_git_sha()} "
        f"-> their_metrics/score_batch.py --store (impl_sha {impl_sha})",
        f"store: {{root: {_relpath(store_root) if Path(store_root).resolve() != REPO_ROOT else '.'}, "
        f"mode: on}}",
        "lenses:",
    ]
    for k in LENS_KEYS:
        lines.append(f"  - {k}")
    lines.append("  - aesthetic  # merged by scripts/aesthetic_from_store.py (LAION head on clip_l14@r224)")
    lines.append("join: (item_id, seed) = grid item_id + seed, from Op-2 per_gen.jsonl (evals 028|030) "
                 "matched by gen path; fallback strips the __s<seed> token.")
    lines.append("arms_scored:")
    for res in results:
        lf = res["lens_finite"]
        lines.append(f"  {res['arm']}:")
        lines.append(f"    gen: {res['gen']}")
        lines.append(f"    rows: {res['n']}")
        lines.append(f"    join: {{pergen: {res['pergen_join']}, fallback: {res['fallback_join']}}}")
        lines.append(f"    missing_input_frame: {res['missing_input_frame']}")
        lines.append("    finite: {" + ", ".join(f"{k}: {lf[k]}" for k in LENS_KEYS) + "}")
    meta_p = eval_dir / "meta.yaml"
    _atomic_write(meta_p, "\n".join(lines) + "\n")
    return meta_p


def append_index_line(eval_id: str, n_arms: int, n_rows: int, impl_sha: str) -> bool:
    index = REPO_ROOT / "store" / "INDEX.md"
    text = index.read_text()
    if f"`{eval_id}`" in text:
        return False
    num = int(eval_id.split("_", 1)[0])
    line = (f"{num}. `{eval_id}` — competitor lenses on the grid-v3 population ({n_arms} arms, "
            f"{n_rows} gens) via the FROZEN scorer `their_metrics/score_batch.py --store` "
            f"(impl_sha {impl_sha}): motion_smoothness, videoprism_sim_ref/input, clip_sim_ref/input, "
            f"det_motion_fidelity, dynamic_degree_mean_mag from the stored `clip_b32@r256` / "
            f"`videoprism@f16r288` / `raft_mag@r256` / `cotracker3@g20-m384-v2` features (input start "
            f"frame embedded fresh) + `aesthetic` (LAION head on `clip_l14@r224`, "
            f"scripts/aesthetic_from_store.py). (item_id, seed) join Op-2's per_gen.jsonl (evals "
            f"028|030) by gen path. GPU shards; NaN+`missing` where a feature/frame is absent "
            f"(idempotent). rows.jsonl are store artifacts (not committed). scripts/lens_pass_gridv3.py.")
    lines = text.splitlines()
    try:
        ev = next(i for i, ln in enumerate(lines) if ln.strip() == "## evals")
    except StopIteration:
        raise SystemExit("INDEX.md: no '## evals' section")
    nxt = next((i for i in range(ev + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    insert_at = nxt
    while insert_at > ev + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, line)
    _atomic_write(index, "\n".join(lines) + "\n")
    return True


def do_collect(pop: dict, eval_dir: Path, eval_id: str, store_root: Path,
               no_index: bool) -> dict:
    seq = int(eval_id.split("_", 1)[0])
    results = []
    for variant_rel in pop["gen_variants"]:
        arm = variant_harness_arm(variant_rel)
        results.append(collect_arm(eval_dir, variant_rel, arm))
    impl = next((r["impl_sha"] for r in results if r["impl_sha"]), None) or "unknown"
    n_rows = sum(r["n"] for r in results)
    write_meta(eval_dir, eval_id, seq, EVAL_DATE, impl, store_root, results)
    added = False if no_index else append_index_line(eval_id, len(results), n_rows, impl)
    print(f"[collect] eval={eval_id} arms={len(results)} rows={n_rows} impl_sha={impl} "
          f"index_added={added}", flush=True)
    for r in results:
        print(f"  {r['arm']:34s} n={r['n']:4d} pergen={r['pergen_join']:4d} "
              f"fallback={r['fallback_join']:3d} miss_frame={r['missing_input_frame']:3d}", flush=True)
    return {"arms": len(results), "rows": n_rows, "impl_sha": impl, "results": results}


# --- plan phase (no models) -------------------------------------------------
def do_plan(pop: dict, eval_dir: Path, shard: int, num_shards: int, sample: int) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from diffusion.feature_store import FeatureStore
    fs = FeatureStore(REPO_ROOT)
    items = all_items(pop)
    slice_ = shard_slice(items, shard, num_shards)
    by_arm: dict[str, int] = {}
    for it in items:
        by_arm[it["arm"]] = by_arm.get(it["arm"], 0) + 1
    print(f"[plan] eval_dir={eval_dir} arms={len(by_arm)} total_gens={len(items)} "
          f"shard={shard}/{num_shards} this_shard={len(slice_)}", flush=True)
    for variant_rel in pop["gen_variants"]:
        arm = variant_harness_arm(variant_rel)
        print(f"  {arm:34s} {by_arm.get(arm,0):4d} gens  <- {variant_rel}", flush=True)
    # store-coverage sample: are the per-video lens namespaces present for the first few gens?
    print(f"[plan] store-coverage sample (first {sample} gens of the shard):", flush=True)
    for it in slice_[:sample]:
        gen = it["gen"]
        st = {ns.split("@")[0]: ("hit" if fs.has(gen, ns) else "MISS") for ns in LENS_NS}
        print(f"    {gen.name}: " + "  ".join(f"{k}={v}" for k, v in st.items()), flush=True)


# --- CLI --------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="grid-v3 competitor-lens pass (drives score_batch --store)")
    ap.add_argument("--eval-id", default=DEFAULT_EVAL_ID)
    ap.add_argument("--population", default=str(POPULATION))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--store-root", default=None, help="feature store root (default: repo root)")
    ap.add_argument("--limit", type=int, default=0, help="cap N gens (validation)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--collect", action="store_true", help="merge per-gen JSON -> rows.jsonl + meta + INDEX")
    ap.add_argument("--no-index", action="store_true", help="collect without touching store/INDEX.md")
    ap.add_argument("--plan", action="store_true", help="enumerate + sample store coverage; no models")
    ap.add_argument("--plan-sample", type=int, default=8)
    a = ap.parse_args(argv)

    pop = load_population(Path(a.population))
    eval_dir = REPO_ROOT / "store" / "evals" / a.eval_id
    store_root = Path(a.store_root).resolve() if a.store_root else REPO_ROOT

    if a.plan:
        do_plan(pop, eval_dir, a.shard, a.num_shards, a.plan_sample)
        return 0
    if a.collect:
        do_collect(pop, eval_dir, a.eval_id, store_root, a.no_index)
        return 0
    score_shard(pop, eval_dir, a.shard, a.num_shards, a.device, store_root, a.limit, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
