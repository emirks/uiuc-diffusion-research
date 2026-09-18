#!/usr/bin/env python
"""copy_metrics — M2a copy rate against each generation's OWN reference, from stored features.

The grid-v3 table needs the M2a copy question answered the way the harness asks it of a
single item: "did the generation replay ITS OWN reference's shots?". In evals/028/030 every
pool clip is an eval item, so ``copy_max`` / ``near_copy`` were scored against POOL references
and the generation's own conditioning reference was never in the pool — near_copy came out
identical across all four paper arms on 100% of the pool rows. This script fixes that by
reusing the harness ``copy_score`` VERBATIM (``src/diffusion/transition_eval/m2_integrity.py``,
``TAU_COPY = 0.858``) between each generation's MID frames and its OWN reference's NON-CORE
DINO frames.

Per generation g with own reference r (the ``reference`` field of the gen's grid.jsonl row;
r's features live in the corpus store at ``data/processed/transitions_std121/<class>/features/
<r>/dino_cls@dinov2b-r256.npz``):
  - ``gen_mid`` = ``mid_mask(T_g, n_pre, n_suf)`` with the SAME n_pre/n_suf rules as
    ``scripts/handoff_metrics.py`` (HF 9/8|0 by the gen's sidedness; ED 1/0; externals 1/0).
  - ``ref_core`` = the reference's core mask computed EXACTLY as the harness computes a
    reference bundle: ``morph_profile(r_feats, n_prefix=9, n_suffix=8, n_endpoints=2)`` then
    ``core_mask_v3(profile, r's sidedness)`` (``_ref_bundle_cache`` in transition_eval/score.py).
  - ``copy_score(gen_feats, gen_mid, ref_feats, ref_core, TAU_COPY)`` -> copy_max, near_copy,
    copy_gen_frame, copy_ref_frame; the score's ``ref_idx`` is the reference's NON-core frames.
  - also emits ``ref_core_frac`` (fraction of reference frames in core) and ``n_mid`` (gen mid
    frames), plus ``missing`` when a required feature is absent.

A missing feature yields NaN + a ``missing`` entry; the script never crashes and is idempotent
per run (rewrites rows.jsonl / meta.yaml deterministically, appends the INDEX line only once).
CPU only; numpy over stored features; no GPU, no backbones.

Output: ``store/evals/<NNN>_copy_gridv3__dai__<date>/<harness_arm>/rows.jsonl`` (one row per gen)
+ ``meta.yaml`` + one appended ``store/INDEX.md`` line.

Read first: ``store/FEATURES.md``. Library imports: ``diffusion.feature_store`` (path-is-identity
store), ``diffusion.transition_eval`` (``copy_score``, ``mid_mask``, ``TAU_COPY``, ``morph_profile``,
``core_mask_v3``) — the copy machinery is reused verbatim, not reimplemented.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.feature_store import FeatureStore  # noqa: E402
from diffusion.transition_eval.m2_integrity import (TAU_COPY, copy_score,  # noqa: E402
                                                    mid_mask)
from diffusion.transition_eval.morph import morph_profile  # noqa: E402
from diffusion.transition_eval.s_structure import core_mask_v3  # noqa: E402

# --- fixed pins -------------------------------------------------------------
DINO_NS = "dino_cls@dinov2b-r256"
CORPUS_MANIFEST = REPO_ROOT / "data" / "processed" / "transitions_std121" / "corpus_manifest.json"
EXTERNAL_ARMS = ("refvfx", "vap", "vfxmaster")
NAN = float("nan")

# The DEFINITION block, verbatim from BRIEF_OP6_copy.md — copied into the eval meta.yaml.
DEFINITION = [
    "For each generation g with own reference r (the `reference` field of the gen's grid.jsonl "
    "row; r's features are in the corpus store, "
    "data/processed/transitions_std121/<class>/features/<r>/dino_cls@dinov2b-r256.npz):",
    "gen_mid = mid_mask(T_g, n_pre, n_suf) with the SAME n_pre/n_suf rules as handoff_metrics "
    "(HF 9/8|0 by sidedness; ED 1/0; externals 1/0).",
    "ref_core = the reference's core mask computed exactly as the harness computes a reference "
    "bundle: morph_profile(r_feats, n_prefix=9, n_suffix=8, n_endpoints=2) then "
    "core_mask_v3(profile, r's sidedness) (transition_eval/score.py::_ref_bundle_cache).",
    "copy_score(gen_feats, gen_mid, ref_feats, ref_core, TAU_COPY) -> copy_max, near_copy, "
    "copy_gen_frame, copy_ref_frame; ref_idx = the reference's NON-core (~ref_core) frames — its "
    "own scenes A/B, the content that must never appear.",
    "Also emit ref_core_frac (fraction of reference frames in core) and n_mid (gen mid frames), "
    "and `missing` when a feature is absent.",
]
WHY = ("In evals/028/030 every pool clip is an eval item, so copy_max/near_copy were scored "
       "against POOL references, never against the generation's own conditioning reference; "
       "near_copy was identical across all four paper arms on 100% of the pool rows (coordinator "
       "check 2026-09-18). This eval scores M2a against the OWN reference: did the generation "
       "replay the reference's own shots?")
CAVEAT = ("copy_max is a max over mid frames, so longer generations (121 f) have more chances "
          "than the externals' 49 f / 33 f — the externals' copy rate is, if anything, "
          "under-estimated relative to ours; disclosed, not corrected.")


# --- small helpers ----------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _parse_stem(video_stem: str) -> tuple[str, int]:
    """A gen video stem is ``<grid item_id>__s<seed>``; split into (item_id, seed)."""
    m = re.search(r"__s(\d+)$", video_stem)
    if not m:
        return video_stem, -1
    return video_stem[: m.start()], int(m.group(1))


def grid_type(arm: str, variant: str) -> str:
    """``external`` / ``ED`` / ``HF`` — the tier that fixes n_pre/n_suf (as handoff_metrics)."""
    if arm in EXTERNAL_ARMS:
        return "external"
    if "ed81" in variant:
        return "ED"
    return "HF"


def windows(gtype: str, sided: str) -> tuple[int, int]:
    """(n_pre, n_suf) per the FIXED rule (identical to handoff_metrics.windows)."""
    if gtype == "HF":
        return 9, (8 if sided == "two" else 0)
    # ED and externals both condition on frame 0.
    return 1, 0


# --- reference resolution (grid.jsonl `reference` -> corpus clip + sidedness) -
class Corpus:
    """corpus_manifest lookups: reference stem -> clip video path, class, sidedness."""

    def __init__(self, manifest_path: Path):
        m = json.loads(manifest_path.read_text())
        self.root = REPO_ROOT / m["corpus_root"]
        self.clips = m["clips"]                       # "<class>/<stem>.mp4" -> {class,...}
        self.classes = m["classes"]                   # class -> {sidedness,...}
        self.stem2key: dict[str, str] = {}
        for key in self.clips:
            self.stem2key.setdefault(Path(key).stem, key)  # stems are unique (verified)

    def resolve(self, row: dict) -> tuple[Path | None, str | None, str | None]:
        """Return (ref_video_path, class, sidedness). The `reference` stem is the primary
        key (present on every arm); `reference_video` is the fallback for externals whose
        stem is not in the corpus. sidedness is the corpus class's ('onesided'/'twosided')."""
        ref = row.get("reference")
        key = self.stem2key.get(ref)
        if key is not None:
            rp = self.root / key
        else:
            refvid = row.get("reference_video")
            if not refvid:
                return None, None, None
            rp = Path(refvid)
            try:
                key = str(rp.resolve().relative_to(self.root.resolve()))
            except Exception:
                key = None
        cls = self.clips.get(key, {}).get("class") if key else None
        side = self.classes.get(cls, {}).get("sidedness") if cls else None
        return rp, cls, side


# --- feature access ---------------------------------------------------------
class Feats:
    """Lazy DINO-feature loader with a small per-reference cache (references are shared
    across many gens; gen features are read once per gen)."""

    def __init__(self, fs: FeatureStore):
        self.fs = fs
        self._refcache: dict[str, np.ndarray | None] = {}

    def gen_feats(self, gen_video: Path) -> np.ndarray | None:
        return self.fs.get(gen_video, DINO_NS)["feats"] if self.fs.has(gen_video, DINO_NS) else None

    def ref_feats(self, ref_video: Path) -> np.ndarray | None:
        k = str(ref_video)
        if k not in self._refcache:
            self._refcache[k] = (self.fs.get(ref_video, DINO_NS)["feats"]
                                 if self.fs.has(ref_video, DINO_NS) else None)
        return self._refcache[k]


# --- per-gen metric computation ---------------------------------------------
def compute_row(feats: Feats, corpus: Corpus, gen_video: Path, row: dict | None,
                gtype: str) -> dict:
    item_id, seed = _parse_stem(gen_video.stem)
    sided = (row.get("sided", "one") if row else "one")
    n_pre, n_suf = windows(gtype, sided)

    missing: set[str] = set()
    n_mid = 0
    ref_core_frac = NAN
    result = {"copy_max": NAN, "near_copy": None,
              "copy_gen_frame": None, "copy_ref_frame": None}

    if row is None:
        missing.add("grid_row")

    ref_video, _cls, side = (corpus.resolve(row) if row is not None else (None, None, None))
    if row is not None and (ref_video is None or side is None):
        missing.add("reference_unresolved")

    g = feats.gen_feats(gen_video)
    if g is None:
        missing.add(f"{DINO_NS}:gen")

    r = feats.ref_feats(ref_video) if (ref_video is not None) else None
    if ref_video is not None and r is None:
        missing.add(f"{DINO_NS}:ref")

    if g is not None and r is not None and side is not None:
        T_g = g.shape[0]
        gen_mid = mid_mask(T_g, n_pre, n_suf)
        n_mid = int(gen_mid.sum())
        try:
            profile = morph_profile(r, n_prefix=9, n_suffix=8, n_endpoints=2)
            ref_core, _meta = core_mask_v3(profile, side)
            ref_core_frac = float(ref_core.mean())
            result = copy_score(g, gen_mid, r, ref_core, TAU_COPY)
        except ValueError:
            missing.add("ref_too_short")

    return {
        "item_id": item_id,
        "seed": seed,
        "arm": None,  # filled by the caller with the harness_arm
        "n_pre": n_pre,
        "n_suf": n_suf,
        "n_mid": n_mid,
        "ref_core_frac": ref_core_frac,
        "copy_max": result["copy_max"],
        "near_copy": result["near_copy"],
        "copy_gen_frame": result["copy_gen_frame"],
        "copy_ref_frame": result["copy_ref_frame"],
        "missing": sorted(missing),
    }


# --- per-variant driver ------------------------------------------------------
def process_variant(feats: Feats, corpus: Corpus, variant_rel: str) -> dict:
    """Score one gen variant; returns {harness_arm, gen, rows(list), coverage}."""
    vdir = REPO_ROOT / variant_rel
    meta = _read_meta(vdir / "meta.yaml")
    harness_arm = meta.get("harness_arm") or meta.get("arm") or vdir.name
    arm = meta.get("arm", "")
    variant = meta.get("variant", "")
    gtype = grid_type(arm, variant)

    grid: dict[str, dict] = {}
    for ln in (vdir / "grid.jsonl").read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        grid[r["item_id"]] = r

    rows = []
    for v in sorted((vdir / "videos").glob("*.mp4")):
        item_id, _seed = _parse_stem(v.stem)
        row = grid.get(item_id)
        r = compute_row(feats, corpus, v, row, gtype)
        r["arm"] = harness_arm
        rows.append(r)

    cov = _coverage(rows)
    cov["grid_type"] = gtype
    return {"harness_arm": harness_arm, "gen": variant_rel, "rows": rows, "coverage": cov}


def _coverage(rows: list[dict]) -> dict:
    n = len(rows)
    defined = [r for r in rows if r["near_copy"] is not None
               and isinstance(r["copy_max"], float) and math.isfinite(r["copy_max"])]
    n_def = len(defined)
    n_near = sum(1 for r in defined if r["near_copy"])
    copy_max_mean = (round(sum(r["copy_max"] for r in defined) / n_def, 4) if n_def else None)
    copy_rate = round(100.0 * n_near / n_def, 2) if n_def else None
    return {
        "n": n,
        "copy_defined": n_def,
        "near_copy": n_near,
        "copy_rate_pct": copy_rate,
        "copy_max_mean": copy_max_mean,
    }


def _read_meta(path: Path) -> dict:
    """Tiny top-level ``key: value`` reader for the gen meta.yaml (no yaml dep needed)."""
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        if ln[:1] in (" ", "\t", "#") or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        v = v.split("  #", 1)[0].strip().strip("'\"")
        out[k.strip()] = v
    return out


# --- writers (atomic; idempotent) -------------------------------------------
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp-{os.getpid()}"
    tmp.write_text(text)
    os.replace(tmp, path)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def write_rows(eval_dir: Path, harness_arm: str, rows: list[dict]) -> Path:
    out = eval_dir / harness_arm / "rows.jsonl"
    _atomic_write(out, _jsonl(rows))
    return out


def write_meta(eval_dir: Path, eval_id: str, seq: int, created: str,
               results: list[dict]) -> Path:
    lines = [
        f"id: {eval_id}",
        f"seq: {seq}",
        "shelf: evals",
        f"created: '{created}'",
        "machine: dai (login CPU, numpy over stored features)",
        f"instrument: scripts/copy_metrics.py @ {_git_sha()}",
        f"copy_score: src/diffusion/transition_eval/m2_integrity.py (reused verbatim)",
        f"tau_copy: {TAU_COPY}",
        "definition:",
    ]
    for d in DEFINITION:
        lines.append(f"  - {json.dumps(d)}")
    lines.append(f"why: {json.dumps(WHY)}")
    lines.append(f"caveat: {json.dumps(CAVEAT)}")
    lines.append("arms_scored:")
    for res in results:
        c = res["coverage"]
        lines.append(f"  {res['harness_arm']}:")
        lines.append(f"    gen: {res['gen']}")
        lines.append(f"    rows: {c['n']}")
        lines.append(f"    grid_type: {c['grid_type']}")
        lines.append("    coverage: {" + ", ".join([
            f"copy_defined: {c['copy_defined']}",
            f"near_copy: {c['near_copy']}",
            f"copy_rate_pct: {c['copy_rate_pct']}",
            f"copy_max_mean: {c['copy_max_mean']}",
        ]) + "}")
    meta_p = eval_dir / "meta.yaml"
    _atomic_write(meta_p, "\n".join(lines) + "\n")
    return meta_p


def append_index_line(eval_id: str, n_arms: int, n_rows: int) -> bool:
    """Append the single INDEX.md line for this eval into the ## evals section, once.
    Returns True if a line was added (False if already present)."""
    index = REPO_ROOT / "store" / "INDEX.md"
    text = index.read_text()
    if f"`{eval_id}`" in text:
        return False
    num = int(eval_id.split("_", 1)[0])
    line = (f"{num}. `{eval_id}` — M2a copy rate vs each generation's OWN reference on the "
            f"grid-v3 population ({n_arms} arms, {n_rows} gens): "
            f"copy_score (src/diffusion/transition_eval/m2_integrity.py, tau=0.858) between the "
            f"gen's MID frames (mid_mask, n_pre/n_suf per grid type as handoff_metrics) and its "
            f"OWN reference's NON-core DINO frames (morph_profile + core_mask_v3 with the "
            f"reference's sidedness, as the harness computes a reference bundle). Fixes the "
            f"evals/028/030 pool-reference bug (near_copy was identical across arms because each "
            f"gen was scored vs POOL clips, never its own conditioning reference). CPU/login, "
            f"numpy only; NaN+`missing` where a feature is absent (idempotent). rows.jsonl are "
            f"store artifacts (not committed); definition/why/caveat in meta.yaml. "
            f"scripts/copy_metrics.py.")
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


# --- CLI --------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population",
                    default="misc/2026-09-17_feature_store/population_gridv3.json")
    ap.add_argument("--eval-id", default=None,
                    help="override the eval entry id (default 039_copy_gridv3__dai__<date>)")
    ap.add_argument("--date", default="2026-09-18")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="repo-relative variant dirs to restrict to (default: the population)")
    ap.add_argument("--no-index", action="store_true", help="do not touch store/INDEX.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print coverage; write nothing")
    args = ap.parse_args(argv)

    pop = json.loads((REPO_ROOT / args.population).read_text())
    variants = args.variants if args.variants is not None else pop["gen_variants"]

    eval_id = args.eval_id or f"039_copy_gridv3__dai__{args.date}"
    seq = int(eval_id.split("_", 1)[0])
    eval_dir = REPO_ROOT / "store" / "evals" / eval_id

    fs = FeatureStore(REPO_ROOT)
    feats = Feats(fs)
    corpus = Corpus(CORPUS_MANIFEST)

    results = []
    total_rows = 0
    for rel in variants:
        res = process_variant(feats, corpus, rel)
        total_rows += res["coverage"]["n"]
        if not args.dry_run:
            write_rows(eval_dir, res["harness_arm"], res["rows"])
        results.append(res)
        c = res["coverage"]
        print(f"[arm] {res['harness_arm']:<34} n={c['n']:>4}  "
              f"copy_def={c['copy_defined']:>4}  near_copy={c['near_copy']:>4}  "
              f"copy_rate={c['copy_rate_pct']}%  copy_max_mean={c['copy_max_mean']}  "
              f"[{c['grid_type']}]")

    if args.dry_run:
        print(f"[dry-run] {len(results)} arms, {total_rows} gens — nothing written")
        return 0

    meta_p = write_meta(eval_dir, eval_id, seq, args.date, results)
    print(f"[meta] {meta_p.relative_to(REPO_ROOT)}")
    if not args.no_index:
        added = append_index_line(eval_id, len(results), total_rows)
        print(f"[index] {'appended' if added else 'already present'}: {eval_id}")
    print(f"[done] {len(results)} arms, {total_rows} gens -> {eval_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
