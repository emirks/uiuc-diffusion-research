#!/usr/bin/env python
"""handoff_metrics — hand-off metrics for the grid-v3 population from stored features.

Per generation, using the colocated feature store (contract v2, clause 10):
  - ``dino_cls@dinov2b-r256`` (``feats [T,768]``, L2-normalized) of the GEN and of its
    CONDITION CLIPS (``eval_ladder/conds/<endpoint>_{start9,end9}.mp4``), and
  - ``cotracker3@g20-m384-v2`` (``tracks [T,N,2]``, ``vis [T,N]``) of the same,

it computes the four hand-off scalars whose FIXED definitions are frozen in
``misc/2026-09-17_feature_store/BRIEF_OP3_handoff.md`` (reproduced verbatim in DEFINITIONS):

  identity_A, identity_B, motion_A, motion_B, seam_free

The conditioning-window rule (``n_pre`` / ``n_suf``) is per grid type / per external, and the
hand-off window length is ``K = max(2, round(fps/3))`` frames with ``fps = probe_fps(gen)``.

A missing feature (e.g. the externals or a condition clip before its extraction lands) yields a
NaN metric plus a ``missing`` entry naming the (namespace, role); the script never crashes and is
idempotent per gen (each run rewrites ``rows.jsonl`` / ``meta.yaml`` deterministically and appends
the INDEX line only when absent). CPU only; numpy over stored features; no GPU, no backbones.

Output: ``store/evals/<NNN>_handoff_gridv3__dai__<date>/<harness_arm>/rows.jsonl`` (one row per gen)
+ ``meta.yaml`` + one appended ``store/INDEX.md`` line.

Library imports: ``diffusion.feature_store`` (path-is-identity store), ``diffusion.transition_eval``
(``motion_fidelity``, ``probe_fps``). Read first: ``store/FEATURES.md``.
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
from diffusion.transition_eval.motion import motion_fidelity  # noqa: E402
from diffusion.transition_eval.video_io import probe_fps  # noqa: E402

# --- fixed pins -------------------------------------------------------------
DINO_NS = "dino_cls@dinov2b-r256"
TRACK_NS = "cotracker3@g20-m384-v2"
CONDS_DIR = REPO_ROOT / "eval_ladder" / "conds"
EXTERNAL_ARMS = ("refvfx", "vap", "vfxmaster")
# v4 evals that carry the seam z-scores: internal (028) and externals (030).
SEAM_EVALS = ("028_grid_v3_paper_arms__dai__2026-09-07",
              "030_external_zs_authornative__dai__2026-09-12")
SEED_SUFFIXES = ("__s42", "__s43")  # the two grid-v3 seeds (fast path for the stem match)

# The DEFINITIONS block, verbatim from BRIEF_OP3_handoff.md — copied into the eval meta.yaml.
DEFINITIONS = [
    "Conditioning windows: HF-grid rows (121 f): n_pre = 9, n_suf = 8 if sided == \"two\" else 0. "
    "ED-grid rows (81 f, frame-0 anchor): n_pre = 1, n_suf = 0. Externals (VAP/VFXMaster 49 f, "
    "refVFX 33 f, frame-0 conditioning): n_pre = 1, n_suf = 0.",
    "Hand-off window length K = max(2, round(fps / 3)) frames (~1/3 s; 8 at 24 fps, 3 at 9.72 fps, "
    "5 at 15 fps). fps via probe_fps(gen_video).",
    "identity_A = mean over t in [n_pre, n_pre+K) of cos(f_gen[t], f_condA[n_pre-1]) where f_condA "
    "= dino of <endpoint>_start9.mp4 (the last GIVEN frame; for n_pre=1 that is frame 0). "
    "identity_B (two-sided only) = mean over t in [T-n_suf-K, T-n_suf) of cos(f_gen[t], "
    "f_condB[T_B - n_suf]) where f_condB = dino of <endpoint>_end9.mp4 and T_B = 9 (the first given "
    "suffix frame = frame 1 of end9). Else NaN.",
    "motion_A = motion_fidelity between the gen's tracks/vis sliced to frames [0, n_pre+K) and the "
    "start clip's full tracks/vis (9 f) — only when n_pre >= 9 (a clip was given); NaN for n_pre = 1 "
    "(a frame has no motion). motion_B symmetric: gen frames [T-n_suf-K, T) vs the end clip's "
    "tracks; NaN unless two-sided.",
    "seam_free = 1 if the relevant seam z-scores <= 3 (prefix only for one-sided; prefix and suffix "
    "for two-sided) — read prefix_seam_z, suffix_seam_z from the v4 rows (evals 028|030), from "
    "Op-2's per_gen.jsonl if present else items.jsonl.",
    "Missing feature (e.g. externals before their extraction lands) -> NaN + a `missing` entry "
    "naming the namespace/role; never crash; idempotent per gen.",
]

NAN = float("nan")


# --- small helpers ----------------------------------------------------------
def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _cos_window(gen_feats: np.ndarray, anchor: np.ndarray, lo: int, hi: int) -> float:
    """mean over t in [lo, hi) of cosine(gen_feats[t], anchor). Feats are L2-normalized
    by the extractor; the cosine is still computed explicitly to stay robust."""
    T = gen_feats.shape[0]
    lo = max(0, lo)
    hi = min(hi, T)
    if hi <= lo:
        return NAN
    W = gen_feats[lo:hi].astype(np.float64)                 # [k,768]
    a = anchor.astype(np.float64)
    an = np.linalg.norm(a)
    wn = np.linalg.norm(W, axis=1)
    den = wn * an
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (W @ a) / den
    cos = cos[np.isfinite(cos)]
    if cos.size == 0:
        return NAN
    return float(cos.mean())


def _parse_stem(video_stem: str) -> tuple[str, int]:
    """A gen video stem is ``<grid item_id>__s<seed>``; split it into (item_id, seed)."""
    m = re.search(r"__s(\d+)$", video_stem)
    if not m:
        return video_stem, -1
    return video_stem[: m.start()], int(m.group(1))


# --- grid classification + window rule --------------------------------------
def grid_type(arm: str, variant: str) -> str:
    """``external`` / ``ED`` / ``HF`` — the tier that fixes n_pre/n_suf."""
    if arm in EXTERNAL_ARMS:
        return "external"
    if "ed81" in variant:
        return "ED"
    return "HF"


def windows(gtype: str, sided: str) -> tuple[int, int]:
    """(n_pre, n_suf) per the FIXED rule."""
    if gtype == "HF":
        return 9, (8 if sided == "two" else 0)
    # ED and externals both condition on frame 0.
    return 1, 0


# --- seam z-score source ----------------------------------------------------
def _seam_eval_dir(harness_arm: str) -> Path | None:
    for name in SEAM_EVALS:
        d = REPO_ROOT / "store" / "evals" / name / harness_arm
        if d.is_dir():
            return d
    return None


def load_seam(harness_arm: str, vidset: set[str]) -> tuple[dict, str | None]:
    """Return ({video_stem: (prefix_seam_z, suffix_seam_z)}, source_label).

    Preference: Op-2's ``per_gen.jsonl`` (keyed (item_id, seed)) if present, else the v4
    ``c*/items.jsonl`` shards (keyed by the gen video stem, matched against ``vidset``)."""
    arm_dir = _seam_eval_dir(harness_arm)
    if arm_dir is None:
        return {}, None
    seam: dict[str, tuple[float, float]] = {}

    per_gen = arm_dir / "per_gen.jsonl"
    if per_gen.exists():
        # Op-2 rows are one-per-gen with explicit (item_id, seed).
        for ln in per_gen.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            iid, seed = r.get("item_id"), r.get("seed")
            stem = f"{iid}__s{seed}"
            if stem in vidset:
                seam[stem] = (_f(r.get("prefix_seam_z")), _f(r.get("suffix_seam_z")))
        return seam, f"per_gen.jsonl@{arm_dir.parent.name}"

    # else: the pooled items.jsonl shards (many rows per gen; seam is a gen property).
    n_rows = 0
    for shard in sorted(arm_dir.glob("c*/items.jsonl")):
        for ln in shard.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            iid = r.get("item_id", "")
            stem = _match_stem(iid, vidset)
            if stem is None or stem in seam:
                continue
            seam[stem] = (_f(r.get("prefix_seam_z")), _f(r.get("suffix_seam_z")))
            n_rows += 1
    return seam, (f"items.jsonl@{arm_dir.parent.name}" if seam else None)


def _match_stem(item_id: str, vidset: set[str]) -> str | None:
    """The v4 item_id is ``<gen video stem>__ref_<gt>``; recover the gen video stem by
    truncating at a seed boundary and confirming membership in ``vidset``."""
    for suf in SEED_SUFFIXES:
        i = item_id.find(suf)
        if i != -1:
            cand = item_id[: i + len(suf)]
            if cand in vidset:
                return cand
    for m in re.finditer(r"__s\d+", item_id):
        cand = item_id[: m.end()]
        if cand in vidset:
            return cand
    return None


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


# --- feature access with a per-clip cache -----------------------------------
class Feats:
    """Lazy feature loader with a small per-condition-clip cache.

    ``has`` is a cheap presence probe (two stats); ``get`` reads the npz only when present and
    (with ``cache=True``, for the condition clips shared across many gens) memoizes it. Gen
    features are loaded lazily by ``compute_row`` ONLY once the matching condition feature is
    present — so a run before the condition features land touches no gen npz."""

    def __init__(self, fs: FeatureStore):
        self.fs = fs
        self._cache: dict[tuple[str, str], dict | None] = {}

    def has(self, video: Path, ns: str) -> bool:
        return self.fs.has(video, ns)

    def get(self, video: Path, ns: str, *, cache: bool = False):
        if cache:
            k = (str(video), ns)
            if k not in self._cache:
                self._cache[k] = self.fs.get(video, ns) if self.fs.has(video, ns) else None
            return self._cache[k]
        return self.fs.get(video, ns) if self.fs.has(video, ns) else None


# --- per-gen metric computation ---------------------------------------------
def compute_row(feats: Feats, gen_video: Path, endpoint: str, sided: str,
                gtype: str, seam: dict) -> dict:
    item_id, seed = _parse_stem(gen_video.stem)
    n_pre, n_suf = windows(gtype, sided)
    two_sided = n_suf > 0
    try:
        fps = float(probe_fps(gen_video))
    except Exception:
        fps = NAN
    K = max(2, int(round(fps / 3.0))) if math.isfinite(fps) else NAN

    missing: set[str] = set()
    identity_A = identity_B = motion_A = motion_B = NAN
    Kok = math.isfinite(K)

    condA = CONDS_DIR / f"{endpoint}_start9.mp4"
    condB = CONDS_DIR / f"{endpoint}_end9.mp4"

    # lazy, memoized gen-feature loader: a gen npz is read ONLY once its matching condition
    # feature is present (so a pre-extraction run touches no gen npz).
    _gcache: dict[str, dict | None] = {}

    def gen(ns: str):
        if ns not in _gcache:
            _gcache[ns] = feats.get(gen_video, ns) if feats.has(gen_video, ns) else None
        return _gcache[ns]

    # identity_A (always defined) --------------------------------------------
    if Kok:
        cA = feats.get(condA, DINO_NS, cache=True) if condA.exists() else None
        g_ok = feats.has(gen_video, DINO_NS)
        if cA is None:
            missing.add(f"{DINO_NS}:condA")
        if not g_ok:
            missing.add(f"{DINO_NS}:gen")
        if cA is not None and g_ok:
            fg, fA = gen(DINO_NS)["feats"], cA["feats"]
            ai = n_pre - 1
            if 0 <= ai < fA.shape[0]:
                identity_A = _cos_window(fg, fA[ai], n_pre, n_pre + K)

    # identity_B (two-sided only) --------------------------------------------
    if two_sided and Kok:
        cB = feats.get(condB, DINO_NS, cache=True) if condB.exists() else None
        g_ok = feats.has(gen_video, DINO_NS)
        if cB is None:
            missing.add(f"{DINO_NS}:condB")
        if not g_ok:
            missing.add(f"{DINO_NS}:gen")
        if cB is not None and g_ok:
            fg, fB = gen(DINO_NS)["feats"], cB["feats"]
            T = fg.shape[0]
            bi = fB.shape[0] - n_suf                 # T_B - n_suf (= 1 for a 9-frame end9)
            if 0 <= bi < fB.shape[0]:
                identity_B = _cos_window(fg, fB[bi], T - n_suf - K, T - n_suf)

    # motion_A (n_pre >= 9 only) ---------------------------------------------
    if Kok and n_pre >= 9:
        cA_tr = feats.get(condA, TRACK_NS, cache=True) if condA.exists() else None
        g_ok = feats.has(gen_video, TRACK_NS)
        if cA_tr is None:
            missing.add(f"{TRACK_NS}:condA")
        if not g_ok:
            missing.add(f"{TRACK_NS}:gen")
        if cA_tr is not None and g_ok:
            g_tr = gen(TRACK_NS)
            hi = n_pre + K
            motion_A = motion_fidelity(g_tr["tracks"][0:hi], g_tr["vis"][0:hi],
                                       cA_tr["tracks"], cA_tr["vis"])

    # motion_B (two-sided only) ----------------------------------------------
    if two_sided and Kok:
        cB_tr = feats.get(condB, TRACK_NS, cache=True) if condB.exists() else None
        g_ok = feats.has(gen_video, TRACK_NS)
        if cB_tr is None:
            missing.add(f"{TRACK_NS}:condB")
        if not g_ok:
            missing.add(f"{TRACK_NS}:gen")
        if cB_tr is not None and g_ok:
            g_tr = gen(TRACK_NS)
            T = g_tr["tracks"].shape[0]
            lo = T - n_suf - K
            motion_B = motion_fidelity(g_tr["tracks"][lo:T], g_tr["vis"][lo:T],
                                       cB_tr["tracks"], cB_tr["vis"])

    # seam_free (from the v4 rows) -------------------------------------------
    seam_free = NAN
    sv = seam.get(gen_video.stem)
    if sv is None:
        missing.add("seam")
    else:
        p_z, s_z = sv
        if two_sided:
            ok = math.isfinite(p_z) and math.isfinite(s_z) and p_z <= 3.0 and s_z <= 3.0
            have = math.isfinite(p_z) and math.isfinite(s_z)
        else:
            ok = math.isfinite(p_z) and p_z <= 3.0
            have = math.isfinite(p_z)
        seam_free = (1.0 if ok else 0.0) if have else NAN
        if not have:
            missing.add("seam")

    return {
        "item_id": item_id,
        "seed": seed,
        "arm": None,  # filled by the caller with the harness_arm
        "n_pre": n_pre,
        "n_suf": n_suf,
        "K": K if isinstance(K, int) else NAN,
        "fps": round(fps, 4) if math.isfinite(fps) else NAN,
        "identity_A": identity_A,
        "identity_B": identity_B,
        "motion_A": motion_A,
        "motion_B": motion_B,
        "seam_free": seam_free,
        "missing": sorted(missing),
    }


# --- per-variant driver ------------------------------------------------------
def process_variant(fs: FeatureStore, feats: Feats, variant_rel: str) -> dict:
    """Score one gen variant; returns {harness_arm, gen, rows(list), coverage}."""
    vdir = REPO_ROOT / variant_rel
    meta = _read_meta(vdir / "meta.yaml")
    harness_arm = meta.get("harness_arm") or meta.get("arm") or vdir.name
    arm = meta.get("arm", "")
    variant = meta.get("variant", "")
    gtype = grid_type(arm, variant)

    # grid.jsonl -> item_id -> (endpoint, sided)
    grid: dict[str, tuple[str, str]] = {}
    for ln in (vdir / "grid.jsonl").read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        grid[r["item_id"]] = (r.get("endpoint"), r.get("sided", "one"))

    videos = sorted((vdir / "videos").glob("*.mp4"))
    vidset = {v.stem for v in videos}
    seam, seam_src = load_seam(harness_arm, vidset)

    rows = []
    for v in videos:
        item_id, _seed = _parse_stem(v.stem)
        endpoint, sided = grid.get(item_id, (None, "one"))
        if endpoint is None:
            row = compute_row(feats, v, "", sided, gtype, seam)
            row["missing"] = sorted(set(row["missing"]) | {"grid_row"})
        else:
            row = compute_row(feats, v, endpoint, sided, gtype, seam)
        row["arm"] = harness_arm
        rows.append(row)

    cov = _coverage(rows)
    cov["seam_source"] = seam_src
    cov["grid_type"] = gtype
    return {"harness_arm": harness_arm, "gen": variant_rel, "rows": rows, "coverage": cov}


def _coverage(rows: list[dict]) -> dict:
    def fin(key):
        return sum(1 for r in rows if isinstance(r[key], float) and math.isfinite(r[key]))

    n = len(rows)
    n_two = sum(1 for r in rows if r["n_suf"] > 0)
    n_motionA = sum(1 for r in rows if r["n_pre"] >= 9)
    return {
        "n": n,
        "identity_A": fin("identity_A"),
        "identity_B": fin("identity_B"),
        "identity_B_defined": n_two,
        "motion_A": fin("motion_A"),
        "motion_A_defined": n_motionA,
        "motion_B": fin("motion_B"),
        "motion_B_defined": n_two,
        "seam_free": fin("seam_free"),
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
        f"instrument: scripts/handoff_metrics.py @ {_git_sha()}",
        "definitions:",
    ]
    for d in DEFINITIONS:
        lines.append(f"  - {json.dumps(d)}")
    lines.append("arms_scored:")
    for res in results:
        c = res["coverage"]
        lines.append(f"  {res['harness_arm']}:")
        lines.append(f"    gen: {res['gen']}")
        lines.append(f"    rows: {c['n']}")
        lines.append(f"    grid_type: {c['grid_type']}")
        lines.append(f"    seam_source: {c['seam_source']}")
        lines.append("    coverage: {" + ", ".join([
            f"identity_A: {c['identity_A']}",
            f"identity_B: {c['identity_B']}/{c['identity_B_defined']}",
            f"motion_A: {c['motion_A']}/{c['motion_A_defined']}",
            f"motion_B: {c['motion_B']}/{c['motion_B_defined']}",
            f"seam_free: {c['seam_free']}",
        ]) + "}")
    meta_p = eval_dir / "meta.yaml"
    _atomic_write(meta_p, "\n".join(lines) + "\n")
    return meta_p


def append_index_line(eval_id: str, n_arms: int, n_rows: int) -> bool:
    """Append the single INDEX.md line for this eval into the ## evals section, once.
    Returns True if a line was added (False if already present)."""
    index = REPO_ROOT / "store" / "INDEX.md"
    text = index.read_text()
    tag = eval_id.split("_", 1)[0]  # "038"
    if f"`{eval_id}`" in text:
        return False
    line = (f"38. `{eval_id}` — hand-off metrics on the grid-v3 population "
            f"({n_arms} arms, {n_rows} gens): identity_A/B, motion_A/B, seam_free from the stored "
            f"`dino_cls@dinov2b-r256` + `cotracker3@g20-m384-v2` features of each gen and its "
            f"condition clips (`eval_ladder/conds/<endpoint>_{{start9,end9}}.mp4`); windows n_pre/n_suf "
            f"per grid type, K = max(2, round(fps/3)); seam_free from the v4 rows (evals 028|030). "
            f"CPU/login, numpy only; NaN+`missing` where a feature is not yet extracted (idempotent). "
            f"rows.jsonl are store artifacts (not committed); definitions in meta.yaml. "
            f"scripts/handoff_metrics.py.")
    lines = text.splitlines()
    # find the ## evals section and the header of the next section after it.
    try:
        ev = next(i for i, ln in enumerate(lines) if ln.strip() == "## evals")
    except StopIteration:
        raise SystemExit("INDEX.md: no '## evals' section")
    nxt = next((i for i in range(ev + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    insert_at = nxt
    while insert_at > ev + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1                                    # skip trailing blank lines
    lines.insert(insert_at, line)
    _atomic_write(index, "\n".join(lines) + "\n")
    return True


# --- CLI --------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population",
                    default="misc/2026-09-17_feature_store/population_gridv3.json")
    ap.add_argument("--eval-id", default=None,
                    help="override the eval entry id (default 038_handoff_gridv3__dai__<date>)")
    ap.add_argument("--date", default="2026-09-18")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="repo-relative variant dirs to restrict to (default: the population)")
    ap.add_argument("--no-index", action="store_true", help="do not touch store/INDEX.md")
    ap.add_argument("--dry-run", action="store_true", help="compute + print coverage; write nothing")
    args = ap.parse_args(argv)

    pop = json.loads((REPO_ROOT / args.population).read_text())
    variants = args.variants if args.variants is not None else pop["gen_variants"]

    eval_id = args.eval_id or f"038_handoff_gridv3__dai__{args.date}"
    seq = int(eval_id.split("_", 1)[0])
    eval_dir = REPO_ROOT / "store" / "evals" / eval_id

    fs = FeatureStore(REPO_ROOT)
    feats = Feats(fs)

    results = []
    total_rows = 0
    for rel in variants:
        res = process_variant(fs, feats, rel)
        total_rows += res["coverage"]["n"]
        if not args.dry_run:
            write_rows(eval_dir, res["harness_arm"], res["rows"])
        results.append(res)
        c = res["coverage"]
        print(f"[arm] {res['harness_arm']:<34} n={c['n']:>4}  "
              f"idA={c['identity_A']}  idB={c['identity_B']}/{c['identity_B_defined']}  "
              f"mA={c['motion_A']}/{c['motion_A_defined']}  mB={c['motion_B']}/{c['motion_B_defined']}  "
              f"seam={c['seam_free']}  [{c['grid_type']}] seam_src={c['seam_source']}")

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
