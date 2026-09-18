#!/usr/bin/env python
"""lens_gate_check — the store-port gate for the competitor lenses.

The paper's committed competitor-metric rows (``their_metrics/rows_v3/vap_author_native/*.json``,
impl_sha ``d63935f4``, produced by the legacy in-process embedders) are the reference. This gate
RECOMPUTES the same lenses through the feature-store path (``score_batch --store``, impl_sha
``8a808635``; identical metric arithmetic, only ``score_batch.py`` bytes changed) for every VAP
author-native gen whose gen AND reference clip already have all four per-video lens namespaces in the
store, and reports the per-lens max |Δ|.

Tolerances (coordinator-set):
  * embedding lenses  (clip_sim_ref, clip_sim_input, motion_smoothness, videoprism_sim_ref,
    videoprism_sim_input, dynamic_degree_mean_mag): **report ≤ 1e-4** (GPU float nondeterminism across
    nodes), **FAIL (exit 1) if any exceeds 1e-3** — a real port error.
  * det_motion_fidelity: **≤ 1e-2, reported separately** (CoTracker re-tracking is not bit-reproducible;
    the store's VAP tracks were re-extracted tonight, the paper's came from ``.track_cache``). Does NOT
    gate the exit code.

Needs a GPU only if features are missing; with the four namespaces present it runs on CPU
(``--device cpu``; the input-start-frame CLIP embed on CPU is fine). Eligibility is checked per row, so
before the gate job fills VAP + its refs this prints "0 eligible" and exits 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THEIR_DIR = REPO_ROOT / "misc" / "2026-08-13_baseline_metric_table" / "their_metrics"
DEFAULT_ROWS_DIR = THEIR_DIR / "rows_v3" / "vap_author_native"

# per-video lens namespaces that must be present for a gen AND its reference to be eligible.
LENS_NS = ("clip_b32@r256", "videoprism@f16r288", "raft_mag@r256", "cotracker3@g20-m384-v2")

EMB_LENSES = ("clip_sim_ref", "clip_sim_input", "motion_smoothness",
              "videoprism_sim_ref", "videoprism_sim_input", "dynamic_degree_mean_mag")
TRACK_LENSES = ("det_motion_fidelity",)

EMB_REPORT_TOL = 1e-4
EMB_FAIL_TOL = 1e-3
TRACK_TOL = 1e-2


def _import_score_batch():
    if str(THEIR_DIR) not in sys.path:
        sys.path.insert(0, str(THEIR_DIR))
    import score_batch as sb
    return sb


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and x is not None and math.isfinite(float(x))


def eligible(fs, gen: Path, ref: Path) -> bool:
    return all(fs.has(gen, ns) for ns in LENS_NS) and all(fs.has(ref, ns) for ns in LENS_NS)


def compare(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs = [(frozen_row, recomputed_row), ...]; per-lens {max_abs_delta, n, n_over_report_tol}.
    Pure comparator — no I/O — so tests can drive it directly.
    Returns {lenses: {...}, fail: bool, worst_emb: (lens, delta)}."""
    stats: dict[str, dict] = {}
    all_lenses = EMB_LENSES + TRACK_LENSES
    for lens in all_lenses:
        tol = TRACK_TOL if lens in TRACK_LENSES else EMB_REPORT_TOL
        max_d = 0.0
        n = n_over = 0
        for frozen, recomputed in pairs:
            a, b = frozen.get(lens), recomputed.get(lens)
            if not (_finite(a) and _finite(b)):
                continue
            d = abs(float(a) - float(b))
            n += 1
            max_d = max(max_d, d)
            if d > tol:
                n_over += 1
        stats[lens] = {"max_abs_delta": max_d, "n": n, "n_over_tol": n_over, "tol": tol}
    # fail iff an EMBEDDING lens exceeds the hard 1e-3 bar.
    worst_emb = ("", 0.0)
    for lens in EMB_LENSES:
        if stats[lens]["max_abs_delta"] > worst_emb[1]:
            worst_emb = (lens, stats[lens]["max_abs_delta"])
    fail = any(stats[lens]["max_abs_delta"] > EMB_FAIL_TOL for lens in EMB_LENSES)
    return {"lenses": stats, "fail": fail, "worst_emb": worst_emb}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="competitor-lens store-port gate (rows_v3 vs score_batch --store)")
    ap.add_argument("--rows-dir", default=str(DEFAULT_ROWS_DIR),
                    help="frozen paper rows to reproduce (default: rows_v3/vap_author_native)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--store-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from diffusion.feature_store import FeatureStore
    fs = FeatureStore(Path(a.store_root).resolve() if a.store_root else REPO_ROOT)

    frozen_files = sorted(Path(a.rows_dir).glob("*.json"))
    if not frozen_files:
        raise SystemExit(f"no frozen rows under {a.rows_dir}")
    frozen_rows = [json.loads(f.read_text()) for f in frozen_files]

    # which rows are eligible (gen + ref features present)?
    todo = []
    for fr in frozen_rows:
        gen, ref = Path(fr["gen"]), Path(fr["reference_video"])
        if eligible(fs, gen, ref):
            todo.append(fr)
    if a.limit:
        todo = todo[:a.limit]
    print(f"[gate] rows_dir={a.rows_dir} total={len(frozen_rows)} eligible={len(todo)} "
          f"device={a.device}", flush=True)
    if not todo:
        print("[gate] 0 eligible — the four lens namespaces are not yet in the store for VAP + its "
              "refs. Run after the gate extraction job (clip_b32/videoprism/raft_mag) lands, then "
              "re-run. Exiting 0 (nothing to check).", flush=True)
        return 0

    sb = _import_score_batch()
    sha = sb.impl_sha()
    store = sb.StoreLenses(fs.root, a.device)
    M = sb.Models(a.device, THEIR_DIR / ".track_cache", store=store)

    pairs = []
    for i, fr in enumerate(todo):
        gen = Path(fr["gen"])
        try:
            rec = sb.score_one(gen, M, sha)
        except Exception as e:
            print(f"[gate] FAIL recompute {gen.name}: {type(e).__name__}: {e}", flush=True)
            continue
        pairs.append((fr, rec))
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"[gate] recomputed {i+1}/{len(todo)}", flush=True)

    rep = compare(pairs)
    print(f"\n[gate] frozen impl_sha={frozen_rows[0].get('impl_sha')}  recompute impl_sha={sha}  "
          f"pairs={len(pairs)}")
    print("  --- embedding lenses (report ≤ 1e-4; FAIL > 1e-3) ---")
    for lens in EMB_LENSES:
        s = rep["lenses"][lens]
        flag = "FAIL" if s["max_abs_delta"] > EMB_FAIL_TOL else ("over" if s["n_over_tol"] else "ok")
        print(f"    {lens:26s} max|Δ|={s['max_abs_delta']:.3e}  n={s['n']:4d}  "
              f"over_1e-4={s['n_over_tol']:4d}  [{flag}]")
    print("  --- track lens (report ≤ 1e-2; does NOT gate) ---")
    for lens in TRACK_LENSES:
        s = rep["lenses"][lens]
        flag = "over" if s["n_over_tol"] else "ok"
        print(f"    {lens:26s} max|Δ|={s['max_abs_delta']:.3e}  n={s['n']:4d}  "
              f"over_1e-2={s['n_over_tol']:4d}  [{flag}]")
    worst = rep["worst_emb"]
    if rep["fail"]:
        print(f"\n[gate] RESULT: FAIL — embedding lens {worst[0]} max|Δ|={worst[1]:.3e} > 1e-3 "
              f"(port error). Exit 1.", flush=True)
        return 1
    print(f"\n[gate] RESULT: PASS — worst embedding lens {worst[0]} max|Δ|={worst[1]:.3e} ≤ 1e-3. "
          f"Exit 0.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
