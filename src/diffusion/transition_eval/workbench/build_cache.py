"""The single GPU cache-build job (OPERATIONS §6 step 3).

Front-loads ALL GPU work. Afterwards every descriptor variant, exam, ablation and
acceptance test is CPU-on-cached-arrays, so the design loop never waits on Slurm
again — which is what lets §3.3's energy-gate epsilon be chosen from the observed
residual distribution without a GPU re-run.

Produces, all under $WB_CACHE:
  zca.npz      — corpus ZCA fitted on S-mask core frames (§1.1)   [CPU]
  anchors.npz  — whitened e_A/e_B, chord D distribution, min-D floor, low_D (§1.2) [CPU]
  null_*.npz   — 223 rendered-lerp nulls, DINO-embedded (§4.0)    [GPU]
  flow_*.npz   — 223 dense SEA-RAFT flow fields (§3.1)            [GPU]
  manifest.json — every pin needed to reproduce the above

THE TWO-CACHE SPLIT IS STRUCTURAL, NOT A CONVENTION. Corpus features are read
from the certified shared cache through bundles.ReadOnlyExtractor, whose
extract() RAISES — so even inside a GPU job, with a live CUDA context and a real
DINO model loaded for the nulls, a corpus-side cache miss stops the run instead
of silently recomputing a feature and writing it into the cache a future
certification's 1e-6 determinism bar depends on. The real extractor is handed
$WB_CACHE and nothing else.

Every artifact is idempotent (a warm entry is skipped) and written atomically via
a .tmp rename, so the job is safe to requeue on the preemptible partition.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import anchors, bundles, flowcache, nulls, paths, whitening


def log(msg: str) -> None:
    print(f"[cache {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-flow", action="store_true")
    ap.add_argument("--skip-nulls", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N clips")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    if device == "cpu" and not (args.skip_flow and args.skip_nulls):
        log("REFUSING: GPU work on CPU. Submit through Slurm (OPERATIONS §3).")
        return 1

    paths.WB_CACHE.mkdir(parents=True, exist_ok=True)
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    if args.limit:
        keys = keys[:args.limit]
    sidedness = paths.sidedness_of(corpus, keys)

    # --- corpus bundles: READ-ONLY against the certified cache -----------------
    log(f"loading {len(keys)} warm bundles (ReadOnlyExtractor — a miss raises)")
    bs = bundles.load_corpus_bundles(keys)
    log("bundles warm, zero decodes, certified cache untouched")

    # --- ZCA (§1.1) — CPU ------------------------------------------------------
    zca_path = paths.WB_CACHE / "zca.npz"
    if zca_path.exists():
        zca = whitening.load(zca_path)
        log(f"ZCA warm ({int(zca['n_frames'])} core frames)")
    else:
        X = whitening.core_frames(bs, sidedness)
        log(f"fitting ZCA on {len(X)} S-mask core frames x {X.shape[1]} dims")
        zca = whitening.fit_zca(X)
        s = whitening.sanity(zca, X)
        log(f"  whitened cov: mean diag {s['mean_diag']:.6f}, max |offdiag| "
            f"{s['max_abs_offdiag']:.2e}, ||C-I||_F/sqrt(d) {s['frobenius_dev_from_I']:.2e}")
        log(f"  eigenvalue floor {float(zca['eig_floor']):.3e} "
            f"({int(zca['n_floored'])} dims floored; raw cond "
            f"{float(zca['condition_number_raw']):.2e})")
        whitening.save(zca, zca_path)
        log(f"  wrote {zca_path.name}")

    # --- anchors + D distribution + min-D floor (§1.2) — CPU -------------------
    anc_path = paths.WB_CACHE / "anchors.npz"
    if anc_path.exists():
        log("anchors warm")
        anc = dict(np.load(anc_path))
    else:
        anc = anchors.corpus_anchors(bs, zca)
        np.savez_compressed(anc_path, **anc)
        D = anc["D"]
        log(f"anchors: chord D min {D.min():.3f} / median {np.median(D):.3f} / "
            f"max {D.max():.3f}; min-D floor (p{anc['min_d_percentile']:.0f}) "
            f"{float(anc['min_d_floor']):.3f} -> {int(anc['n_low_D'])} low_D clips flagged")
        log(f"  wrote {anc_path.name}")

    # --- rendered-lerp nulls (§4.0) — GPU -------------------------------------
    if not args.skip_nulls:
        from ..features import DinoExtractor
        log(f"building {len(keys)} rendered-lerp nulls (DINO -> $WB_CACHE only)")
        dino = DinoExtractor(paths.DINO_MODEL, device=device)
        t0 = time.time()
        for i, k in enumerate(keys):
            nulls.build_clip_null(paths.clip_path(k), dino, paths.WB_CACHE)
            if (i + 1) % 40 == 0:
                log(f"  nulls {i + 1}/{len(keys)} ({time.time() - t0:.0f}s)")
        dino.free()
        log(f"nulls done in {time.time() - t0:.0f}s")

    # --- dense optical flow (§3.1) — GPU --------------------------------------
    if not args.skip_flow:
        log(f"building flow for {len(keys)} clips "
            f"({flowcache.PINS['backbone']} @ {flowcache.FLOW_H}x{flowcache.FLOW_W}, "
            f"iters={flowcache.ITERS})")
        raft = flowcache.SeaRaftExtractor(device=device)
        t0 = time.time()
        for i, k in enumerate(keys):
            cache = flowcache.build_clip_flow(paths.clip_path(k), raft, paths.WB_CACHE)
            if i == 0:
                # Fail on the FIRST clip, not the 223rd. RAFT-family pyramids need
                # H,W % 8 == 0; the natural decode (short_side=320 -> 427x320) does
                # not satisfy it, so this asserts the resize actually landed.
                f = np.load(cache)["flow"]
                exp = (120, flowcache.FLOW_H, flowcache.FLOW_W, 2)
                if f.shape != exp or f.dtype != np.float16:
                    log(f"ABORT: flow is {f.shape}/{f.dtype}, expected {exp}/float16")
                    return 1
                mag = np.abs(f.astype(np.float32))
                log(f"  flow shape OK {f.shape} float16; |flow| mean {mag.mean():.3f} px, "
                    f"p99 {np.percentile(mag, 99):.2f} px, max {mag.max():.1f} px")
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                log(f"  flow {i + 1}/{len(keys)} ({el:.0f}s, "
                    f"eta {el / (i + 1) * (len(keys) - i - 1):.0f}s)")
        raft.free()
        log(f"flow done in {time.time() - t0:.0f}s")

    # --- manifest --------------------------------------------------------------
    man = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clips": len(keys),
        "corpus_root": str(paths.CORPUS_ROOT),
        "shared_cache_readonly": str(paths.SHARED_CACHE),
        "wb_cache": str(paths.WB_CACHE),
        "flow": flowcache.PINS,
        "nulls": nulls.PINS,
        "zca": {"fit_population": "s_mask_core_frames",
                "n_frames": int(zca["n_frames"]),
                "dim": int(zca["dim"]),
                "eig_floor_ratio": float(zca["eig_floor_ratio"]),
                "eig_floor": float(zca["eig_floor"]),
                "n_floored": int(zca["n_floored"]),
                "tag": str(zca["tag"])},
        "anchors": {"min_d_percentile": float(anc["min_d_percentile"]),
                    "min_d_floor": float(anc["min_d_floor"]),
                    "n_low_D": int(anc["n_low_D"])},
    }
    (paths.WB_CACHE / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"wrote manifest.json")

    n_flow = len(list(paths.WB_CACHE.glob("flow_*.npz")))
    n_null = len(list(paths.WB_CACHE.glob("null_*.npz")))
    log(f"CACHE COMPLETE: {n_flow} flow, {n_null} nulls, zca, anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
