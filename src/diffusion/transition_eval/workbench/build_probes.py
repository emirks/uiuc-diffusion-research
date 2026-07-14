"""GPU pass 2 — flow for the §3.4 acceptance probes.

The §3.4 tests are graded on CONSTRUCTED TRUTH, which means the metric must be run
end-to-end (flow AND fit) on videos whose true camera motion we know independently.
That needs flow fields the main cache does not contain:

  REVERSED clips — for the reversal probe. The flow of a time-reversed clip is NOT
    the negation of the forward flow (RAFT(t+1 -> t) is a different estimate than
    -RAFT(t -> t+1)), so it has to be computed, not derived. Otherwise the probe
    would be grading an algebraic identity instead of the metric.
  INJECTED-TRAJECTORY clips — a STATIC clip warped by a known synthetic camera move.
    Static clips are chosen by the metric's own camera fits (the ones with the least
    camera motion), so the only motion in the warped result is the injected one and
    the ground truth is exact.

No videos are written to disk: SEA-RAFT runs directly on the constructed frame
arrays. Probe flows go to $WB_CACHE/probes/ under their own keys; the certified
shared cache is never touched (the corpus is read through bundles.ReadOnlyExtractor
as everywhere else).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from ..video_io import load_frames
from . import acceptance, flowcache, paths

PROBE_DIR = paths.WB_CACHE / "probes"
INJECT_KINDS = ("pan_x", "pan_y", "zoom", "rotate", "pan_zoom")
INJECT_AMPS = {"pan_x": 24.0, "pan_y": 24.0, "zoom": 0.25, "rotate": 0.20,
               "pan_zoom": 20.0}
N_STATIC_CLIPS = 8


def probe_path(kind: str, key: str, extra: str = "") -> str:
    h = hashlib.sha1(f"{kind}|{key}|{extra}|{flowcache.CACHE_TAG}".encode()).hexdigest()[:16]
    return str(PROBE_DIR / f"{kind}_{h}.npz")


def log(m: str) -> None:
    print(f"[probes {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("REFUSING: GPU work on CPU. Submit through Slurm.")
        return 1

    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    fits = np.load(paths.WB_OUT / "phase1/camera_fits.npz", allow_pickle=False)
    fkeys = [str(k) for k in fits["keys"]]
    params, defined, core = fits["params"], fits["defined"], fits["core_pairs"]

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    raft = flowcache.SeaRaftExtractor(device=device)

    # --- 1. reversal: every camera-tagged clip --------------------------------
    cam_classes = {c for c, v in corpus["classes"].items() if "camera" in v.get("tags", [])}
    cam_keys = [k for k in keys if corpus["clips"][k]["class"] in cam_classes]
    if args.limit:
        cam_keys = cam_keys[:args.limit]
    log(f"reversal: {len(cam_keys)} camera-tagged clips")
    t0 = time.time()
    for i, k in enumerate(cam_keys):
        out = probe_path("rev", k)
        if pathlib_exists(out):
            continue
        frames, _ = load_frames(paths.clip_path(k), short_side=None)
        frames = flowcache.resize_for_flow(frames)[::-1]          # TIME-REVERSED
        flow = raft.flow_pairs(np.ascontiguousarray(frames))
        _atomic_save(out, flow=flow, src=k, kind="reversed")
        if (i + 1) % 20 == 0:
            log(f"  reversed {i + 1}/{len(cam_keys)} ({time.time() - t0:.0f}s)")

    # --- 2. injected trajectories on the STATIC clips --------------------------
    # "static" = smallest total camera motion over defined core pairs, per the
    # metric's own fits. The clip supplies content; the injected move supplies the
    # only motion, so the ground truth is exact.
    motion = []
    for i, k in enumerate(fkeys):
        m = core[i] & defined[i]
        if m.sum() < 20:
            motion.append(np.inf)
            continue
        p = params[i][m]
        motion.append(float(np.abs(p[:, :2]).mean() + 50 * np.abs(p[:, 2:]).mean()))
    order = np.argsort(motion)
    static_keys = [fkeys[i] for i in order[:N_STATIC_CLIPS]]
    log(f"injection: {N_STATIC_CLIPS} most-static clips "
        f"(motion score {motion[order[0]]:.4f} .. {motion[order[N_STATIC_CLIPS-1]]:.4f})")
    for k in static_keys:
        log(f"    {k}")

    truth = {}
    t0 = time.time()
    for k in static_keys:
        frames, _ = load_frames(paths.clip_path(k), short_side=None)
        frames = flowcache.resize_for_flow(frames)
        for kind in INJECT_KINDS:
            out = probe_path("inj", k, kind)
            cum = acceptance.trajectory(kind, len(frames), INJECT_AMPS[kind])
            rel = acceptance.relative_params(cum)
            truth[f"{k}|{kind}"] = rel.tolist()
            if pathlib_exists(out):
                continue
            warped = acceptance.warp_frames(frames, cum)
            flow = raft.flow_pairs(np.ascontiguousarray(warped))
            _atomic_save(out, flow=flow, src=k, kind=kind, truth=rel)
    log(f"injection flows done in {time.time() - t0:.0f}s")

    raft.free()
    (PROBE_DIR / "manifest.json").write_text(json.dumps({
        "reversed_clips": cam_keys,
        "static_clips": static_keys,
        "inject_kinds": list(INJECT_KINDS),
        "inject_amplitudes": INJECT_AMPS,
        "n_static": N_STATIC_CLIPS,
        "flow_pins": flowcache.PINS,
        "ground_truth_relative_params": truth,
    }, indent=1))
    log(f"CACHE COMPLETE: {len(cam_keys)} reversed, "
        f"{len(static_keys) * len(INJECT_KINDS)} injected")
    return 0


def pathlib_exists(p: str) -> bool:
    import pathlib
    return pathlib.Path(p).exists()


def _atomic_save(path: str, **arrays) -> None:
    import pathlib
    p = pathlib.Path(path)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(p)


if __name__ == "__main__":
    raise SystemExit(main())
