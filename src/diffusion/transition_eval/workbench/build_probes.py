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
N_STATIC_CLIPS = 8

# AMPLITUDES ARE DERIVED FROM THE CORPUS, NOT INVENTED (advisor C5). The first
# construction used a single invented scalar per kind, which (a) injected a
# SUB-PIXEL per-frame translation — testing the flow estimator's noise floor rather
# than the metric — and (b) mixed units in the compound probe, putting e**6 ~ 403x
# cumulative zoom into "pan+zoom". The derivation, pre-declared in CONSULTATIONS.md
# before any corrected probe flow was computed, is read from the frozen artifact
# written by the derivation step; per-channel, in each channel's own units.
AMPLITUDES_JSON = paths.WB_OUT / "phase1/probe_amplitudes_derived.json"


def inject_amplitudes() -> dict:
    d = json.loads(AMPLITUDES_JSON.read_text())
    return d["total_amplitudes"]


def probe_path(kind: str, key: str, extra: str = "", rung: str = "") -> str:
    """Cache key. For INJECTED probes the key MUST include the RUNG: amplitudes
    changed between constructions and differ per rung, and a key that ignored them
    would silently reuse the first construction's flow while pairing it with the
    second construction's ground truth — the worst kind of stale-cache bug, because
    every number would still look plausible. Reversed probes are unchanged between
    constructions and keep their key (their flow is reused, saving GPU)."""
    salt = f"|{rung}|border_masked" if kind == "inj" else ""
    h = hashlib.sha1(f"{kind}|{key}|{extra}{salt}|{flowcache.CACHE_TAG}"
                     .encode()).hexdigest()[:16]
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

    from . import probe_ladder
    ladder = json.loads((paths.WB_OUT / "phase1/probe_ladder.json").read_text())
    amps_by_rung = ladder["ladder_amplitudes"]

    # Skip base clips the FROZEN texture gate already declares undefined — they are
    # not valid constructed-truth substrates, and building them would waste GPU.
    gates = paths.load_gates()
    tmin = gates["phase1"]["m1b_flow"]["min_pair_texture"]
    tex = np.load(paths.WB_CACHE / "texture.npz")["pair_texture"]
    usable, skipped = [], []
    for k in static_keys:
        i = fkeys.index(k)
        below = float((tex[i][core[i]] < tmin).mean())
        (skipped if below > 0.30 else usable).append((k, below))
    for k, b in skipped:
        log(f"  SKIP base clip {k}: {b:.0%} of core pairs below the frozen texture gate")
    static_keys = [k for k, _ in usable]

    log(f"injection: {len(static_keys)} substrates x {len(INJECT_KINDS)} kinds x "
        f"{len(probe_ladder.RUNGS)} rungs")
    for r in probe_ladder.RUNGS:
        log(f"  {r}: " + "  ".join(f"{n}={amps_by_rung[r][n]:.4f}"
                                   for n in ("tx", "ty", "log_scale", "rotation")))
    truth = {}
    t0 = time.time()
    for k in static_keys:
        frames, _ = load_frames(paths.clip_path(k), short_side=None)
        frames = flowcache.resize_for_flow(frames)
        for rung in probe_ladder.RUNGS:
            for kind in INJECT_KINDS:
                out = probe_path("inj", k, kind, rung)
                cum = acceptance.trajectory(kind, len(frames), amps_by_rung[rung])
                rel = acceptance.relative_params(cum)
                truth[f"{k}|{kind}|{rung}"] = rel.tolist()
                if pathlib_exists(out):
                    continue
                warped = acceptance.warp_frames(frames, cum)
                # BORDER_REFLECT mirror pixels are content moving the WRONG WAY and
                # were never part of the constructed truth. §3.2's first branch
                # ("where S provides a spatial effect mask, fit on its COMPLEMENT")
                # is the sanctioned pathway for excluding them; the invalid set is
                # known EXACTLY from the cumulative warp (a pixel is invalid iff its
                # source coordinate falls outside the source frame), so no radius and
                # no threshold is invented. For a PAIR both frames must be real.
                vm = acceptance.warp_valid_mask(frames.shape, cum)
                vpair = vm[:-1] & vm[1:]
                flow = raft.flow_pairs(np.ascontiguousarray(warped))
                _atomic_save(out, flow=flow, src=k, kind=kind, rung=rung,
                             truth=rel, valid=vpair)
        log(f"  {k} done ({time.time() - t0:.0f}s)")
    log(f"injection flows done in {time.time() - t0:.0f}s")

    raft.free()
    (PROBE_DIR / "manifest.json").write_text(json.dumps({
        "reversed_clips": cam_keys,
        "static_clips": static_keys,
        "inject_kinds": list(INJECT_KINDS),
        "rungs": list(probe_ladder.RUNGS),
        "inject_amplitudes_by_rung": amps_by_rung,
        "base_clips_skipped_texture": [{"clip": k, "frac_below_gate": b}
                                       for k, b in skipped],
        "amplitude_derivation": ladder["purpose"],
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
