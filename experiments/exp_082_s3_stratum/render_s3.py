"""ctt_v2 S3 mass renderer — 300 exact camera-ops x 6 content pairs = 1,800 clips.

    SHARD=k NSHARDS=n python render_s3.py

Structurally identical to render_s2.py (same op-block contract, same all-or-nothing incidence
gate, same content-pair-swap retry) but on the 3D engine: per-frame stabilised depth ->
displaced mesh -> one continuous camera trajectory across both live streams.

MAY NOT BE RUN until the 63-clip pilot has cleared its pre-committed bars AND the advisor has
signed off on the pilot evidence — that gate is the advisor's single mid-campaign sign-off.
This script refuses to start if PILOT_RESULT.json is missing or not signed off.

INVARIANTS ENFORCED HERE:
  * pure phases are the SOURCE FRAMES byte-for-byte, asserted per clip (MAX condition);
  * an op's 6 clips share every operator field AND the timing — the retry axis is the content
    pair, never the timing;
  * the 6 pairs use 12 DISTINCT endpoint clips, so any (ref, target) draw in the block is
    content-disjoint;
  * join ratio <= 2.0 at BOTH joins, gating;
  * an op ships with exactly 6 gate-passed clips or it is DROPPED whole, media AND manifest.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

from engine3d import depth, metrics, ops3d, videoio  # noqa: E402
from engine3d.render3d import MeshRenderer  # noqa: E402

log = logging.getLogger("s3render")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    cfg = load_config(HERE / "config_s3.yaml")
    inf, s3, gate = cfg["inference"], cfg["s3"], cfg["gate"]
    NF, dev = inf["num_frames"], cfg["runtime"]["device"]

    # ---- S3 WAS DROPPED FROM THIS DELIVERY (2026-07-25) ----------------------------------
    if (HERE / "S3_DROPPED.json").exists():
        d = json.loads((HERE / "S3_DROPPED.json").read_text())
        sys.exit("[s3] STRATUM DROPPED: " + d["trigger"] + "\n     " + d["finding"] +
                 "\n     See S3_DROPPED.json. Reviving S3 needs a learned defect detector or "
                 "an engine redesign with frustum-constrained camera paths — not a re-run.")

    # ---- the pilot gate is a HARD precondition ------------------------------------------
    pilot_path = HERE / "PILOT_RESULT.json"
    if not pilot_path.exists():
        sys.exit("[s3] REFUSING TO RUN: PILOT_RESULT.json is missing. The 63-clip pilot on bank "
                 "contents is the advisor's mandatory gate before the 1,800-clip render.")
    pilot = json.loads(pilot_path.read_text())
    if pilot.get("automated_verdict") != "PASS":
        sys.exit(f"[s3] REFUSING TO RUN: pilot automated verdict = "
                 f"{pilot.get('automated_verdict')}. Bars are pre-committed and may not be relaxed.")
    if pilot.get("visual_audit", {}).get("status", "").startswith("PENDING"):
        sys.exit("[s3] REFUSING TO RUN: the pilot's blind visual audit has not been recorded.")
    if not pilot.get("advisor_signoff"):
        sys.exit("[s3] REFUSING TO RUN: PILOT_RESULT.json carries no `advisor_signoff`. The "
                 "advisor's sign-off on pilot evidence is the one required mid-campaign gate.")

    plan = json.loads((HERE / "PLAN_S3.json").read_text())
    ops_plan, pairs = plan["ops"], plan["pairs"]
    m = plan["design"]["contents_per_op"]
    max_attempts = s3["max_pair_attempts"]

    root = REPO_ROOT / cfg["outputs"]["dir"] / "full"
    vid_dir, strip_dir, meta_dir = root / "videos", root / "filmstrips", root / "meta"
    for d in (vid_dir, strip_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    rows_path = meta_dir / f"clips_shard{shard:02d}.jsonl"
    ops_path = meta_dir / f"ops_shard{shard:02d}.jsonl"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    log.info("shard=%d/%d host=%s root=%s", shard, nshards, os.uname().nodename, root)

    pool = json.loads((REPO_ROOT / cfg["inputs"]["content_pool"]).read_text())
    clip_path = {e["clip_id"]: Path(e["mp4"])          # absolute; spans both banks
                 for e in pool["training"] + pool["reserved"]}
    cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]

    renderer = MeshRenderer(inf["width"], inf["height"], step=inf["mesh_step"])
    log.info("GL context: %s", renderer.renderer_name())

    frames_cache: collections.OrderedDict = collections.OrderedDict()
    depth_cache: collections.OrderedDict = collections.OrderedDict()

    def clip_frames(cid: str) -> np.ndarray:
        if cid not in frames_cache:
            f = videoio.read_clip(clip_path[cid])[:NF]
            assert f.shape[0] == NF, f"{cid}: {f.shape[0]} frames < {NF}"
            frames_cache[cid] = f
            while len(frames_cache) > 20:
                frames_cache.popitem(last=False)
        frames_cache.move_to_end(cid)
        return frames_cache[cid]

    def clip_depth(cid: str) -> np.ndarray:
        if cid not in depth_cache:
            npy = cache_dir / f"{cid}.npy"
            if npy.exists():
                d = np.load(npy).astype(np.float32)
            else:
                log.warning("depth cache MISS for %s — computing inline", cid)
                d = depth.disparity_stack(clip_frames(cid), device=dev)
                np.save(npy, d.astype(np.float16))
            depth_cache[cid] = d
            while len(depth_cache) > 20:
                depth_cache.popitem(last=False)
        depth_cache.move_to_end(cid)
        return depth_cache[cid]

    mine = [i for i in range(len(ops_plan)) if i % nshards == shard]
    log.info("plan: %d ops x %d = %d clips | this shard: %d ops",
             len(ops_plan), m, len(ops_plan) * m, len(mine))

    done_ops = set()
    if ops_path.exists():
        for line in ops_path.read_text().splitlines():
            if line.strip():
                done_ops.add(json.loads(line)["op_index"])
        log.info("resuming: %d ops already finalised in this shard", len(done_ops))

    commit = git_commit()
    frow = open(rows_path, "a", buffering=1)
    fop = open(ops_path, "a", buffering=1)
    t_start = time.time()
    n_render = n_accept = n_drop = 0
    fam_stat: dict = collections.defaultdict(lambda: {"rendered": 0, "accepted": 0})

    for n, oi in enumerate(mine):
        if oi in done_ops:
            continue
        o = ops_plan[oi]
        onset, release = o["onset"], o["release"]
        params = dict(o["params"])
        params["fog_color"] = tuple(params["fog_color"])

        accepted: list[dict] = []
        used: set[str] = set()
        attempts = 0
        rejects: list[dict] = []

        for pi in o["candidates"]:
            if len(accepted) == m or attempts >= max_attempts:
                break
            pr = pairs[pi]
            a_id, b_id = pr["A"], pr["B"]
            if a_id in used or b_id in used:
                continue
            attempts += 1

            op = ops3d.Operator3D(**params)     # a FRESH instance per clip: render_* mutates
            A, B = clip_frames(a_id), clip_frames(b_id)
            da, db = clip_depth(a_id), clip_depth(b_id)
            t_r = time.time()
            clip = ops3d.render_transition_stream(renderer, op, A, B, da, db, onset, release)
            render_s = time.time() - t_r
            n_render += 1
            fam_stat[o["family"]]["rendered"] += 1

            byte_exact = bool(np.array_equal(clip[:onset + 1], A[:onset + 1])
                              and np.array_equal(clip[release:], B[release:]))
            max_pure = float(max(np.abs(clip[:onset + 1].astype(np.int16)
                                        - A[:onset + 1].astype(np.int16)).max(),
                                 np.abs(clip[release:].astype(np.int16)
                                        - B[release:].astype(np.int16)).max()))
            r_in = ops3d.join_ratio(clip, onset + 1)
            r_out = ops3d.join_ratio(clip, release)
            # pure-phase identity is BY CONSTRUCTION here; a violation is an engine bug, not a
            # content problem, so it aborts rather than quietly rejecting the pair
            assert byte_exact and max_pure == 0.0, \
                f"op {oi} pair {pi}: pure phase is not byte-exact (max_pure={max_pure})"

            if max(r_in, r_out) > gate["join_max"]:
                rejects.append({"pair_id": pi, "stage": "join",
                                "join": [round(r_in, 3), round(r_out, 3)]})
                continue

            za0 = depth.to_view_depth(da[onset + 1], op.depth_near, op.depth_far, op.depth_gamma)
            pi_m = metrics.parallax_index(clip[onset + 1: onset + 7], za0)

            slot = len(accepted)
            stem = f"s3_{oi:04d}_c{slot:02d}"
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            ramp = np.linspace(onset, release, 7).astype(int).tolist()
            PIL.Image.fromarray(videoio.filmstrip(clip, [0, 8] + ramp + [112, 120])).save(
                strip_dir / f"{stem}.jpg", quality=88)

            accepted.append({
                "stem": stem, "op_index": oi, "op_id": o["op_id"], "slot": slot,
                "family": o["family"], "params": o["params"],
                "onset": onset, "release": release,
                "pair_id": pi, "A": a_id, "B": b_id,
                "byte_exact": byte_exact, "max_pure": max_pure,
                "join_in": round(r_in, 3), "join_out": round(r_out, 3),
                "join_max": round(max(r_in, r_out), 3),
                "parallax": pi_m, "describe": o["describe"],
                "render_s": round(render_s, 2), "engine_git_commit": commit,
            })
            used |= {a_id, b_id}
            n_accept += 1
            fam_stat[o["family"]]["accepted"] += 1

        complete = len(accepted) == m
        if complete:
            for r in accepted:
                frow.write(json.dumps(r) + "\n")
        elif s3["drop_op_on_underfill"]:
            n_drop += 1
            for r in accepted:
                (vid_dir / f"{r['stem']}.mp4").unlink(missing_ok=True)
                (strip_dir / f"{r['stem']}.jpg").unlink(missing_ok=True)
            n_accept -= len(accepted)
            fam_stat[o["family"]]["accepted"] -= len(accepted)
            log.warning("op %d (%s) DROPPED: %d/%d slots in %d attempts",
                        oi, o["family"], len(accepted), m, attempts)

        distinct = sorted({c for r in accepted for c in (r["A"], r["B"])})
        fop.write(json.dumps({
            "op_index": oi, "op_id": o["op_id"], "family": o["family"],
            "complete": complete, "dropped": (not complete) and s3["drop_op_on_underfill"],
            "n_slots": len(accepted) if complete else 0, "attempts": attempts,
            "n_distinct_endpoint_clips": len(distinct) if complete else 0,
            "stems": [r["stem"] for r in accepted] if complete else [],
            "rejects": rejects}) + "\n")

        if (n + 1) % 5 == 0 or n == len(mine) - 1:
            el = time.time() - t_start
            log.info("shard %d: %d/%d ops | %d clips | %d rendered (overdraw %.2fx) | "
                     "%d dropped | %.1f min | %.1f s/clip",
                     shard, n + 1, len(mine), n_accept, n_render,
                     n_render / max(n_accept, 1), n_drop, el / 60, el / max(n_render, 1))

    frow.close()
    fop.close()
    summary = {"shard": shard, "nshards": nshards, "ops_attempted": len(mine),
               "clips_accepted": n_accept, "clips_rendered": n_render,
               "overdraw": round(n_render / max(n_accept, 1), 4), "ops_dropped": n_drop,
               "minutes": round((time.time() - t_start) / 60, 1),
               "per_family": {k: v for k, v in sorted(fam_stat.items())},
               "engine_git_commit": commit}
    (meta_dir / f"summary_shard{shard:02d}.json").write_text(json.dumps(summary, indent=1))
    log.info("shard %d DONE: %d clips, %d rendered (overdraw %.2fx), %d dropped, %.1f min",
             shard, n_accept, n_render, summary["overdraw"], n_drop, summary["minutes"])


if __name__ == "__main__":
    main()
