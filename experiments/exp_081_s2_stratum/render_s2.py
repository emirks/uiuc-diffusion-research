"""ctt_v2 S2 renderer — 800 exact ops x 10 content pairs over two REAL PLAYING streams.

    SHARD=k NSHARDS=n python render_s2.py          # mass render, shardable + resumable
    MODE=smoke python render_s2.py                 # 3 ops end-to-end with hard asserts

WHAT MAKES THIS DIFFERENT FROM exp_077 D2 (the whole point of the stratum):

  D2 rendered a (target, reference) TUPLE: one exact op on 2 content pairs, with the reference
  clip rendered specifically to pair with that target. 2,728 ops x 2 contents. An op seen on two
  contents supports exactly ONE (ref, target) presentation, so anything incidentally shared
  between those two renders is a shortcut the model can latch onto instead of reading manner.

  S2 renders an OP BLOCK: one exact op on 10 ENDPOINT-DISJOINT content pairs, and there is no
  dedicated reference render at all. Reference and target are drawn from the same block at TRAIN
  time (90 ordered combinations per op, resampled every epoch), which is the direct training
  signal that the reference's content carries no information about the target.

INVARIANTS THIS FILE ENFORCES (each one is an assert or a gate, never a comment):
  * pure phases are the SOURCE FRAMES, byte-for-byte — `max_pure` is a MAX condition, never a
    mean (exp_077 shipped a mean here and 20 tuples with localised max_pure up to 240 passed);
  * an op's 10 clips share (shader, uniforms, easing, onset/release, flip, swap) EXACTLY —
    timing is part of op identity, so the retry axis is the CONTENT PAIR, never the timing;
  * the 10 pairs of an op use 20 DISTINCT endpoint clips, so any (ref, target) draw inside the
    block is content-disjoint by construction;
  * an op ships with exactly 10 gate-passed clips or it is DROPPED whole — a silently
    underfilled op would reintroduce the low-multiplicity regime for exactly the ops most
    likely to be interesting.
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
sys.path.insert(0, str(REPO_ROOT / "experiments/exp_077_synth_stratum"))

from diffusion.exp_utils import load_config  # noqa: E402

import d2_metrics  # noqa: E402
import streams_real as sr  # noqa: E402
from engine import operators, shaders, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = HERE / "config_s2.yaml"
log = logging.getLogger("s2")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


class ClipCache:
    """LRU cache of FULL 121-frame source clips (~111 MB decoded each)."""

    def __init__(self, root: Path, maxlen: int):
        self.root, self.maxlen = root, maxlen
        self.d: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self.hits = self.misses = 0

    def get(self, clip_id: str, mp4: str) -> np.ndarray:
        if clip_id in self.d:
            self.d.move_to_end(clip_id)
            self.hits += 1
            return self.d[clip_id]
        self.misses += 1
        arr = videoio.read_clip(self.root / mp4)
        self.d[clip_id] = arr
        while len(self.d) > self.maxlen:
            self.d.popitem(last=False)
        return arr


def strip_indices(total: int) -> list[int]:
    return [i for i in (0, 4, 8, 14, 22, 32, 44, 56, 68, 80, 92, 100, 108, 112, 116, 120)
            if i < total]


def save_strip(path: Path, clip: np.ndarray, idx: list[int], frame_w: int) -> None:
    sel = clip[idx]
    h, w = sel.shape[1:3]
    fh = int(round(h * frame_w / w))
    tiles = [np.asarray(PIL.Image.fromarray(f).resize((frame_w, fh), PIL.Image.LANCZOS))
             for f in sel]
    strip = np.full((fh, len(tiles) * (frame_w + 2) - 2, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        strip[:, k * (frame_w + 2): k * (frame_w + 2) + frame_w] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(strip).save(path, quality=90)


def rebuild_operator(o: dict) -> operators.Operator:
    """Reconstruct the frozen exact operator from its plan row — no resampling, ever."""
    op = operators.Operator(op_id=o["op_id"], shader=o["shader"], params=o["params"],
                            easing=o["easing"], flip=o["flip"], swap=o["swap"],
                            extension="none", aux_kind=None, aux_seed=0)
    assert op.op_id == o["op_id"], "operator identity drifted on reconstruction"
    return op


def main() -> None:
    mode = os.environ.get("MODE", "mass")
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    cfg = load_config(CONFIG_PATH)
    inf, smp, s2, gate = cfg["inference"], cfg["sampling"], cfg["s2"], cfg["gate"]
    H, W = inf["height"], inf["width"]
    T, K = inf["num_frames"], inf["anchor_frames"]

    plan = json.loads((HERE / "PLAN_S2.json").read_text())
    ops_plan, pairs = plan["ops"], plan["pairs"]
    m = plan["design"]["contents_per_op"]
    max_attempts = s2["max_pair_attempts"]

    root = REPO_ROOT / cfg["outputs"]["dir"] / ("smoke" if mode == "smoke" else "full")
    vid_dir, strip_dir, meta_dir = root / "videos", root / "filmstrips", root / "meta"
    for d in (vid_dir, strip_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    rows_path = meta_dir / f"clips_shard{shard:02d}.jsonl"
    ops_path = meta_dir / f"ops_shard{shard:02d}.jsonl"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    log.info("MODE=%s shard=%d/%d host=%s root=%s", mode, shard, nshards,
             os.uname().nodename, root)

    # ---- GL + the frozen shader bank -----------------------------------------------------
    runner = GLRunner(W, H)
    log.info("GL: %s", runner.renderer_name())
    raw = shaders.load_bank(Path(cfg["model"]["shader_bank"]))
    gated, _ = operators.validate_bank(runner, raw, tol=smp["endpoint_tol"])
    need = {o["shader"] for o in ops_plan}
    bank = {k: v for k, v in gated.items() if k in need}
    assert set(bank) == need, f"shader bank is missing planned shaders: {sorted(need - set(bank))}"

    clips_dir = REPO_ROOT / cfg["inputs"]["clips_dir"]
    cache = ClipCache(clips_dir, s2["clip_cache"])

    # endpoint-frame cache: gate-2 is a 2-frame check and must never cost a full decode
    z = np.load(REPO_ROOT / cfg["inputs"]["endpoint_frame_cache"])
    fa_cache = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
    fb_cache = {k[2:]: z[k] for k in z.files if k.startswith("b_")}

    mine = [i for i in range(len(ops_plan)) if i % nshards == shard]
    if mode == "smoke":
        mine, nshards, shard = mine[:3], 1, 0
    log.info("plan: %d ops x %d contents = %d clips | this shard: %d ops",
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
    idx_strip = strip_indices(T)
    t_start = time.time()
    n_render = n_accept = n_drop = 0
    pair_rejects: dict[str, int] = collections.defaultdict(int)   # advisor rider: log these
    shader_stat: dict[str, dict] = collections.defaultdict(lambda: {"rendered": 0, "accepted": 0})

    for n, oi in enumerate(mine):
        if oi in done_ops:
            continue
        o = ops_plan[oi]
        op = rebuild_operator(o)
        i0, j0 = o["phase"]["i0"], o["phase"]["j0"]
        p = sr.progress_ramp(T, K, op.easing, o["timing"]["onset"], o["timing"]["release"])

        accepted: list[dict] = []
        used_clips: set[str] = set()
        attempts = 0
        rejects: list[dict] = []

        for pi in o["candidates"]:
            if len(accepted) == m or attempts >= max_attempts:
                break
            pr = pairs[pi]
            a_id, b_id = pr["A"], pr["B"]
            # endpoint-disjointness ACROSS the op's accepted block (ruling 1b/2)
            if a_id in used_clips or b_id in used_clips:
                continue
            attempts += 1

            # gate-2 first: a 2-frame endpoint-identity check is ~1000x cheaper than a render
            d0, d1 = operators.check_operator(runner, bank, op, fa_cache[a_id], fb_cache[b_id])
            if max(d0, d1) > smp["endpoint_tol_operator"]:
                rejects.append({"pair_id": pi, "stage": "gate2",
                                "gate2": [round(d0, 4), round(d1, 4)]})
                pair_rejects[f"{a_id}|{b_id}"] += 1
                continue

            a_src, b_src = cache.get(a_id, pr["A_mp4"]), cache.get(b_id, pr["B_mp4"])
            assert a_src.shape == (T, H, W, 3) and b_src.shape == (T, H, W, 3), \
                f"unexpected clip shape {a_src.shape}/{b_src.shape}"
            t_r = time.time()
            clip = sr.render_real(runner, bank, op, a_src, b_src, p)
            render_s = time.time() - t_r
            n_render += 1
            shader_stat[op.shader]["rendered"] += 1

            mm = d2_metrics.score_clip(clip, a_src, b_src, i0, j0)
            v = d2_metrics.verdict(mm, gate["tau"], assert1_tol=gate["assert1_tol"],
                                   seam_max=gate["seam_max"], m2_max=gate["m2_max_dq"])
            if not v["pass"]:
                failed = [k for k in ("assert1", "assert2", "m1", "m2") if not v[k]]
                rejects.append({"pair_id": pi, "stage": "gate", "failed": failed,
                                "max_pure": round(mm["assert1"]["max_pure"], 3),
                                "seam": round(mm["assert2"]["seam_max_ratio"], 3),
                                "m1_p10": round(mm["m1_p10"], 4),
                                "m2_max_dq": round(mm["m2_max_dq"], 4)})
                pair_rejects[f"{a_id}|{b_id}"] += 1
                continue

            slot = len(accepted)
            stem = f"s2_{oi:04d}_c{slot:02d}"
            # the contract, asserted per clip and not merely gated
            assert mm["assert1"]["max_pure"] <= gate["assert1_tol"]
            assert np.array_equal(clip[: i0 + 1], a_src[: i0 + 1]), \
                f"{stem}: pure-A phase is not the source frames"
            assert np.array_equal(clip[j0:], b_src[j0:]), \
                f"{stem}: pure-B phase is not the source frames"

            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            save_strip(strip_dir / f"{stem}.jpg", clip, idx_strip, s2["strip_frame_w"])
            row = {
                "stem": stem, "op_index": oi, "op_id": op.op_id, "slot": slot,
                "shader": op.shader, "params": op.params, "easing": op.easing,
                "flip": op.flip, "swap": op.swap, "extension": "none", "aux_kind": None,
                "timing": o["timing"], "phase": {"i0": i0, "j0": j0},
                "pair_id": pi, "A": a_id, "B": b_id,
                "gate2": [round(d0, 4), round(d1, 4)],
                "assert1": mm["assert1"], "assert2": mm["assert2"],
                "m1_p10": mm["m1_p10"], "m1_min": mm["m1_min"], "m1_mean": mm["m1_mean"],
                "m2_max_dq": mm["m2_max_dq"], "n_ramp": mm["n_ramp"],
                "m1_min_flag": bool(mm["m1_min"] < gate["m1_min_flag_threshold"]),
                "render_s": round(render_s, 2), "engine_git_commit": commit,
            }
            # NOT written yet — see the flush below. An op is all-or-nothing, so its clip rows
            # may only enter the manifest once the op is known to be complete.
            accepted.append(row)
            used_clips |= {a_id, b_id}
            n_accept += 1
            shader_stat[op.shader]["accepted"] += 1

        # ---- incidence integrity is a GATE (advisor ruling 1c) ---------------------------
        # All-or-nothing, and that has to include the MANIFEST, not just the media. The smoke
        # run caught this: rows were streamed as clips were accepted, so a later drop deleted
        # the mp4s but left their rows behind — a manifest claiming clips that do not exist.
        # Clip rows are therefore buffered and flushed only when the op is known complete.
        complete = len(accepted) == m
        if complete:
            for r in accepted:
                frow.write(json.dumps(r) + "\n")
        elif s2["drop_op_on_underfill"]:
            n_drop += 1
            for r in accepted:                       # a dropped op leaves NOTHING behind
                (vid_dir / f"{r['stem']}.mp4").unlink(missing_ok=True)
                (strip_dir / f"{r['stem']}.jpg").unlink(missing_ok=True)
            n_accept -= len(accepted)
            shader_stat[op.shader]["accepted"] -= len(accepted)
            log.warning("op %d (%s) DROPPED: only %d/%d slots filled in %d attempts",
                        oi, op.shader, len(accepted), m, attempts)

        distinct = sorted({c for r in accepted for c in (r["A"], r["B"])})
        fop.write(json.dumps({
            "op_index": oi, "op_id": op.op_id, "shader": op.shader,
            "complete": complete, "dropped": (not complete) and s2["drop_op_on_underfill"],
            "n_slots": len(accepted) if complete else 0, "attempts": attempts,
            "n_distinct_endpoint_clips": len(distinct) if complete else 0,
            "stems": [r["stem"] for r in accepted] if complete else [],
            "pair_ids": [r["pair_id"] for r in accepted] if complete else [],
            "rejects": rejects,
        }) + "\n")

        if (n + 1) % 5 == 0 or n == len(mine) - 1:
            el = time.time() - t_start
            log.info("shard %d: %d/%d ops | %d clips accepted | %d rendered "
                     "(overdraw %.2fx) | %d dropped | %.1f min | cache h/m=%d/%d",
                     shard, n + 1, len(mine), n_accept, n_render,
                     n_render / max(n_accept, 1), n_drop, el / 60, cache.hits, cache.misses)

    frow.close()
    fop.close()
    summary = {
        "shard": shard, "nshards": nshards, "mode": mode,
        "ops_attempted": len(mine), "clips_accepted": n_accept, "clips_rendered": n_render,
        "overdraw": round(n_render / max(n_accept, 1), 4), "ops_dropped": n_drop,
        "minutes": round((time.time() - t_start) / 60, 1),
        "per_shader": {k: v for k, v in sorted(shader_stat.items())},
        "pair_reject_counts": dict(sorted(pair_rejects.items(), key=lambda kv: -kv[1])[:40]),
        "engine_git_commit": commit,
    }
    (meta_dir / f"summary_shard{shard:02d}.json").write_text(json.dumps(summary, indent=1))
    log.info("shard %d DONE: %d clips, %d rendered (overdraw %.2fx), %d ops dropped, %.1f min",
             shard, n_accept, n_render, summary["overdraw"], n_drop, summary["minutes"])

    if mode == "smoke":
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        by_op = collections.defaultdict(list)
        for r in rows:
            by_op[r["op_index"]].append(r)
        ok = True
        for oi, rs in by_op.items():
            ids = {(r["op_id"], r["easing"], r["flip"], r["swap"],
                    json.dumps(r["params"], sort_keys=True),
                    round(r["timing"]["onset"], 6), round(r["timing"]["release"], 6))
                   for r in rs}
            clips_used = [c for r in rs for c in (r["A"], r["B"])]
            chk = {
                "op_index": oi, "shader": rs[0]["shader"], "n_slots": len(rs),
                "exact_op_identical_across_slots": len(ids) == 1,
                "n_distinct_endpoint_clips": len(set(clips_used)),
                "endpoint_disjoint": len(set(clips_used)) == 2 * len(rs),
                "max_pure_over_slots": max(r["assert1"]["max_pure"] for r in rs),
                "max_seam_over_slots": round(max(r["assert2"]["seam_max_ratio"] for r in rs), 3),
                "min_m1_p10_over_slots": round(min(r["m1_p10"] for r in rs), 4),
            }
            chk["pass"] = (chk["exact_op_identical_across_slots"] and chk["endpoint_disjoint"]
                           and chk["n_slots"] == m and chk["max_pure_over_slots"] == 0.0)
            ok = ok and chk["pass"]
            print(json.dumps(chk))

        # A DROPPED op is correct behaviour, not a failure — but it must leave nothing behind:
        # no manifest row, no mp4, no filmstrip. That is the invariant to check here.
        oplog = [json.loads(l) for l in ops_path.read_text().splitlines() if l.strip()]
        dropped = [r for r in oplog if r["dropped"]]
        orphan_rows = [r["op_index"] for r in dropped if r["op_index"] in by_op]
        orphan_files = [s for r in dropped for s in r["stems"]
                        if (vid_dir / f"{s}.mp4").exists() or (strip_dir / f"{s}.jpg").exists()]
        drop_clean = not orphan_rows and not orphan_files
        ok = ok and drop_clean
        print(json.dumps({"dropped_ops": len(dropped), "orphan_manifest_rows": orphan_rows,
                          "orphan_media_files": orphan_files, "drop_clean": drop_clean}))
        print(json.dumps({"SMOKE": "PASS" if ok else "FAIL", "n_complete_ops": len(by_op),
                          "n_dropped_ops": len(dropped)}, indent=1))
        (root / "SMOKE.json").write_text(json.dumps(
            {"verdict": "PASS" if ok else "FAIL", "n_complete_ops": len(by_op),
             "n_dropped_ops": len(dropped), "drop_clean": drop_clean,
             "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
