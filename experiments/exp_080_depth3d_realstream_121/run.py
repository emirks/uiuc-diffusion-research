"""exp_080 — 3D camera transitions at FULL 121 frames over two REAL PLAYING streams.

Fork of exp_076 promoted to the D2 contract: both source clips play in lockstep
t=0..120; frames outside the transition window are source frames VERBATIM (pure
phases byte-identical, endpoint blocks included); inside the window the camera
flies out of scene A and into scene B, but every rendered frame uses the CURRENT
frame + per-frame stabilised depth of both live streams — the world never freezes.
"""

from __future__ import annotations

import json
import logging
import pathlib
import random
import sys
import time

import numpy as np
import PIL.Image
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion.exp_utils import TeeLogger, load_config, next_run_dir  # noqa: E402

from engine3d import cameras, depth, metrics, ops3d, videoio  # noqa: E402
from engine3d.render3d import MeshRenderer  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

log = logging.getLogger("exp080")


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S", stream=sys.stdout, force=True,
        )
        yaml.safe_dump(cfg, open(run_dir / "config_snapshot.yaml", "w"))

        inf, smp, tim = cfg["inference"], cfg["sampling"], cfg["timing"]
        NF = inf["num_frames"]
        dev = cfg["runtime"]["device"]
        rng = random.Random(cfg["runtime"]["seed"])

        # ---- contents: full 121f train-class clips, paired across classes ----
        clips_dir = REPO_ROOT / cfg["inputs"]["clips_dir"]
        train_classes = {p.name for p in
                         (REPO_ROOT / cfg["inputs"]["train_classes_from"]).iterdir()
                         if p.is_dir()}
        clip_paths = {p.stem: p for p in sorted(clips_dir.glob("*/*.mp4"))
                      if p.parent.name in train_classes}
        ids = sorted(clip_paths)
        log.info("contents: %d train-class clips from %d classes",
                 len(ids), len(train_classes))

        renderer = MeshRenderer(inf["width"], inf["height"], step=inf["mesh_step"])
        log.info("GL context: %s", renderer.renderer_name())

        cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        frames_cache: dict[str, np.ndarray] = {}
        depth_cache: dict[str, np.ndarray] = {}

        def clip_frames(cid: str) -> np.ndarray:
            if cid not in frames_cache:
                f = videoio.read_clip(clip_paths[cid])[:NF]
                assert f.shape[0] == NF, f"{cid}: {f.shape[0]} frames < {NF}"
                frames_cache[cid] = f
            return frames_cache[cid]

        def clip_depth(cid: str) -> np.ndarray:
            if cid not in depth_cache:
                npy = cache_dir / f"{cid}.npy"
                if npy.exists():
                    depth_cache[cid] = np.load(npy).astype(np.float32)
                else:
                    t = time.time()
                    d = depth.disparity_stack(clip_frames(cid), device=dev)
                    np.save(npy, d.astype(np.float16))
                    log.info("depth stack %-28s %.1fs  flicker=%.4f",
                             cid, time.time() - t, depth.flicker(d))
                    depth_cache[cid] = d
            return depth_cache[cid]

        def draw_timing() -> tuple[int, int]:
            w0, w1 = tim["window"]
            span = w1 - w0
            onset = int(round(w0 + rng.random() * tim["jitter_frac"] * span))
            release = int(round(w1 - rng.random() * tim["jitter_frac"] * span))
            return onset, release

        # ---- sample plan: cross-class content pairs --------------------------
        def draw_pair() -> dict:
            while True:
                a, b = rng.sample(ids, 2)
                if clip_paths[a].parent.name != clip_paths[b].parent.name:
                    return {"pair_id": f"{a}__{b}", "from": a, "to": b}

        plan = [draw_pair() for _ in range(smp["n_pairs"])]

        manifest: list[dict] = []
        vid_dir, strip_dir = run_dir / "videos", run_dir / "filmstrips"
        strip_dir.mkdir(parents=True, exist_ok=True)

        def render_one(entry: dict, op: ops3d.Operator3D, tag: str,
                       timing: tuple[int, int] | None = None) -> None:
            onset, release = timing if timing else draw_timing()
            op.amplitude *= smp["amplitude_scale"]
            A, B = clip_frames(entry["from"]), clip_frames(entry["to"])
            da, db = clip_depth(entry["from"]), clip_depth(entry["to"])
            t = time.time()
            clip = ops3d.render_transition_stream(renderer, op, A, B, da, db,
                                                  onset, release)
            # pure-phase identity is BY CONSTRUCTION; assert it stays that way
            assert np.array_equal(clip[:onset + 1], A[:onset + 1])
            assert np.array_equal(clip[release:], B[release:])
            r_in = ops3d.join_ratio(clip, onset + 1)
            r_out = ops3d.join_ratio(clip, release)
            za0 = depth.to_view_depth(da[onset + 1], op.depth_near, op.depth_far,
                                      op.depth_gamma)
            pi = metrics.parallax_index(clip[onset + 1: onset + 7], za0)
            stem = (f"{tag}__{entry['pair_id']}__{op.short()}"
                    f"__on{onset:03d}_re{release:03d}__{op.seed % 10**6:06d}")
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            ramp = np.linspace(onset, release, 7).astype(int).tolist()
            strip_idx = [0, 8] + ramp + [112, 120]
            PIL.Image.fromarray(videoio.filmstrip(clip, strip_idx)).save(
                strip_dir / f"{stem}.jpg", quality=88)
            manifest.append({
                "stem": stem, "tag": tag, "pair_id": entry["pair_id"],
                "from": entry["from"], "to": entry["to"], "family": op.path,
                "blend": op.blend, "easing": op.easing,
                "onset": onset, "release": release,
                "describe": op.describe(),
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in op.__dict__.items()},
                "join_ratio_in": round(r_in, 3), "join_ratio_out": round(r_out, 3),
                "parallax": pi, "render_s": round(time.time() - t, 2),
            })
            log.info("%-64s join=(%.2f,%.2f) PI=%.2f %.1fs",
                     stem[:64], r_in, r_out, pi["pi"], time.time() - t)

        def clean(op, **over):
            op.amplitude, op.sign, op.easing = 1.15, 1, "in_out_cubic"
            op.blend, op.blend_easing, op.blend_window = "crossfade", "smoothstep", 0.5
            op.handheld = op.fog = op.focus = op.dolly_zoom = 0.0
            op.motion_blur, op.dissolve = 1, "none"
            for k, v in over.items():
                setattr(op, k, v)
            return op

        show_t = (tim["showcase_onset"], tim["showcase_release"])

        # 1. camera-family showcase — everything else held fixed, fixed timing
        if smp["family_showcase"]:
            for fam in sorted(cameras.PATHS):
                render_one(plan[0], clean(ops3d.sample_operator(rng), path=fam),
                           "family", show_t)

        # 1b. optical-effect showcase — one axis at a time on a dolly base
        if smp["effect_showcase"]:
            for label, over in [
                ("dollyzoom", dict(path="dolly", dolly_zoom=1.0)),
                ("depthwipe", dict(path="dolly", blend="depth_wipe", wipe_band=0.2)),
                ("dissolve_fbm", dict(path="dolly", dissolve="fbm", dissolve_freq=1.6)),
                ("fog", dict(path="dolly", fog=1.6)),
                ("rackfocus", dict(path="dolly", focus=12.0)),
                ("handheld_mblur", dict(path="orbit", handheld=0.7, motion_blur=3)),
            ]:
                render_one(plan[0], clean(ops3d.sample_operator(rng), **over),
                           f"effect_{label}", show_t)

        # 2. counterfactual — one pair, many operators (same content x diff op)
        for _ in range(smp["n_counterfactual"]):
            render_one(plan[1], ops3d.sample_operator(rng), "counterfactual")

        # 3. shared operator — one operator, several pairs (same op x diff content)
        for k in range(smp["n_shared_operators"]):
            base = ops3d.sample_operator(rng)
            timing = draw_timing()   # op identity includes its timing: share it
            for entry in plan[: smp["n_pairs_per_shared_operator"]]:
                op = ops3d.Operator3D(**base.__dict__)
                render_one(entry, op, f"sharedop{k}", timing)

        # 4. diversity sample — random operators x random pairs
        for entry in plan[: smp["n_diverse_pairs"]]:
            render_one(entry, ops3d.sample_operator(rng), "diverse")

        json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=1)
        joins = ([m["join_ratio_in"] for m in manifest]
                 + [m["join_ratio_out"] for m in manifest])
        pis = [m["parallax"]["pi"] for m in manifest]
        log.info("rendered %d clips @ %d frames | join ratio median %.2f p90 %.2f "
                 "max %.2f | parallax median %.2f | mean %.0fs/clip",
                 len(manifest), NF, float(np.median(joins)),
                 float(np.percentile(joins, 90)), float(np.max(joins)),
                 float(np.median(pis)),
                 float(np.mean([m["render_s"] for m in manifest])))
        print(f"[done] {run_id} → {run_dir}")


if __name__ == "__main__":
    main()
