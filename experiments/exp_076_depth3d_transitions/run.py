"""exp_076 — 3D-plausible procedural transitions between two 9-frame buckets.

Output is start9 + a rendered middle + end9 (33 frames), not a padded 121-frame
clip: the buckets are copied through verbatim and the middle is a 2.5D render of
a virtual camera flying out of scene A and into scene B.
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

log = logging.getLogger("exp076")


def discover_pairs(cond_dir: pathlib.Path) -> dict[str, dict]:
    out = {}
    for p in sorted(cond_dir.glob("*_start9.mp4")):
        cid = p.name[: -len("_start9.mp4")]
        end = cond_dir / f"{cid}_end9.mp4"
        if end.exists():
            out[cid] = {"start9": p, "end9": end}
    return out


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

        inf, smp = cfg["inference"], cfg["sampling"]
        H, W = inf["height"], inf["width"]
        K, NM = inf["anchor_frames"], inf["n_middle"]
        dev = cfg["runtime"]["device"]
        rng = random.Random(cfg["runtime"]["seed"])

        cond_dir = REPO_ROOT / cfg["inputs"]["cond_dir"]
        clips = discover_pairs(cond_dir)
        ids = sorted(clips)
        log.info("found %d endpoint pairs; total frames per sample = %d",
                 len(ids), K + NM + K)

        renderer = MeshRenderer(W, H, step=inf["mesh_step"])
        log.info("GL context: %s", renderer.renderer_name())

        cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[tuple[str, str], np.ndarray] = {}
        disps: dict[tuple[str, str], np.ndarray] = {}

        def bucket(cid: str, which: str) -> np.ndarray:
            if (cid, which) not in frames:
                frames[(cid, which)] = videoio.read_clip(clips[cid][which])
            return frames[(cid, which)]

        def bucket_disp(cid: str, which: str) -> np.ndarray:
            """Disparity of the frame facing the transition, cached on disk."""
            key = (cid, which)
            if key in disps:
                return disps[key]
            npy = cache_dir / f"{cid}_{which}.npy"
            if npy.exists():
                disps[key] = np.load(npy)
            else:
                f = bucket(cid, which)
                face = f[-1] if which == "start9" else f[0]
                t = time.time()
                d = depth.disparity(face[None], device=dev)[0]
                log.info("depth %-34s %.1fs", f"{cid}/{which}", time.time() - t)
                np.save(npy, d)
                disps[key] = d
            return disps[key]

        # -- sample plan ----------------------------------------------------
        plan = []
        for _ in range(smp["n_pairs"]):
            a, b = rng.sample(ids, 2)
            plan.append({"pair_id": f"{a}__{b}", "from": a, "to": b})

        manifest: list[dict] = []
        vid_dir, strip_dir = run_dir / "videos", run_dir / "filmstrips"
        strip_dir.mkdir(parents=True, exist_ok=True)
        strip_idx = [0, 4, 8, 11, 14, 17, 20, 23, 24, 28, 32]

        def render_one(entry: dict, op: ops3d.Operator3D, tag: str) -> None:
            s9 = bucket(entry["from"], "start9")[:K]
            e9 = bucket(entry["to"], "end9")[-K:]
            da = bucket_disp(entry["from"], "start9")
            db = bucket_disp(entry["to"], "end9")
            t = time.time()
            clip = ops3d.render_transition(renderer, op, s9, e9, da, db, NM)
            d0, d1, r0, r1 = ops3d.seam_error(clip, K, NM)
            za = depth.to_view_depth(da, op.depth_near, op.depth_far, op.depth_gamma)
            pi = metrics.parallax_index(clip[K:K + 6], za)
            stem = f"{tag}__{entry['pair_id']}__{op.short()}__{op.seed % 10**6:06d}"
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            PIL.Image.fromarray(videoio.filmstrip(clip, strip_idx)).save(
                strip_dir / f"{stem}.jpg", quality=88)
            manifest.append({
                "stem": stem, "tag": tag, "pair_id": entry["pair_id"],
                "from": entry["from"], "to": entry["to"], "family": op.path,
                "blend": op.blend, "easing": op.easing,
                "describe": op.describe(), "params": dataclass_dict(op),
                "seam_mae_in": round(d0, 3), "seam_mae_out": round(d1, 3),
                "seam_ratio_in": round(r0, 3), "seam_ratio_out": round(r1, 3),
                "parallax": pi,
                "render_s": round(time.time() - t, 2),
            })
            log.info("%-52s seam=(%.2f,%.2f) PI=%.2f/%.2f rho=%+.2f %.1fs",
                     stem[:52], r0, r1, pi["pi"], pi["pi_pred"], pi["rho"],
                     time.time() - t)

        def dataclass_dict(op) -> dict:
            return {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in op.__dict__.items()}

        def clean(op, **over):
            """Strip every optional effect so one axis can be compared in isolation."""
            op.amplitude, op.sign, op.easing = 1.15, 1, "in_out_cubic"
            op.blend, op.blend_easing, op.blend_window = "crossfade", "smoothstep", 0.5
            op.handheld = op.fog = op.focus = op.dolly_zoom = 0.0
            op.motion_blur, op.dissolve = 1, "none"
            for k, v in over.items():
                setattr(op, k, v)
            return op

        # 1. camera-family showcase — everything else held fixed
        if smp["family_showcase"]:
            showcase = plan[0]
            for fam in sorted(cameras.PATHS):
                render_one(showcase, clean(ops3d.sample_operator(rng), path=fam),
                           "family")

        # 1b. optical-effect showcase — one axis at a time, on a dolly base, so the
        # contribution of each physically-motivated term is visible on its own.
        for label, over in [
            ("dollyzoom", dict(path="dolly", dolly_zoom=1.0)),
            ("motionblur", dict(path="orbit", motion_blur=4)),
            ("handheld", dict(path="dolly", handheld=0.8)),
            ("fog", dict(path="dolly", fog=1.6)),
            ("rackfocus", dict(path="dolly", focus=12.0)),
            ("depthwipe", dict(path="dolly", blend="depth_wipe", wipe_band=0.2)),
            ("dissolve_fbm", dict(path="dolly", dissolve="fbm", dissolve_freq=1.6)),
            ("dissolve_worley", dict(path="dolly", dissolve="worley", dissolve_freq=1.0)),
            ("dissolve_plane", dict(path="dolly", dissolve="plane", dissolve_freq=1.0)),
            ("dissolve_sphere", dict(path="dolly", dissolve="sphere", dissolve_freq=1.0)),
        ]:
            render_one(plan[0], clean(ops3d.sample_operator(rng), **over),
                       f"effect_{label}")

        # 2. counterfactual — one pair, many operators
        for _ in range(smp["n_counterfactual"]):
            render_one(plan[0], ops3d.sample_operator(rng), "counterfactual")

        # 3. shared operator — one operator, several pairs
        for k in range(smp["n_shared_operators"]):
            op = ops3d.sample_operator(rng)
            for entry in plan[: smp["n_pairs_per_shared_operator"]]:
                render_one(entry, op, f"sharedop{k}")

        # 4. diversity sample — random operators across random pairs
        for entry in plan:
            for _ in range(smp["n_operators_per_pair"]):
                render_one(entry, ops3d.sample_operator(rng), "diverse")

        json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=1)
        ratios = [m["seam_ratio_in"] for m in manifest] + \
                 [m["seam_ratio_out"] for m in manifest]
        pis = [m["parallax"]["pi"] for m in manifest]
        rhos = [m["parallax"]["rho"] for m in manifest]
        log.info("rendered %d clips | seam ratio (1.0 = as smooth as the content's "
                 "own motion) median %.2f p90 %.2f max %.2f | mean %.1fs/clip",
                 len(manifest), float(np.median(ratios)),
                 float(np.percentile(ratios, 90)), max(ratios),
                 float(np.mean([m["render_s"] for m in manifest])))
        log.info("parallax index (1.0 = flat/2D): median %.2f  p10 %.2f  "
                 "| depth-flow rho median %+.2f",
                 float(np.median(pis)), float(np.percentile(pis, 10)),
                 float(np.median(rhos)))
        print(f"[done] {run_id} → {run_dir}")


if __name__ == "__main__":
    main()
