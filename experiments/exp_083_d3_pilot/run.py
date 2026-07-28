"""exp_083 — D3 / S3 PILOT: depth-parallax 3D procedural transitions.

A look-and-decide sample (~110 tuples), not a stratum. Three things separate it
from the D2 (exp_077) shader stratum whose generalist missed its win bar:

1. **Content coupling.** D2 assumed operator _|_ content: a screen-space shader
   applies to any pair. The post-mortem said the losses concentrated exactly on
   the content-coupled donors (saint_glow -12.3, shadow -12.1, display_transition
   -10.3) — effects that attach to a foreground object and travel with it, which
   a screen-space wipe structurally cannot do. Here the dissolve field is sampled
   at **unprojected scene positions**, so it sticks to surfaces, parallaxes and
   foreshortens; the `subject`/`subject_fbm` families additionally centre the
   field on the foreground subject's own world position.
2. **Our endpoint bank.** Endpoints are sliced as REAL consecutive frames out of
   the curated 227-clip bank (`bank_tightened.json`), not the old dark/weak
   exp_062 conditioning dir. Nothing is fabricated, held or reversed.
3. **Varying clip lengths.** n_middle in {7, 15, 23, 31} with 9+9 anchors gives
   totals 25/33/41/49 — every one legal for the causal VAE (F = 8k+1).

Format is unchanged: `start9 (verbatim) + rendered middle + end9 (verbatim)`, so
conditioning fidelity is exact by construction rather than by gate.
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

log = logging.getLogger("exp083")


# --------------------------------------------------------------------------
# Operator recipes. `coupling` records WHAT the effect is tied to, which is the
# axis the D2 post-mortem says matters:
#   none    — pure camera move, nothing attached to the scene (contrast group)
#   depth   — tied to the depth field (wipe order, fog extinction, focus plane)
#   world   — a field sampled at scene positions; sticks to surfaces
#   subject — that field centred on the foreground object itself
# --------------------------------------------------------------------------
RECIPES: dict[str, dict] = {
    "bare_move":      dict(coupling="none",    dissolve="none"),
    "dolly_zoom":     dict(coupling="none",    dissolve="none", path="dolly",
                           dolly_zoom=0.85),
    "depth_wipe":     dict(coupling="depth",   dissolve="none", blend="depth_wipe",
                           wipe_band=0.20),
    "rack_defocus":   dict(coupling="depth",   dissolve="none", focus=11.0),
    "atmos_travel":   dict(coupling="depth",   dissolve="none", fog=1.5,
                           motion_blur=3),
    "world_fbm":      dict(coupling="world",   dissolve="fbm",    dissolve_freq=1.6),
    "world_worley":   dict(coupling="world",   dissolve="worley", dissolve_freq=1.0),
    "sweep_plane":    dict(coupling="world",   dissolve="plane",  dissolve_freq=1.0),
    "shell_sphere":   dict(coupling="world",   dissolve="sphere", dissolve_freq=1.0),
    "subject_shell":  dict(coupling="subject", dissolve="subject", dissolve_freq=1.0,
                           dissolve_edge=0.10),
    "subject_smoke":  dict(coupling="subject", dissolve="subject_fbm",
                           dissolve_freq=1.8, dissolve_edge=0.14, fog=1.1,
                           motion_blur=3),
}
CONTENT_COUPLED = [k for k, v in RECIPES.items() if v["coupling"] in ("world", "subject")]


def make_op(rng: random.Random, name: str, **over) -> ops3d.Operator3D:
    """A named recipe with its free parameters sampled and everything else off."""
    r = RECIPES[name]
    op = ops3d.sample_operator(rng)
    # Neutral baseline: only what the recipe asks for is switched on.
    op.amplitude, op.sign = 1.15, rng.choice([-1, 1])
    op.easing = rng.choice(ops3d.PATH_EASINGS)
    op.blend, op.blend_easing, op.blend_window = "crossfade", "smoothstep", 0.5
    op.handheld = op.fog = op.focus = op.dolly_zoom = 0.0
    op.motion_blur, op.dissolve = 1, "none"
    op.depth_near, op.depth_far = 1.0, rng.uniform(2.6, 4.6)
    op.depth_gamma = rng.uniform(0.7, 1.2)
    op.fovy = np.radians(rng.uniform(40.0, 68.0))
    for k, v in r.items():
        if k != "coupling":
            setattr(op, k, v)
    for k, v in over.items():
        setattr(op, k, v)
    op.recipe = name                       # noqa: attribute added for the manifest
    op.coupling = r["coupling"]
    return op


def dataclass_dict(op) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in op.__dict__.items()}


def strip_indices(k: int, n_mid: int) -> list[int]:
    """Filmstrip tiles: both anchors, both seams, and the middle sampled evenly."""
    n = k + n_mid + k
    mids = sorted({k + round(i * (n_mid - 1) / 4) for i in range(5)})
    return [0, k - 1] + mids + [k + n_mid, n - 1]


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
        K, NF = inf["anchor_frames"], inf["src_frames"]
        LENGTHS = list(inf["n_middle_options"])
        dev = cfg["runtime"]["device"]
        rng = random.Random(cfg["runtime"]["seed"])

        bank_dir = REPO_ROOT / cfg["inputs"]["bank_dir"]
        bank = json.load(open(REPO_ROOT / cfg["inputs"]["bank_filter"]))
        entries = {e["clip_id"]: e for e in bank["clips"]}
        stack_dir = REPO_ROOT / cfg["inputs"]["depth_stack_cache"]
        cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        for nm in LENGTHS:
            total = K + nm + K
            assert (total - 1) % 8 == 0, f"illegal length {total} (F must be 8k+1)"
        log.info("bank: %d clips pass bank_tightened (of %d); lengths %s -> totals %s",
                 len(entries), bank["n_input"], LENGTHS,
                 [K + nm + K for nm in LENGTHS])

        pool = sorted(rng.sample(sorted(entries), smp["n_pool"]))

        renderer = MeshRenderer(W, H, step=inf["mesh_step"])
        log.info("GL context: %s", renderer.renderer_name())

        # -- endpoint anchors: REAL consecutive frames sliced out of a real clip ---
        # A-role  start9 = frames[NF-9 : NF], the transition-facing frame is the last.
        # B-role  end9   = frames[0 : 9],     the transition-facing frame is the first.
        # Nothing is generated, extended, held or reversed.
        anchors: dict[tuple[str, str], np.ndarray] = {}
        disps: dict[tuple[str, str], np.ndarray] = {}
        SLICE = {"A": (NF - K, NF), "B": (0, K)}
        FACE = {"A": NF - 1, "B": 0}

        def anchor(cid: str, role: str) -> np.ndarray:
            if (cid, role) not in anchors:
                f = videoio.read_clip(bank_dir / entries[cid]["mp4"])
                assert f.shape == (NF, H, W, 3), f"{cid}: {f.shape}"
                lo, hi = SLICE[role]
                anchors[(cid, role)] = np.ascontiguousarray(f[lo:hi])
            return anchors[(cid, role)]

        def anchor_disp(cid: str, role: str) -> np.ndarray:
            """Disparity of the transition-facing frame only (2 per tuple)."""
            key = (cid, role)
            if key in disps:
                return disps[key]
            npy = cache_dir / f"{cid}_{role}.npy"
            stack = stack_dir / f"{cid}.npy"
            if npy.exists():
                d = np.load(npy)
            elif stack.exists():
                d = np.asarray(np.load(stack, mmap_mode="r")[FACE[role]], np.float32)
            else:                                     # fallback: compute it
                face = anchor(cid, role)[-1 if role == "A" else 0]
                t = time.time()
                d = depth.disparity(face[None], device=dev)[0]
                log.info("depth (uncached) %-28s %.1fs", f"{cid}/{role}",
                         time.time() - t)
                np.save(npy, d)
            disps[key] = d.astype(np.float32)
            return disps[key]

        # -- sample plan ------------------------------------------------------
        def pair() -> dict:
            a, b = rng.sample(pool, 2)
            return {"pair_id": f"{a}__{b}", "from": a, "to": b}

        pairs = [pair() for _ in range(24)]

        manifest: list[dict] = []
        vid_dir, strip_dir = run_dir / "videos", run_dir / "filmstrips"
        strip_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()

        def render_one(entry: dict, op, tag: str, n_mid: int) -> None:
            s9 = anchor(entry["from"], "A")
            e9 = anchor(entry["to"], "B")
            da = anchor_disp(entry["from"], "A")
            db = anchor_disp(entry["to"], "B")
            cov: list[dict] = []
            t = time.time()
            clip = ops3d.render_transition(renderer, op, s9, e9, da, db, n_mid,
                                           coverage_out=cov)
            render_s = time.time() - t
            n_tot = clip.shape[0]
            assert n_tot == K + n_mid + K
            assert (n_tot - 1) % 8 == 0

            d0, d1, r0, r1 = ops3d.seam_error(clip, K, n_mid)
            za = depth.to_view_depth(da, op.depth_near, op.depth_far, op.depth_gamma)
            pi = metrics.parallax_index(clip[K:K + min(6, n_mid)], za)

            # Endpoint fidelity, in-array: the anchors must be the source frames.
            fid_a = int(np.abs(clip[:K].astype(np.int16) - s9.astype(np.int16)).max())
            fid_b = int(np.abs(clip[-K:].astype(np.int16) - e9.astype(np.int16)).max())

            stem = (f"{tag}__{op.recipe}__{op.path}__n{n_tot}"
                    f"__{entry['pair_id']}__{op.seed % 10**6:06d}")
            stem = stem[:150]
            while stem in seen:
                stem += "x"
            seen.add(stem)
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            PIL.Image.fromarray(
                videoio.filmstrip(clip, strip_indices(K, n_mid))
            ).resize((len(strip_indices(K, n_mid)) * 108, 144),
                     PIL.Image.LANCZOS).save(strip_dir / f"{stem}.jpg", quality=86)

            # Codec round-trip: what a decoder actually sees at the anchors.
            dec = videoio.read_clip(vid_dir / f"{stem}.mp4")
            rt = int(max(
                np.abs(dec[:K].astype(np.int16) - s9.astype(np.int16)).max(),
                np.abs(dec[-K:].astype(np.int16) - e9.astype(np.int16)).max()))

            manifest.append({
                "stem": stem, "tag": tag, "block": tag.split("_")[0],
                "pair_id": entry["pair_id"], "from": entry["from"], "to": entry["to"],
                "label_from": entries[entry["from"]]["label"],
                "label_to": entries[entry["to"]]["label"],
                "recipe": op.recipe, "coupling": op.coupling, "family": op.path,
                "dissolve": op.dissolve, "blend": op.blend, "easing": op.easing,
                "n_middle": n_mid, "n_frames": int(n_tot),
                "vae_legal": bool((n_tot - 1) % 8 == 0),
                "describe": op.describe(), "params": dataclass_dict(op),
                "seam_mae_in": round(d0, 3), "seam_mae_out": round(d1, 3),
                "seam_ratio_in": round(r0, 3), "seam_ratio_out": round(r1, 3),
                "endpoint_maxabs": max(fid_a, fid_b),
                "codec_roundtrip_maxabs": rt,
                "parallax": pi,
                "uncovered_mean": round(float(np.mean([c["uncovered"] for c in cov])), 4),
                "uncovered_max": round(float(np.max([c["uncovered"] for c in cov])), 4),
                "hole_radius_max": round(float(np.max([c["hole_radius"] for c in cov])), 1),
                "weak_radius_max": round(float(np.max([c["weak_radius"] for c in cov])), 1),
                "render_s": round(render_s, 2),
            })
            log.info("%-70s n=%2d seam=(%.2f,%.2f) PI=%.2f rho=%+.2f hole=%.0fpx %.1fs",
                     stem[:70], n_tot, r0, r1, pi["pi"], pi["rho"],
                     manifest[-1]["hole_radius_max"], render_s)

        # 1. camera families — bare move, one pair, everything else matched
        show = pairs[0]
        for fam in sorted(cameras.PATHS):
            render_one(show, make_op(rng, "bare_move", path=fam), "family", 15)

        # 2. AXIS A: same content x different operator. All 11 recipes on each of
        #    two fixed pairs. This is the signal a real corpus cannot supply.
        for i in range(smp["n_axis_operator_pairs"]):
            for name in RECIPES:
                op = make_op(rng, name)
                render_one(pairs[i], op, f"axisop{i}", rng.choice(LENGTHS))

        # 3. AXIS B: same operator x different content. One operator instance,
        #    byte-identical parameters, applied across several endpoint pairs.
        shared = rng.sample(CONTENT_COUPLED, smp["n_axis_content_ops"] - 1) + ["bare_move"]
        for j, name in enumerate(shared):
            op = make_op(rng, name)
            nm = rng.choice(LENGTHS)
            for entry in pairs[2:2 + smp["n_axis_content_pairs"]]:
                render_one(entry, op, f"axiscontent{j}", nm)

        # 4. Length sweep — same pair, same operator, the four legal totals.
        lp = pairs[10]
        for name in rng.sample(CONTENT_COUPLED, smp["n_length_ops"]):
            op = make_op(rng, name)
            for nm in LENGTHS:
                render_one(lp, op, "length", nm)

        # 5. Amplitude sweep — the honest disocclusion probe: how far can the
        #    camera travel before push-pull is inventing more than it repairs?
        #    Deliberately on `bare_move`: a dissolve punches its own holes in both
        #    layers, so it would confound geometry-driven disocclusion with the
        #    effect's intended erosion. This block isolates the geometry.
        ap = pairs[11]
        for pth in smp["amp_sweep_paths"]:
            for amp in smp["amp_sweep"]:
                render_one(ap, make_op(rng, "bare_move", path=pth, amplitude=amp,
                                       sign=1), "amp", 15)

        # 6. Diversity sample — random recipe x random pair x random length,
        #    with the optional effects re-enabled so the sample is not sterile.
        names = list(RECIPES)
        for _ in range(smp["n_diverse"]):
            name = rng.choice(names)
            op = make_op(rng, name, amplitude=rng.uniform(0.6, 1.55))
            if rng.random() < 0.35:
                op.handheld = rng.uniform(0.25, 0.9)
            if rng.random() < 0.30:
                op.motion_blur = rng.choice([3, 4])
            if rng.random() < 0.25 and op.fog == 0:
                op.fog = rng.uniform(0.6, 1.8)
            op.blend_easing = rng.choice(sorted(ops3d.EASINGS))
            render_one(pair(), op, "diverse", rng.choice(LENGTHS))

        json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=1)

        ratios = [max(m["seam_ratio_in"], m["seam_ratio_out"]) for m in manifest]
        log.info("=" * 78)
        log.info("rendered %d clips | seam ratio median %.2f p90 %.2f max %.2f | "
                 "over 2.0: %d/%d", len(manifest), float(np.median(ratios)),
                 float(np.percentile(ratios, 90)), max(ratios),
                 sum(r > 2.0 for r in ratios), len(ratios))
        log.info("endpoint fidelity (in-array) max abs diff = %d | codec round-trip "
                 "max abs diff = %d",
                 max(m["endpoint_maxabs"] for m in manifest),
                 max(m["codec_roundtrip_maxabs"] for m in manifest))
        log.info("parallax index median %.2f | depth-flow rho median %+.2f",
                 float(np.median([m["parallax"]["pi"] for m in manifest])),
                 float(np.median([m["parallax"]["rho"] for m in manifest])))
        log.info("disocclusion: uncovered mean %.1f%% | hole radius median %.0fpx "
                 "p90 %.0fpx max %.0fpx",
                 100 * float(np.mean([m["uncovered_mean"] for m in manifest])),
                 float(np.median([m["hole_radius_max"] for m in manifest])),
                 float(np.percentile([m["hole_radius_max"] for m in manifest], 90)),
                 max(m["hole_radius_max"] for m in manifest))
        print(f"[done] {run_id} → {run_dir}")


if __name__ == "__main__":
    main()
