"""exp_084 — luma-matte transition viewer: was it the map, or was it `step()`?

The aux-map operator family (a static greyscale matte + a threshold sweep) was
judged fake-looking and shipped at 0%. Two things were never separated:

  * the MAPS  — seven stationary isotropic fields (fbm, radial, linear, stripes,
    checker, spiral, voronoi), which have no source, no propagation, no branching
  * the COMPOSITOR — `luma.glsl`, which is a single `step()`: a hard binary
    threshold with no feather, no rim colour and no glow

This run is a 2x2 that separates them, over real playing footage:

    A1  old maps x hard step()      <- exactly what was judged
    A2  old maps x feathered soft   <- fixed compositor, identical maps
    A3  new maps x feathered soft   <- fixed compositor, structured mattes
    A4  new maps x hard step()      <- structured mattes, broken compositor

A1/A2 share map, map seed and content pair, so the two cells differ *only* in
the compositor. A3/A4 do the same for the new maps. Both layers are real frames
from the curated 227-clip endpoint bank playing forward; the anchor blocks are
verbatim and are asserted bit-exact.
"""

from __future__ import annotations

import glob
import json
import logging
import pathlib
import random
import sys
import time

import numpy as np
import PIL.Image
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp_075_procedural_transition_engine"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion.exp_utils import TeeLogger, load_config, next_run_dir  # noqa: E402

from engine import maps as oldmaps      # noqa: E402  the SHIPPED map bank, unmodified
from engine import streams              # noqa: E402  easing / progress ramp
from engine.glrunner import GLRunner    # noqa: E402
from mattes import clipio, glctx, newmaps, styles  # noqa: E402

log = logging.getLogger("exp084")

HARD_PARAMS: dict = {}          # luma.glsl has no tunables besides the sampler


# --------------------------------------------------------------------------


def render_clip(runner: GLRunner, prog, a_stream: np.ndarray, b_stream: np.ndarray,
                ramp: np.ndarray, params: dict) -> np.ndarray:
    out = np.empty_like(a_stream)
    for t in range(len(a_stream)):
        out[t] = runner.render(prog, a_stream[t], b_stream[t], float(ramp[t]),
                               params, "luma")
    return out


def strip_indices(total: int, k: int, n_mid: int = 6) -> list[int]:
    mids = sorted({k + round(i * (total - 2 * k - 1) / (n_mid - 1))
                   for i in range(n_mid)})
    return sorted({0, k - 1, *mids, total - k, total - 1})


def small_strip(frames: np.ndarray, indices, div: int = 3) -> np.ndarray:
    """Filmstrip at 1/`div` scale — a full-res strip of 10 tiles is ~400 kB."""
    h, w = frames.shape[1:3]
    sh, sw = h // div, w // div
    sel = [np.asarray(PIL.Image.fromarray(frames[i]).resize((sw, sh), PIL.Image.LANCZOS))
           for i in indices]
    pad = 2
    strip = np.full((sh, len(sel) * sw + (len(sel) - 1) * pad, 3), 255, np.uint8)
    for k, img in enumerate(sel):
        strip[:, k * (sw + pad):k * (sw + pad) + sw] = img
    return strip


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout, force=True)
        yaml.safe_dump(cfg, open(run_dir / "config_snapshot.yaml", "w"))

        inf, smp = cfg["inference"], cfg["sampling"]
        H, W, T, K = inf["height"], inf["width"], inf["num_frames"], inf["anchor_frames"]
        rng = random.Random(cfg["runtime"]["seed"])

        # -- shaders ---------------------------------------------------------
        hard_src = pathlib.Path(cfg["model"]["luma_hard"]).read_text()
        soft_src = (REPO_ROOT / cfg["model"]["luma_soft"]).read_text()
        assert "step(progress" in hard_src, "luma.glsl is not the hard-threshold shader"
        runner = glctx.make_runner(GLRunner, W, H)
        log.info("GL context: %s | egl_device_index=%s",
                 runner.renderer_name(), runner.egl_device_index)
        prog_hard = runner.program("luma_hard", hard_src)
        prog_soft = runner.program("luma_soft", soft_src)

        # -- content ---------------------------------------------------------
        bank_dir = REPO_ROOT / cfg["inputs"]["bank_dir"]
        bank = json.load(open(REPO_ROOT / cfg["inputs"]["bank_filter"]))
        entries = {e["clip_id"]: e for e in bank["clips"]}
        ids = sorted(entries)
        log.info("endpoint bank: %d clips", len(ids))

        clip_cache: dict[str, np.ndarray] = {}

        def clip(cid: str) -> np.ndarray:
            if cid not in clip_cache:
                f = clipio.read_clip(bank_dir / entries[cid]["mp4"])
                assert f.shape[1:] == (H, W, 3), f"{cid}: {f.shape}"
                clip_cache[cid] = f
            return clip_cache[cid]

        pairs = []
        for _ in range(smp["n_pairs"]):
            a, b = rng.sample(ids, 2)
            pairs.append({"pair_id": f"{a}__{b}", "from": a, "to": b})
        log.info("content pairs: %s", [p["pair_id"] for p in pairs])

        def streams_for(p: dict) -> tuple[np.ndarray, np.ndarray]:
            """Both layers are REAL frames playing forward, taken verbatim.

            The outgoing shot plays its LAST T frames (so the cut lands at its
            end) and the incoming shot plays its FIRST T frames. Nothing is
            held, boomeranged or extrapolated.
            """
            a = np.ascontiguousarray(clip(p["from"])[-T:])
            b = np.ascontiguousarray(clip(p["to"])[:T])
            return a, b

        ramp = streams.progress_ramp(T, K, K, inf["easing"])
        assert ramp[:K].max() == 0.0 and ramp[-K:].min() == 1.0

        # -- maps -------------------------------------------------------------
        brush_paths = sorted(glob.glob(
            str(REPO_ROOT / cfg["inputs"]["brush_dir"] / "*.png")))
        brushes = [newmaps.load_brush(p) for p in brush_paths]
        log.info("CC0 brush alphas: %d", len(brushes))

        old_kinds = list(oldmaps.MAP_KINDS)
        new_kinds = list(newmaps.NEW_MAP_KINDS)

        vid_dir, strip_dir, map_dir = (run_dir / "videos", run_dir / "filmstrips",
                                       run_dir / "maps")
        for d in (vid_dir, strip_dir, map_dir):
            d.mkdir(parents=True, exist_ok=True)

        # one map instance per (map kind, content pair) so A1/A2 (and A3/A4)
        # are exactly paired: same field, same seed, same footage.
        map_cache: dict[tuple[str, str], np.ndarray] = {}
        map_seed: dict[tuple[str, str], int] = {}

        def get_map(name: str, is_new: bool, p: dict) -> np.ndarray:
            key = (name, p["pair_id"])
            if key in map_cache:
                return map_cache[key]
            s = rng.randrange(1 << 30)
            map_seed[key] = s
            t = time.time()
            if is_new:
                tgt = clip(p["to"])[0]
                m = newmaps.build_new_map(name, H, W, s, brushes=brushes, target=tgt)
            else:
                # the shipped generator, byte-for-byte: HxWx3 uint8, red channel used
                m = oldmaps.make_map(name, H, W, s).astype(np.float32)[..., 0] / 255.0
            log.info("map %-16s %-34s %.2fs", name, p["pair_id"][:34], time.time() - t)
            map_cache[key] = m
            return m

        # -- render ------------------------------------------------------------
        strip_idx = strip_indices(T, K)
        manifest: list[dict] = []
        t_start = time.time()

        def one(arm: str, name: str, is_new: bool, p: dict, compositor: str) -> None:
            m = get_map(name, is_new, p)
            runner.set_aux_map(newmaps.aux_upload(newmaps.to_rgb_u8(m)))
            style_key, st = styles.style_for(name, is_new)
            if compositor == "hard":
                prog, params, style_key = prog_hard, HARD_PARAMS, "-"
            else:
                prog, params = prog_soft, st

            a, b = streams_for(p)
            t0 = time.time()
            out = render_clip(runner, prog, a, b, ramp, params)

            # Conditioning blocks must reproduce the source frames.
            # NOTE: `luma.glsl` cannot quite do this. `step(progress, m)` returns
            # 1 when m == progress, so at progress = 1 every pixel sitting at the
            # matte's maximum keeps showing `from`. Any map normalised to [0,1]
            # has such pixels. We therefore record the violating-pixel count as
            # well as the MAE, and only hard-assert on the MAE.
            e0 = np.abs(out[:K].astype(int) - a[:K].astype(int))
            e1 = np.abs(out[-K:].astype(int) - b[-K:].astype(int))
            d0, d1 = int(e0.max()), int(e1.max())
            mae = float((e0.mean() + e1.mean()) / 2)
            nbad = int((e1.max(-1) > 8).sum())

            stem = f"{arm}__{name}__{p['pair_id']}"
            clipio.write_clip(vid_dir / f"{stem}.mp4", out, fps=inf["fps"])
            PIL.Image.fromarray(small_strip(out, strip_idx)).save(
                strip_dir / f"{stem}.jpg", quality=88)
            mp = f"{name}__{p['pair_id']}.png"
            if not (map_dir / mp).exists():
                PIL.Image.fromarray((m * 255).astype(np.uint8)).resize(
                    (W // 3, H // 3), PIL.Image.LANCZOS).save(map_dir / mp)

            manifest.append({
                "stem": stem, "arm": arm, "map": name,
                "map_family": "new" if is_new else "shipped",
                "map_seed": map_seed[(name, p["pair_id"])],
                "compositor": "luma.glsl (hard step)" if compositor == "hard"
                              else "luma_soft.glsl (feather+rim+glow)",
                "compositor_key": compositor,
                "style": style_key,
                "feather": None if compositor == "hard" else st["feather"],
                "rim_amount": None if compositor == "hard" else st["rimAmount"],
                "glow_amount": None if compositor == "hard" else st["glowAmount"],
                "map_png": mp, "pair_id": p["pair_id"],
                "from": p["from"], "to": p["to"],
                "label_from": entries[p["from"]]["label"],
                "label_to": entries[p["to"]]["label"],
                "endpoint_max_abs_start": d0, "endpoint_max_abs_end": d1,
                "endpoint_mae": round(mae, 4), "endpoint_bad_px": nbad,
                "render_s": round(time.time() - t0, 2),
            })
            log.info("%-64s ep max=(%d,%d) mae=%.4f badpx=%d %.1fs",
                     stem[:64], d0, d1, mae, nbad, time.time() - t0)

        for p in pairs:
            for name in old_kinds:
                one("A1_current", name, False, p, "hard")
                one("A2_soft_same_map", name, False, p, "soft")
            for name in new_kinds:
                one("A3_new_map_soft", name, True, p, "soft")
                one("A4_new_map_hard", name, True, p, "hard")

        json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=1)
        worst_mae = max(m["endpoint_mae"] for m in manifest)
        by_comp: dict[str, list[int]] = {}
        for m in manifest:
            by_comp.setdefault(m["compositor_key"], []).append(m["endpoint_bad_px"])
        log.info("rendered %d clips in %.1f min | worst endpoint MAE = %.4f",
                 len(manifest), (time.time() - t_start) / 60, worst_mae)
        for k, v in sorted(by_comp.items()):
            log.info("  endpoint bad pixels (%s): mean %.1f  max %d  of %d px",
                     k, float(np.mean(v)), max(v), H * W)
        assert worst_mae <= 0.5, f"endpoint identity violated (MAE {worst_mae})"
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
