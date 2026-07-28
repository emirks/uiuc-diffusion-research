"""exp_077 TASK 0e (CPU render + gate stats) — operator-bank audit.

Two products, both on ONE fixed endpoint pair so the operator is the only thing that varies:

(A) Broad natural sample: draw `audit_n_operators` operators uniformly from the gated bank,
    run the per-operator endpoint gate (gate-2) on the fixed pair, render every ACCEPTED
    operator, and record the natural presence of the aux-map (medium-bearing) families.
    The rendered clips feed embed_cluster.py, which reports effective-K.

(B) Per-family gate-2 acceptance: for each aux shader (luma / displacement) x each of the 7
    aux-map families, force `audit_aux_per_family` draws and record the gate-2 acceptance
    rate. This is the flag for whether the medium-bearing families (particle/smoke/ink-like)
    are being disproportionately rejected — they are the most valuable operators.

Rendering is CPU (EGL/llvmpipe). Embedding + clustering (DINOv2, GPU) is embed_cluster.py.
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion.exp_utils import TeeLogger, load_config, next_run_dir  # noqa: E402

from engine import maps, operators, shaders, streams, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
log = logging.getLogger("exp077.audit")


def discover_pairs(cond_dir: pathlib.Path) -> dict:
    found = {}
    for p in sorted(cond_dir.glob("*_start9.mp4")):
        cid = p.name[: -len("_start9.mp4")]
        end = cond_dir / f"{cid}_end9.mp4"
        if end.exists():
            found[cid] = {"start9": p, "end9": end}
    return found


def make_aux_operator(bank, rng, shader_name, family, *, easings, p_flip, p_swap, extension):
    """Build an operator forced onto a given aux shader + aux-map family (for gate stats)."""
    sh = bank[shader_name]
    params = shaders.sample_params(sh, rng)
    easing = rng.choice(easings)
    flip = rng.choice(operators.FLIPS[1:]) if rng.random() < p_flip else "none"
    swap = rng.random() < p_swap
    aux_seed = rng.randrange(1 << 30)
    return operators.Operator(op_id=f"{shader_name}:{family}:{aux_seed}", shader=shader_name,
                              params=params, easing=easing, flip=flip, swap=swap,
                              extension=extension, aux_kind=family, aux_seed=aux_seed)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    out_dir = REPO_ROOT / (cfg["outputs"]["dir"] + "_audit")
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout, force=True)
        yaml.safe_dump(cfg, open(run_dir / "config_snapshot.yaml", "w"))

        inf, smp = cfg["inference"], cfg["sampling"]
        H, W = inf["height"], inf["width"]
        T, K = inf["num_frames"], inf["anchor_frames"]
        rng = random.Random(cfg["runtime"]["seed"] + 1)
        easings = sorted(streams.EASINGS)

        bank = shaders.load_bank(pathlib.Path(cfg["model"]["shader_bank"]))
        runner = GLRunner(W, H)
        log.info("GL: %s | parsed %d shaders", runner.renderer_name(), len(bank))
        bank, _ = operators.validate_bank(runner, bank, tol=smp["endpoint_tol"])
        aux_present = [s for s in shaders.AUX_SAMPLER_SHADERS if s in bank]
        log.info("usable shaders after gate-1: %d | aux shaders present: %s",
                 len(bank), aux_present)

        cond_dir = REPO_ROOT / cfg["inputs"]["cond_dir"]
        clips = discover_pairs(cond_dir)
        ids = sorted(clips)
        fixed = ids[smp["audit_fixed_pair_index"]]
        s9 = videoio.read_clip(clips[fixed]["start9"])[:K]
        e9 = videoio.read_clip(clips[fixed]["end9"])[-K:]
        fa, fb = s9[-1], e9[0]
        log.info("fixed endpoint pair: %s", fixed)

        vid_dir = run_dir / "videos"
        strip_dir = run_dir / "filmstrips"
        vid_dir.mkdir(parents=True, exist_ok=True)
        strip_dir.mkdir(parents=True, exist_ok=True)
        strip_idx = [i for i in (0, 8, 30, 50, 60, 70, 90, 112, 120) if i < T]
        tol = smp["endpoint_tol_operator"]

        # ---- (A) broad natural sample ------------------------------------------------
        rendered, gate_rows, family_natural = [], [], {}
        n = smp["audit_n_operators"]
        log.info("(A) sampling %d operators, gating on the fixed pair ...", n)
        for i in range(n):
            op = operators.sample_operator(bank, rng, extensions=(smp["extension"],),
                                           easings=easings, p_flip=smp["p_flip"],
                                           p_swap=smp["p_swap"])
            d0, d1 = operators.check_operator(runner, bank, op, fa, fb)
            passed = max(d0, d1) <= tol
            row = {"i": i, "shader": op.shader, "aux_kind": op.aux_kind,
                   "easing": op.easing, "flip": op.flip, "swap": op.swap,
                   "mae": [round(d0, 4), round(d1, 4)], "passed": passed}
            gate_rows.append(row)
            if op.aux_kind:
                fk = family_natural.setdefault(op.aux_kind, {"tried": 0, "passed": 0})
                fk["tried"] += 1
                fk["passed"] += int(passed)
            if passed:
                stem = f"op{len(rendered):03d}_{op.shader}"
                clip = operators.render_sample(runner, bank, op, s9, e9, total=T)
                videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
                PIL.Image.fromarray(videoio.filmstrip(clip, strip_idx)).save(
                    strip_dir / f"{stem}.jpg", quality=85)
                rendered.append({"stem": stem, "shader": op.shader, "aux_kind": op.aux_kind,
                                 "easing": op.easing, "op_id": op.op_id})
            if (i + 1) % 25 == 0:
                log.info("  gated %d/%d  rendered=%d", i + 1, n, len(rendered))

        n_pass = sum(r["passed"] for r in gate_rows)
        distinct_shaders = len({r["shader"] for r in rendered})
        log.info("(A) sampled=%d passed=%d (%.1f%%) rendered=%d distinct_shaders=%d",
                 n, n_pass, 100.0 * n_pass / max(n, 1), len(rendered), distinct_shaders)

        # ---- (B) forced per-family gate-2 acceptance ---------------------------------
        per_family = {}
        m = smp["audit_aux_per_family"]
        log.info("(B) forced aux draws: %d per (shader,family) over %d families ...",
                 m, len(maps.MAP_KINDS))
        for shader_name in aux_present:
            for family in maps.MAP_KINDS:
                tried = passed = 0
                t = time.time()
                for _ in range(m):
                    op = make_aux_operator(bank, rng, shader_name, family, easings=easings,
                                           p_flip=smp["p_flip"], p_swap=smp["p_swap"],
                                           extension=smp["extension"])
                    d0, d1 = operators.check_operator(runner, bank, op, fa, fb)
                    tried += 1
                    passed += int(max(d0, d1) <= tol)
                key = f"{shader_name}/{family}"
                per_family[key] = {"shader": shader_name, "family": family,
                                   "tried": tried, "passed": passed,
                                   "acceptance": round(passed / max(tried, 1), 4)}
                log.info("  %-22s %2d/%2d accepted (%.0f%%)  %.1fs",
                         key, passed, tried, 100.0 * passed / max(tried, 1), time.time() - t)

        # aggregate per family across both aux shaders
        family_agg = {}
        for v in per_family.values():
            a = family_agg.setdefault(v["family"], {"tried": 0, "passed": 0})
            a["tried"] += v["tried"]
            a["passed"] += v["passed"]
        for f, a in family_agg.items():
            a["acceptance"] = round(a["passed"] / max(a["tried"], 1), 4)

        report = {
            "fixed_pair": fixed, "num_frames": T,
            "gate_tol": tol,
            "broad_sample": {"sampled": n, "passed": n_pass,
                             "acceptance": round(n_pass / max(n, 1), 4),
                             "rendered": len(rendered),
                             "distinct_shaders_rendered": distinct_shaders,
                             "aux_families_natural_presence": family_natural},
            "aux_present": aux_present,
            "per_family_gate": per_family,
            "per_family_aggregate": family_agg,
        }
        json.dump(report, open(run_dir / "audit_report.json", "w"), indent=2)
        json.dump(gate_rows, open(run_dir / "broad_gate_rows.json", "w"), indent=2)
        json.dump(rendered, open(run_dir / "rendered_index.json", "w"), indent=2)
        log.info("per-family aggregate acceptance: %s",
                 {f: a["acceptance"] for f, a in sorted(family_agg.items())})
        log.info("DONE %s -> %s", run_id, run_dir)
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
