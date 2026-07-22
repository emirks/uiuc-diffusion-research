"""exp_075 — procedural transition operator engine: bank validation + sample render.

Takes the 9-frame endpoint clips (`*_start9.mp4` / `*_end9.mp4`) and synthesises
121-frame transitions between them with procedurally sampled operators, so that
the conditioning blocks of every output reproduce the given endpoints exactly.
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

from engine import operators, shaders, streams, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

log = logging.getLogger("exp075")


def discover_pairs(cond_dir: pathlib.Path) -> dict[str, dict]:
    """Map clip id -> {'start9': path, 'end9': path} for ids that have both."""
    found: dict[str, dict] = {}
    for p in sorted(cond_dir.glob("*_start9.mp4")):
        cid = p.name[: -len("_start9.mp4")]
        end = cond_dir / f"{cid}_end9.mp4"
        if end.exists():
            found[cid] = {"start9": p, "end9": end}
    return found


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
        T, K = inf["num_frames"], inf["anchor_frames"]
        rng = random.Random(cfg["runtime"]["seed"])

        # -- 1. shader bank -------------------------------------------------
        bank_dir = pathlib.Path(cfg["model"]["shader_bank"])
        bank = shaders.load_bank(bank_dir)
        log.info("parsed %d shaders from %s", len(bank), bank_dir)

        runner = GLRunner(W, H)
        log.info("GL context: %s", runner.renderer_name())

        t0 = time.time()
        bank, report = operators.validate_bank(runner, bank,
                                               tol=smp["endpoint_tol"])
        json.dump(report, open(run_dir / "bank_validation.json", "w"), indent=1)
        by_status: dict[str, int] = {}
        for r in report:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        log.info("bank validation (%.1fs): %s", time.time() - t0, by_status)
        log.info("usable shaders: %d | est. distinguishable operators: %.2e",
                 len(bank), operators.bank_capacity(bank))

        # -- 2. endpoint pairs ----------------------------------------------
        cond_dir = REPO_ROOT / cfg["inputs"]["cond_dir"]
        clips = discover_pairs(cond_dir)
        log.info("found %d endpoint pairs in %s", len(clips), cond_dir.name)
        ids = sorted(clips)

        cache: dict[pathlib.Path, np.ndarray] = {}

        def load(path: pathlib.Path) -> np.ndarray:
            if path not in cache:
                cache[path] = videoio.read_clip(path)
            return cache[path]

        plan: list[dict] = []
        same = rng.sample(ids, min(smp["n_pairs_same"], len(ids)))
        for cid in same:
            plan.append({"kind": "same", "pair_id": cid,
                         "from": cid, "to": cid})
        for _ in range(smp["n_pairs_cross"]):
            a, b = rng.sample(ids, 2)
            plan.append({"kind": "cross", "pair_id": f"{a}__{b}",
                         "from": a, "to": b})

        # -- 3. render ------------------------------------------------------
        easings = sorted(streams.EASINGS)
        manifest: list[dict] = []
        vid_dir = run_dir / "videos"
        strip_dir = run_dir / "filmstrips"
        strip_idx = [0, 8, 20, 35, 50, 60, 70, 85, 100, 112, 120]

        def render_one(entry: dict, op: operators.Operator, tag: str) -> None:
            s9 = load(clips[entry["from"]]["start9"])[:K]
            e9 = load(clips[entry["to"]]["end9"])[-K:]
            t = time.time()
            clip = operators.render_sample(runner, bank, op, s9, e9, total=T)
            d0, d1 = operators.endpoint_fidelity(clip, s9, e9)
            stem = f"{tag}__{entry['pair_id']}__{op.shader}__{op.op_id[-6:]}"
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            strip = videoio.filmstrip(clip, strip_idx)
            PIL.Image.fromarray(strip).save(strip_dir / f"{stem}.jpg", quality=88)
            manifest.append({
                "stem": stem, "tag": tag, "pair_kind": entry["kind"],
                "pair_id": entry["pair_id"], "from": entry["from"],
                "to": entry["to"], "shader": op.shader, "op_id": op.op_id,
                "easing": op.easing, "flip": op.flip, "swap": op.swap,
                "extension": op.extension, "aux_kind": op.aux_kind,
                "params": op.params, "describe": op.describe(),
                "endpoint_mae_start": round(d0, 3),
                "endpoint_mae_end": round(d1, 3),
                "render_s": round(time.time() - t, 2),
            })
            log.info("%-58s %-16s ep=(%.2f,%.2f) %.1fs",
                     stem[:58], op.easing, d0, d1, time.time() - t)

        strip_dir.mkdir(parents=True, exist_ok=True)

        # The endpoint gate has to run per OPERATOR, not per shader: the p=0/p=1
        # identities are parameter-dependent, so a shader that is clean at its
        # defaults can still violate them at a sampled parameter setting.
        gate: dict = {}

        def draw(entry: dict, **kw) -> operators.Operator:
            s9 = load(clips[entry["from"]]["start9"])[:K]
            e9 = load(clips[entry["to"]]["end9"])[-K:]
            return operators.sample_valid_operator(
                runner, bank, rng, s9[-1], e9[0], tol=smp["endpoint_tol_operator"],
                stats=gate, easings=easings, p_flip=smp["p_flip"],
                p_swap=smp["p_swap"], **kw)

        # 3a. operator diversity across many endpoint pairs
        for entry in plan:
            for _ in range(smp["n_operators_per_pair"]):
                render_one(entry, draw(entry, extensions=tuple(smp["extensions"])),
                           "diverse")

        # 3b. counterfactual block: ONE endpoint pair, many operators
        cf = plan[0]
        for _ in range(smp["n_counterfactual"]):
            render_one(cf, draw(cf, extensions=("boomerang",)), "counterfactual")

        # 3c. the complementary axis: ONE operator, many endpoint pairs.
        # Together with 3b this is the operator ⊥ content factorisation demo —
        # 3b holds content fixed and varies the operator, 3c does the reverse.
        for k in range(smp["n_shared_operators"]):
            op = draw(plan[0], extensions=("boomerang",))
            for entry in plan[: smp["n_pairs_per_shared_operator"]]:
                render_one(entry, op, f"sharedop{k}")

        log.info("operator gate: %d sampled, %d rejected (%.1f%%) — %s",
                 gate.get("tried", 0), gate.get("rejected", 0),
                 100.0 * gate.get("rejected", 0) / max(gate.get("tried", 1), 1),
                 sorted(gate.get("rejected_shaders", {}).items(),
                        key=lambda kv: -kv[1])[:8])
        json.dump(gate, open(run_dir / "operator_gate.json", "w"), indent=1)

        # 3d. extension-policy ablation: same pair, same operator, 3 policies
        abl_op = operators.sample_operator(bank, rng, extensions=("boomerang",),
                                           easings=["smoothstep"], p_flip=0.0,
                                           p_swap=0.0)
        for pol in smp["extensions"]:
            op = operators.Operator(**{**abl_op.__dict__, "extension": pol})
            render_one(plan[0], op, f"ext_{pol}")

        # -- 4. reference: the real transition for the same endpoints --------
        real_root = REPO_ROOT / cfg["inputs"]["real_dir"]
        for entry in plan:
            if entry["kind"] != "same":
                continue
            cid = entry["from"]
            cls = cid.rsplit("_", 1)[0]
            src = real_root / cls / f"{cid}.mp4"
            if src.exists():
                real = videoio.read_clip(src)
                stem = f"REAL__{cid}"
                videoio.write_clip(vid_dir / f"{stem}.mp4", real, fps=inf["fps"])
                PIL.Image.fromarray(videoio.filmstrip(real, strip_idx)).save(
                    strip_dir / f"{stem}.jpg", quality=88)
                manifest.append({"stem": stem, "tag": "real", "pair_kind": "same",
                                 "pair_id": cid, "from": cid, "to": cid,
                                 "shader": "(human-made ground truth)",
                                 "describe": "real transition from the 49-clip corpus",
                                 "op_id": "real", "easing": "-", "flip": "none",
                                 "swap": False, "extension": "-", "aux_kind": None,
                                 "params": {}, "endpoint_mae_start": 0.0,
                                 "endpoint_mae_end": 0.0, "render_s": 0.0})

        json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=1)

        maes = [m["endpoint_mae_start"] for m in manifest if m["tag"] != "real"] + \
               [m["endpoint_mae_end"] for m in manifest if m["tag"] != "real"]
        log.info("rendered %d clips | endpoint MAE max %.3f mean %.3f",
                 len(manifest), max(maes), float(np.mean(maes)))
        print(f"[done] {run_id} → {run_dir}")


if __name__ == "__main__":
    main()
