"""exp_052 phase 1 — the harness's own exam, zero generated videos needed.

Computes morph profiles, tracklets, and effect signatures for every real clip
in data/processed/transitions (higgsfield excluded), builds per-clip lerp
controls, then runs:
  1. style-discrimination: leave-one-out 1-NN retrieval per metric
     (morph-DTW / effect appearance / motion fidelity) — a metric that cannot
     tell shadow_smoke from earth_wave on ground truth cannot evaluate transfer;
  2. lerp floor: synthesized crossfades must sit near depth=0, separated from
     the real clips' transformation-depth distribution.

Outputs distance matrices, confusion tables, morph-profile figures,
results.json, report.md.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.exp_utils import TeeLogger, load_config, next_run_dir  # noqa: E402
from diffusion.transition_eval.appearance import set_similarity  # noqa: E402
from diffusion.transition_eval.controls import make_lerp  # noqa: E402
from diffusion.transition_eval.features import DinoExtractor  # noqa: E402
from diffusion.transition_eval.manifest import scan_references  # noqa: E402
from diffusion.transition_eval.morph import profile_distance  # noqa: E402
from diffusion.transition_eval.motion import Tracker, motion_fidelity  # noqa: E402
from diffusion.transition_eval.pipeline import process_video, process_video_file  # noqa: E402
from diffusion.transition_eval.report import md_table, retrieval_eval  # noqa: E402
from diffusion.transition_eval.video_io import write_video  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def make_figures(run_dir, names, styles, bundles, lerp_bundles, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(set(styles))
    ncol = 3
    nrow = (len(uniq) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False)
    for k, style in enumerate(uniq):
        ax = axes[k // ncol][k % ncol]
        for n, s in zip(names, styles):
            if s != style:
                continue
            p = bundles[n].profile
            t = np.linspace(0, 1, len(p["a_hat"]))
            ax.plot(t, p["a_hat"], lw=1.2, alpha=0.8)
            ax.plot(t, p["b_hat"], lw=1.2, alpha=0.8, ls="--")
        ax.set_title(style, fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    for k in range(len(uniq), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Morph profiles per style — â(t) solid, b̂(t) dashed", fontsize=11)
    fig.tight_layout()
    fig.savefig(run_dir / "morph_profiles.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    real_d = [bundles[n].scalars["depth"] for n in names]
    lerp_d = [lerp_bundles[n].scalars["depth"] for n in names]
    bins = np.linspace(0, 1, 21)
    ax.hist(real_d, bins=bins, alpha=0.6, label="real clips")
    ax.hist(lerp_d, bins=bins, alpha=0.6, label="lerp controls")
    ax.set_xlabel("transformation depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "depth_hist.png", dpi=140)
    plt.close(fig)

    for metric, res in results.items():
        classes = sorted(res["per_class_recall"])
        M = np.array([[res["confusion"][c1][c2] for c2 in classes] for c1 in classes], dtype=float)
        M /= M.sum(axis=1, keepdims=True) + 1e-9
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(f"1-NN confusion — {metric} (acc={res['accuracy_1nn']:.2f})", fontsize=10)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if M[i, j] > 0.01:
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] > 0.5 else "black")
        fig.tight_layout()
        fig.savefig(run_dir / f"confusion_{metric}.png", dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(pathlib.Path(args.config))

    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / ("validation_smoke" if args.smoke else "validation")
    run_id, run_dir = next_run_dir(out_dir)
    cache_dir = REPO_ROOT / cfg["features"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    with TeeLogger(run_dir / "run.log"):
        (run_dir / "config_snapshot.yaml").write_text(pathlib.Path(args.config).read_text())
        refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                               exclude=tuple(cfg["data"]["exclude"]))
        if args.smoke:
            refs = {s: refs[s][:cfg["smoke"]["clips_per_style"]] for s in cfg["smoke"]["styles"]}
        n_clips = sum(len(v) for v in refs.values())
        print(f"[info] {run_id}: {len(refs)} styles, {n_clips} clips (smoke={args.smoke})")

        mk = cfg["morph"]
        device = cfg["runtime"]["device"]
        extractor = DinoExtractor(cfg["features"]["model"], device=device)
        tracker = Tracker(device=device, grid_size=cfg["motion"]["grid_size"],
                          max_side=cfg["motion"]["max_side"])

        names, styles = [], []
        bundles, lerp_bundles = {}, {}
        controls_dir = out_dir.parent / "controls"
        for style, vids in refs.items():
            for vid in vids:
                name = f"{style}/{vid.stem}"
                print(f"[info] processing {name}")
                b, frames = process_video_file(
                    vid, cache_dir, extractor, tracker,
                    short_side=cfg["features"]["short_side"],
                    n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
                lerp = make_lerp(frames[:mk["n_prefix"]], frames[-mk["n_suffix"]:], len(frames))
                lb = process_video(lerp, b.key + ":lerp", cache_dir, extractor, tracker,
                                   n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
                lerp_path = controls_dir / style / f"lerp_{vid.stem}.mp4"
                if not lerp_path.exists():
                    write_video(lerp, lerp_path, fps=b["fps"])
                names.append(name)
                styles.append(style)
                bundles[name] = b
                lerp_bundles[name] = lb
        extractor.free()
        tracker.free()

        # --- distance matrices over real clips --------------------------------
        n = len(names)
        D_morph = np.zeros((n, n))
        D_app = np.zeros((n, n))
        D_mot = np.zeros((n, n))
        core_feats = {nm: bundles[nm].feats[bundles[nm].core] for nm in names}
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = bundles[names[i]], bundles[names[j]]
                D_morph[i, j] = D_morph[j, i] = profile_distance(
                    bi.profile, bj.profile, n=mk["resample_points"])["dtw"]
                D_app[i, j] = D_app[j, i] = 1.0 - set_similarity(
                    core_feats[names[i]], core_feats[names[j]])
                mf = motion_fidelity(bi.tracks, bi.vis, bj.tracks, bj.vis,
                                     n_steps=cfg["motion"]["n_steps"])
                D_mot[i, j] = D_mot[j, i] = 1.0 - mf if np.isfinite(mf) else np.nan

        results = {m: retrieval_eval(D, styles) for m, D in
                   [("morph_dtw", D_morph), ("effect_appearance", D_app), ("motion_fidelity", D_mot)]}

        # --- lerp floor --------------------------------------------------------
        real_depth = np.array([bundles[nm].scalars["depth"] for nm in names])
        lerp_depth = np.array([lerp_bundles[nm].scalars["depth"] for nm in names])
        floor = {
            "real_depth_mean": float(real_depth.mean()), "real_depth_min": float(real_depth.min()),
            "lerp_depth_mean": float(lerp_depth.mean()), "lerp_depth_max": float(lerp_depth.max()),
            "lerp_below_02": float((lerp_depth < 0.2).mean()),
            "separation_ok": bool(lerp_depth.mean() + 2 * lerp_depth.std() < real_depth.mean()),
        }

        scalar_rows = [[nm, styles[k]] + [round(bundles[nm].scalars[s], 3) for s in
                       ("depth", "depart", "arrive", "hold", "core_frac")]
                       for k, nm in enumerate(names)]

        make_figures(run_dir, names, styles, bundles, lerp_bundles, results)
        np.savez_compressed(run_dir / "distance_matrices.npz", names=names, styles=styles,
                            morph=D_morph, appearance=D_app, motion=D_mot)
        (run_dir / "results.json").write_text(json.dumps(
            {"retrieval": results, "lerp_floor": floor,
             "scalars": {nm: bundles[nm].scalars for nm in names}}, indent=2))

        rep = ["# exp_052 harness validation — style-discrimination exam\n",
               f"{len(refs)} styles, {n} real clips; chance = {results['morph_dtw']['chance']:.2f}\n",
               md_table(["metric", "1-NN acc", "cohens_d", "within", "cross"],
                        [[m, r["accuracy_1nn"], r["separation_cohens_d"],
                          r["within_mean"], r["cross_mean"]] for m, r in results.items()]),
               "\n## Lerp floor\n", json.dumps(floor, indent=2),
               "\n## Per-clip scalars\n",
               md_table(["clip", "style", "depth", "depart", "arrive", "hold", "core_frac"],
                        scalar_rows)]
        (run_dir / "report.md").write_text("\n".join(rep))

        if cfg["wandb"]["enabled"] and not args.smoke:
            try:
                import wandb
                run = wandb.init(project=cfg["wandb"]["project"], name="exp052_validation",
                                 tags=["exp_052", "eval-harness", "validation"],
                                 config={"n_clips": n, "styles": sorted(set(styles))})
                for m, r in results.items():
                    run.summary[f"{m}/acc_1nn"] = r["accuracy_1nn"]
                    run.summary[f"{m}/cohens_d"] = r["separation_cohens_d"]
                run.summary.update({f"lerp/{k}": v for k, v in floor.items()})
                for img in run_dir.glob("*.png"):
                    run.log({img.stem: wandb.Image(str(img))})
                run.finish()
            except Exception as e:  # W&B must never sink the job
                print(f"[warn] wandb logging failed: {e}")

        print("[summary]")
        for m, r in results.items():
            print(f"  {m}: 1-NN acc {r['accuracy_1nn']:.2f} (chance {r['chance']:.2f}), "
                  f"d={r['separation_cohens_d']:.2f}")
        print(f"  lerp floor: mean depth {floor['lerp_depth_mean']:.3f} vs real "
              f"{floor['real_depth_mean']:.3f}; separated={floor['separation_ok']}")
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
