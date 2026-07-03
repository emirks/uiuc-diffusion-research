"""exp_053 scorer v2 — the standard scoring path going forward.

Changes vs exp_052/run_score.py: endpoint windows come from the manifest's
condition num_frames (9/8 only as fallback), the M1 cross-similarity edge
guard is propagated, the inline local-judge path is GONE (judges run via
run_judge_gemini.py / exp_052's run_judge.py and merge at report time), and
reporting is the library's headline/analysis split with per-style trust flags
and mean±std everywhere.

`--from-items <items.jsonl>` re-reports an existing scoring run without
recomputation (numpy-only; login-node safe).
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
from diffusion.transition_eval.appearance import leakage, set_similarity  # noqa: E402
from diffusion.transition_eval.controls import make_lerp  # noqa: E402
from diffusion.transition_eval.endpoints import (  # noqa: E402
    LpipsScorer, endpoint_fidelity, seam_scores, temporal_lpips,
)
from diffusion.transition_eval.features import DinoExtractor  # noqa: E402
from diffusion.transition_eval.manifest import load_manifest, scan_references  # noqa: E402
from diffusion.transition_eval.morph import profile_distance  # noqa: E402
from diffusion.transition_eval.motion import Tracker, motion_fidelity  # noqa: E402
from diffusion.transition_eval.pipeline import process_video, process_video_file  # noqa: E402
from diffusion.transition_eval.report import (  # noqa: E402
    md_table, normalize_score, score_tables, trust_flags,
)
from diffusion.transition_eval.video_io import load_frames  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def resolve(p: str) -> pathlib.Path:
    p = os.path.expandvars(p)
    return pathlib.Path(p) if os.path.isabs(p) else REPO_ROOT / p


def fidelity_vs_refs(bundle, ref_bundles, n_resample, n_steps):
    dtws, apps, mfs = [], [], []
    core = bundle.feats[bundle.core]
    for rb in ref_bundles:
        dtws.append(profile_distance(bundle.profile, rb.profile, n=n_resample)["dtw"])
        apps.append(set_similarity(core, rb.feats[rb.core]))
        if bundle.tracks is not None and rb.tracks is not None:
            mfs.append(motion_fidelity(bundle.tracks, bundle.vis, rb.tracks, rb.vis, n_steps=n_steps))
    mfs = [m for m in mfs if np.isfinite(m)]
    return {"profile_dtw_best": float(np.min(dtws)), "profile_dtw_mean": float(np.mean(dtws)),
            "appearance_best": float(np.max(apps)), "appearance_mean": float(np.mean(apps)),
            "motion_fidelity_mean": float(np.mean(mfs)) if mfs else float("nan")}


def load_trust(cfg):
    val = json.loads((REPO_ROOT / cfg["validation_run"] / "results.json").read_text())
    refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                           exclude=tuple(cfg["data"]["exclude"]))
    return trust_flags(val, {s: len(v) for s, v in refs.items()},
                       motion_recall_min=cfg["trust"]["motion_recall_min"],
                       min_ceiling_clips=cfg["trust"]["min_ceiling_clips"])


def write_report(run_dir, rows, cfg, label, judge_summary_path=None):
    trust = load_trust(cfg)
    judge_by_arm = None
    if judge_summary_path and pathlib.Path(judge_summary_path).exists():
        judge_by_arm = json.loads(pathlib.Path(judge_summary_path).read_text())
    report = [f"# harness scores — {label} ({len(rows)} items)\n",
              "Normalized scores: 0 = lerp-control floor, 1 = real-clip LOO ceiling.\n",
              score_tables(rows, trust=trust, judge_by_arm=judge_by_arm)]
    (run_dir / "report.md").write_text("\n".join(report))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted({r["arm"] for r in rows})
    metrics = [("norm_appearance_best", "appearance"),
               ("norm_motion_fidelity_mean", "motion"),
               ("norm_profile_dtw_best", "profile DTW")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        for k, arm in enumerate(arms):
            ys = [r.get(col) for r in rows if r["arm"] == arm]
            ys = [y for y in ys if y is not None and np.isfinite(y)]
            x = np.full(len(ys), k) + np.linspace(-0.12, 0.12, max(len(ys), 1))[:len(ys)]
            ax.scatter(x, ys, s=20, alpha=0.8)
            if ys:
                ax.hlines(np.mean(ys), k - 0.2, k + 0.2, color="black", lw=1.5)
        ax.set_xticks(range(len(arms)), arms, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
    axes[0].set_ylabel("normalized score")
    fig.tight_layout()
    fig.savefig(run_dir / "scatter.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--manifest")
    ap.add_argument("--from-items", help="existing items.jsonl: re-report only, no compute")
    ap.add_argument("--judge-summary", help="judge_summary.json to merge into the headline")
    ap.add_argument("--label", default="score")
    args = ap.parse_args()
    cfg = load_config(pathlib.Path(args.config))

    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / args.label
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        (run_dir / "config_snapshot.yaml").write_text(pathlib.Path(args.config).read_text())

        if args.from_items:
            rows = [json.loads(l) for l in pathlib.Path(resolve(args.from_items)).read_text().splitlines() if l.strip()]
            print(f"[info] {run_id}: re-reporting {len(rows)} items from {args.from_items}")
            write_report(run_dir, rows, cfg, args.label, args.judge_summary)
            print(f"[done] {run_id} -> {run_dir}")
            return

        assert args.manifest, "--manifest required unless --from-items"
        items = load_manifest(resolve(args.manifest))
        cache_dir = REPO_ROOT / cfg["features"]["cache_dir"]
        mk, short = cfg["morph"], cfg["features"]["short_side"]
        device = cfg["runtime"]["device"]
        print(f"[info] {run_id}: {len(items)} items from {args.manifest}")

        extractor = DinoExtractor(cfg["features"]["model"], device=device)
        tracker = Tracker(device=device, grid_size=cfg["motion"]["grid_size"],
                          max_side=cfg["motion"]["max_side"])
        lpips_scorer = LpipsScorer(device=device)

        refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                               exclude=tuple(cfg["data"]["exclude"]))
        ref_bundles, style_frames = {}, {}
        for style, vids in refs.items():
            ref_bundles[style] = []
            for vid in vids:
                b, _ = process_video_file(vid, cache_dir, extractor, tracker, short_side=short,
                                          n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
                ref_bundles[style].append(b)
            style_frames[style] = np.concatenate([b.feats for b in ref_bundles[style]])

        ceilings = {}
        for style, bs in ref_bundles.items():
            crows = [fidelity_vs_refs(b, [o for o in bs if o is not b],
                                      mk["resample_points"], cfg["motion"]["n_steps"]) for b in bs]
            ceilings[style] = {k: float(np.mean([r[k] for r in crows])) for k in crows[0]}

        rows = []
        for it in items:
            n_pre = it.condition_prefix.num_frames if it.condition_prefix else mk["n_prefix"]
            n_suf = it.condition_suffix.num_frames if it.condition_suffix else mk["n_suffix"]
            print(f"[info] scoring {it.item_id} (windows {n_pre}/{n_suf})")
            gb, gen_frames = process_video_file(
                resolve(it.generated_video), cache_dir, extractor, tracker, short_side=short,
                n_prefix=n_pre, n_suffix=n_suf, n_endpoints=it.n_endpoints)
            style_refs = ref_bundles[it.style]
            row = {"item_id": it.item_id, "arm": it.arm, "style": it.style,
                   "n_endpoints": it.n_endpoints,
                   **{f"scalar_{k}": v for k, v in gb.scalars.items()},
                   "scalar_cross": gb.profile["cross"],
                   "scalar_cross_high": gb.profile["cross_high"]}
            row.update(fidelity_vs_refs(gb, style_refs, mk["resample_points"], cfg["motion"]["n_steps"]))

            if it.condition_prefix and it.condition_suffix:
                pre, _ = load_frames(resolve(it.condition_prefix.video), short_side=short)
                suf, _ = load_frames(resolve(it.condition_suffix.video), short_side=short)
                pre, suf = pre[:n_pre], suf[-n_suf:]
            else:
                pre, suf = gen_frames[:n_pre], gen_frames[-n_suf:]
            lerp = make_lerp(pre, suf, len(gen_frames))
            lb = process_video(lerp, gb.key + ":lerpfloor", cache_dir, extractor, tracker,
                               n_prefix=n_pre, n_suffix=n_suf)
            floor = fidelity_vs_refs(lb, style_refs, mk["resample_points"], cfg["motion"]["n_steps"])
            if not np.isfinite(floor["motion_fidelity_mean"]):
                floor["motion_fidelity_mean"] = 0.0
            ceil = ceilings[it.style]
            for k, hib in [("profile_dtw_best", False), ("appearance_best", True),
                           ("motion_fidelity_mean", True)]:
                row[f"norm_{k}"] = normalize_score(row[k], floor[k], ceil[k], higher_better=hib)
                row[f"floor_{k}"], row[f"ceil_{k}"] = floor[k], ceil[k]

            if it.condition_prefix:
                row.update(endpoint_fidelity(gen_frames, gb.feats, pre,
                                             lambda f: extractor.extract(f), lpips_scorer, "prefix"))
            if it.condition_suffix:
                row.update(endpoint_fidelity(gen_frames, gb.feats, suf,
                                             lambda f: extractor.extract(f), lpips_scorer, "suffix"))
            row.update(seam_scores(temporal_lpips(gen_frames, lpips_scorer), n_pre, n_suf))
            others = {s: F for s, F in style_frames.items() if s != it.style}
            row.update({f"leak_{k}": v for k, v in
                        leakage(gb.feats[gb.core], style_frames[it.style], others).items()})
            rows.append(row)

        extractor.free(); tracker.free(); lpips_scorer.free()

        with open(run_dir / "items.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        (run_dir / "ceilings.json").write_text(json.dumps(ceilings, indent=2))
        write_report(run_dir, rows, cfg, args.label, args.judge_summary)

        if cfg["wandb"]["enabled"]:
            try:
                import wandb
                run = wandb.init(project=cfg["wandb"]["project"], name=f"exp053_{args.label}",
                                 tags=["exp_053", "eval-harness", args.label],
                                 config={"manifest": str(args.manifest), "n_items": len(rows)})
                table = wandb.Table(columns=list(rows[0].keys()),
                                    data=[[r.get(k) for k in rows[0].keys()] for r in rows])
                run.log({"items": table})
                run.finish()
            except Exception as e:
                print(f"[warn] wandb logging failed: {e}")
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
