"""exp_052 phase 2 — score a manifest of generated videos through the harness.

Per item: morph-profile fidelity vs the style's reference profiles, effect
appearance on core frames, motion fidelity, endpoint fidelity + boundary
seams (when condition clips are given), leakage retrieval, and the
experimental rubric judge. Every fidelity number is reported raw AND
normalized between the item's own lerp-control floor and the style's real-clip
leave-one-out ceiling — computed by the identical pipeline.
"""

import argparse
import dataclasses
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
from diffusion.transition_eval.report import md_table, normalize_score  # noqa: E402
from diffusion.transition_eval.video_io import load_frames  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def resolve(p: str) -> pathlib.Path:
    p = os.path.expandvars(p)
    return pathlib.Path(p) if os.path.isabs(p) else REPO_ROOT / p


def fidelity_vs_refs(bundle, ref_bundles, n_resample, n_steps):
    """Transition-fidelity numbers for one video against a reference set."""
    dtws, pearsons, apps, mfs = [], [], [], []
    core = bundle.feats[bundle.core]
    for rb in ref_bundles:
        pd = profile_distance(bundle.profile, rb.profile, n=n_resample)
        dtws.append(pd["dtw"])
        pearsons.append(pd["pearson"])
        apps.append(set_similarity(core, rb.feats[rb.core]))
        if bundle.tracks is not None and rb.tracks is not None:
            mfs.append(motion_fidelity(bundle.tracks, bundle.vis, rb.tracks, rb.vis, n_steps=n_steps))
    mfs = [m for m in mfs if np.isfinite(m)]
    return {"profile_dtw_best": float(np.min(dtws)), "profile_dtw_mean": float(np.mean(dtws)),
            "profile_pearson_best": float(np.max(pearsons)),
            "appearance_best": float(np.max(apps)), "appearance_mean": float(np.mean(apps)),
            "motion_fidelity_mean": float(np.mean(mfs)) if mfs else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", default="ladder", help="output subdir name")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(pathlib.Path(args.config))

    items = load_manifest(resolve(args.manifest))
    if args.smoke:
        items = items[:cfg["smoke"]["max_items"]]
    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / (args.label + ("_smoke" if args.smoke else ""))
    run_id, run_dir = next_run_dir(out_dir)
    cache_dir = REPO_ROOT / cfg["features"]["cache_dir"]
    mk, short = cfg["morph"], cfg["features"]["short_side"]
    device = cfg["runtime"]["device"]

    with TeeLogger(run_dir / "run.log"):
        (run_dir / "config_snapshot.yaml").write_text(pathlib.Path(args.config).read_text())
        print(f"[info] {run_id}: {len(items)} items from {args.manifest}")

        extractor = DinoExtractor(cfg["features"]["model"], device=device)
        tracker = Tracker(device=device, grid_size=cfg["motion"]["grid_size"],
                          max_side=cfg["motion"]["max_side"])
        lpips_scorer = LpipsScorer(device=device)

        # --- references: all styles (leakage contrast needs the others too) ---
        refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                               exclude=tuple(cfg["data"]["exclude"]))
        ref_bundles, style_frames = {}, {}
        for style, vids in refs.items():
            ref_bundles[style] = []
            feats_all = []
            for vid in vids:
                b, _ = process_video_file(vid, cache_dir, extractor, tracker, short_side=short,
                                          n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
                ref_bundles[style].append(b)
                feats_all.append(b.feats)
            style_frames[style] = np.concatenate(feats_all)
        print(f"[info] references ready: { {s: len(v) for s, v in ref_bundles.items()} }")

        # --- real-clip LOO ceilings per style ---------------------------------
        ceilings = {}
        for style, bs in ref_bundles.items():
            rows = [fidelity_vs_refs(b, [o for o in bs if o is not b],
                                     mk["resample_points"], cfg["motion"]["n_steps"])
                    for b in bs]
            ceilings[style] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

        # --- score items -------------------------------------------------------
        rows = []
        judge_inputs = []
        for it in items:
            print(f"[info] scoring {it.item_id}")
            gen_path = resolve(it.generated_video)
            gb, gen_frames = process_video_file(
                gen_path, cache_dir, extractor, tracker, short_side=short,
                n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"], n_endpoints=it.n_endpoints)
            style_refs = ref_bundles[it.style]
            row = {"item_id": it.item_id, "arm": it.arm, "style": it.style,
                   "n_endpoints": it.n_endpoints, **{f"scalar_{k}": v for k, v in gb.scalars.items()}}
            row.update(fidelity_vs_refs(gb, style_refs, mk["resample_points"], cfg["motion"]["n_steps"]))

            # lerp floor from the item's own endpoints (condition clips if given)
            if it.condition_prefix and it.condition_suffix:
                pre, _ = load_frames(resolve(it.condition_prefix.video), short_side=short)
                suf, _ = load_frames(resolve(it.condition_suffix.video), short_side=short)
                pre, suf = pre[:it.condition_prefix.num_frames], suf[-it.condition_suffix.num_frames:]
            else:
                pre, suf = gen_frames[:mk["n_prefix"]], gen_frames[-mk["n_suffix"]:]
            lerp = make_lerp(pre, suf, len(gen_frames))
            lb = process_video(lerp, gb.key + ":lerpfloor", cache_dir, extractor, tracker,
                               n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
            floor = fidelity_vs_refs(lb, style_refs, mk["resample_points"], cfg["motion"]["n_steps"])
            if not np.isfinite(floor["motion_fidelity_mean"]):
                floor["motion_fidelity_mean"] = 0.0  # a static crossfade carries zero motion
            ceil = ceilings[it.style]
            for k, hib in [("profile_dtw_best", False), ("appearance_best", True),
                           ("motion_fidelity_mean", True)]:
                row[f"norm_{k}"] = normalize_score(row[k], floor[k], ceil[k], higher_better=hib)
                row[f"floor_{k}"], row[f"ceil_{k}"] = floor[k], ceil[k]

            # endpoint fidelity + seams (2-endpoint items with conditions)
            if it.condition_prefix:
                row.update(endpoint_fidelity(gen_frames, gb.feats, pre,
                                             lambda f: extractor.extract(f), lpips_scorer, "prefix"))
            if it.condition_suffix:
                row.update(endpoint_fidelity(gen_frames, gb.feats, suf,
                                             lambda f: extractor.extract(f), lpips_scorer, "suffix"))
            d = temporal_lpips(gen_frames, lpips_scorer)
            row.update(seam_scores(d, mk["n_prefix"], mk["n_suffix"]))

            others = {s: F for s, F in style_frames.items() if s != it.style}
            row.update({f"leak_{k}": v for k, v in
                        leakage(gb.feats[gb.core], style_frames[it.style], others).items()})
            rows.append(row)
            judge_inputs.append((it, gen_path))

        extractor.free()
        tracker.free()
        lpips_scorer.free()

        # --- experimental rubric judge (loads Gemma AFTER the others free) ----
        if cfg["judge"]["enabled"] and not args.no_judge and not args.smoke:
            from diffusion.transition_eval.judge import RubricJudge, judge_pass_rate
            print("[info] loading judge model")
            judge = RubricJudge(os.path.expandvars(cfg["judge"]["model_path"]), device=device)
            judge_results = {}
            for (it, gen_path), row in zip(judge_inputs, rows):
                ref_path = refs[it.style][0]  # fixed canonical reference per style
                ref_frames, _ = load_frames(ref_path, short_side=short)
                gen_frames, _ = load_frames(gen_path, short_side=short)
                try:
                    res = judge.judge(ref_frames, gen_frames, n_frames=cfg["judge"]["n_frames"])
                except Exception as e:  # judge is experimental; never sink the run
                    res = {"parse_error": True, "_raw": f"EXCEPTION: {e}"}
                judge_results[it.item_id] = res
                for q in ("q1_same_type", "q2_dynamics", "q3_endpoints", "q4_leakage", "q5_artifacts"):
                    ans = res.get(q, {})
                    row[f"judge_{q}"] = ans.get("answer") if isinstance(ans, dict) else None
                print(f"[info] judged {it.item_id}: "
                      f"{ {q: row.get('judge_' + q) for q in ('q1_same_type', 'q3_endpoints')} }")
            judge.free()
            (run_dir / "judge_results.json").write_text(json.dumps(judge_results, indent=2))

        # --- outputs ------------------------------------------------------------
        with open(run_dir / "items.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        arms = sorted({r["arm"] for r in rows})
        agg_cols = ["norm_profile_dtw_best", "norm_appearance_best", "norm_motion_fidelity_mean",
                    "scalar_depth", "prefix_dino", "suffix_dino", "max_seam_z", "leak_excess"]
        agg_rows = []
        for arm in arms:
            sub = [r for r in rows if r["arm"] == arm]
            agg_rows.append([arm, len(sub)] + [
                float(np.nanmean([r.get(c, np.nan) for r in sub])) for c in agg_cols])
        report = ["# exp_052 — harness scores\n",
                  f"manifest: {args.manifest} ({len(rows)} items)\n",
                  "Normalized scores: 0 = lerp-control floor, 1 = real-clip LOO ceiling.\n",
                  md_table(["arm", "n"] + agg_cols, agg_rows)]
        (run_dir / "report.md").write_text("\n".join(report))
        (run_dir / "ceilings.json").write_text(json.dumps(ceilings, indent=2))

        if cfg["wandb"]["enabled"] and not args.smoke:
            try:
                import wandb
                run = wandb.init(project=cfg["wandb"]["project"], name=f"exp052_{args.label}",
                                 tags=["exp_052", "eval-harness", args.label],
                                 config={"manifest": str(args.manifest), "n_items": len(rows)})
                table = wandb.Table(columns=list(rows[0].keys()),
                                    data=[[r.get(k) for k in rows[0].keys()] for r in rows])
                run.log({"items": table})
                run.finish()
            except Exception as e:
                print(f"[warn] wandb logging failed: {e}")

        print("[summary] per-arm normalized means:")
        for r in agg_rows:
            print("  ", r)
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
