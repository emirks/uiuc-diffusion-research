"""exp_053 checks A/B/C — one GPU job, mostly cache hits.

A: all-frames vs core-mask effect appearance on the 41-real-clip exam
   (retrieval + lerp floor/ceiling separation per variant) -> ADOPT/KEEP-MASK.
B: motion sanity from the cached exp_052 distance matrix — per-style
   within-style fidelity must beat cross-style.
C: adversarial manifest (known near-copies + live-branch negatives) — the
   pre-registered hard bar: every src_copy/donor_pin/self_inject_g1 item must
   show leak_max_sim_target >= copy_min_leak, else M6 is broken for real
   cheats and NOTHING else ships until it is fixed.
"""

import argparse
import json
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
from diffusion.transition_eval.pipeline import process_video, process_video_file  # noqa: E402
from diffusion.transition_eval.report import md_table, retrieval_eval  # noqa: E402
from diffusion.transition_eval.video_io import load_frames  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest_adversarial.json"


def paired_separation(real_sims, lerp_sims):
    margin = np.asarray(real_sims) - np.asarray(lerp_sims)
    d = float(margin.mean() / (margin.std(ddof=1) + 1e-12))
    return {"sep_frac": float((margin > 0).mean()), "sep_d": d,
            "margin_mean": float(margin.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()
    cfg = load_config(pathlib.Path(args.config))

    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / "checks"
    run_id, run_dir = next_run_dir(out_dir)
    cache_dir = REPO_ROOT / cfg["features"]["cache_dir"]
    mk, short = cfg["morph"], cfg["features"]["short_side"]
    device = cfg["runtime"]["device"]
    val_dir = REPO_ROOT / cfg["validation_run"]

    with TeeLogger(run_dir / "run.log"):
        (run_dir / "config_snapshot.yaml").write_text(pathlib.Path(args.config).read_text())
        extractor = DinoExtractor(cfg["features"]["model"], device=device)

        refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                               exclude=tuple(cfg["data"]["exclude"]))
        names, styles, bundles, lerp_bundles = [], [], {}, {}
        for style, vids in refs.items():
            for vid in vids:
                name = f"{style}/{vid.stem}"
                b, frames = process_video_file(vid, cache_dir, extractor, tracker=None,
                                               short_side=short,
                                               n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
                lerp = make_lerp(frames[:mk["n_prefix"]], frames[-mk["n_suffix"]:], len(frames))
                lb = process_video(lerp, b.key + ":lerp", cache_dir, extractor, tracker=None,
                                   n_prefix=mk["n_prefix"], n_suffix=mk["n_suffix"])
                names.append(name); styles.append(style)
                bundles[name] = b; lerp_bundles[name] = lb
        n = len(names)
        print(f"[info] {run_id}: {len(refs)} styles, {n} real clips ready")

        # ---------------- Check A: all-frames vs core-mask M3 -----------------
        feats_all = {nm: bundles[nm].feats for nm in names}
        feats_core = {nm: bundles[nm].feats[bundles[nm].core] for nm in names}
        D = {"core": np.zeros((n, n)), "all": np.zeros((n, n))}
        for i in range(n):
            for j in range(i + 1, n):
                D["core"][i, j] = D["core"][j, i] = 1 - set_similarity(
                    feats_core[names[i]], feats_core[names[j]])
                D["all"][i, j] = D["all"][j, i] = 1 - set_similarity(
                    feats_all[names[i]], feats_all[names[j]])
        checkA = {}
        for variant in ("core", "all"):
            res = retrieval_eval(D[variant], styles)
            fsets = feats_core if variant == "core" else feats_all
            real_s, lerp_s = [], []
            for i, nm in enumerate(names):
                mates = [names[j] for j in range(n) if styles[j] == styles[i] and j != i]
                if not mates:
                    continue
                real_s.append(max(set_similarity(fsets[nm], fsets[m]) for m in mates))
                lf = (lerp_bundles[nm].feats[lerp_bundles[nm].core] if variant == "core"
                      else lerp_bundles[nm].feats)
                lerp_s.append(max(set_similarity(lf, fsets[m]) for m in mates))
            checkA[variant] = {"retrieval": {k: res[k] for k in
                               ("accuracy_1nn", "accuracy_wilson95", "separation_cohens_d",
                                "per_class_recall")},
                               "floor_ceiling": paired_separation(real_s, lerp_s)}
        bar = cfg["checks"]["allframes_adopt"]
        a = checkA["all"]
        adopt = (a["retrieval"]["accuracy_1nn"] >= bar["min_acc"]
                 and a["floor_ceiling"]["sep_frac"] >= bar["min_sep_frac"]
                 and a["floor_ceiling"]["sep_d"] >= bar["min_sep_d"])
        checkA["verdict"] = "ADOPT_ALL_FRAMES" if adopt else "KEEP_CORE_MASK"
        print(f"[checkA] core acc={checkA['core']['retrieval']['accuracy_1nn']:.3f} "
              f"all acc={a['retrieval']['accuracy_1nn']:.3f} -> {checkA['verdict']}")

        # ---------------- Check B: motion sanity from cached matrix -----------
        z = np.load(val_dir / "distance_matrices.npz", allow_pickle=True)
        vnames, vstyles, Dm = list(z["names"]), list(z["styles"]), z["motion"]
        fid = 1.0 - Dm
        checkB = {"per_style": {}, "clips_within_gt_cross": None}
        wins = []
        for i, nm in enumerate(vnames):
            same = [j for j in range(len(vnames)) if vstyles[j] == vstyles[i] and j != i]
            diff = [j for j in range(len(vnames)) if vstyles[j] != vstyles[i]]
            wi = np.nanmean(fid[i, same]) if same else np.nan
            cr = np.nanmean(fid[i, diff])
            if np.isfinite(wi) and np.isfinite(cr):
                wins.append(wi > cr)
        checkB["clips_within_gt_cross"] = float(np.mean(wins))
        for s in sorted(set(vstyles)):
            idx = [i for i, st in enumerate(vstyles) if st == s]
            w = np.nanmean([fid[i, j] for i in idx for j in idx if i != j])
            c = np.nanmean([fid[i, j] for i in idx for j in range(len(vnames)) if vstyles[j] != s])
            checkB["per_style"][s] = {"within": float(w), "cross": float(c),
                                      "ok": bool(np.isfinite(w) and np.isfinite(c) and w > c)}
        n_ok = sum(1 for v in checkB["per_style"].values() if v["ok"])
        checkB["styles_ok"] = f"{n_ok}/{len(checkB['per_style'])}"
        print(f"[checkB] within>cross for {checkB['styles_ok']} styles; "
              f"per-clip fraction {checkB['clips_within_gt_cross']:.2f}")

        # ---------------- Check C: adversarial manifest -----------------------
        lpips_scorer = LpipsScorer(device=device)
        style_frames = {}
        for s in refs:
            style_frames[s] = np.concatenate([bundles[f"{s}/{v.stem}"].feats for v in refs[s]])
        items = load_manifest(MANIFEST_PATH)
        adv_rows = []
        for it in items:
            n_pre = it.condition_prefix.num_frames if it.condition_prefix else mk["n_prefix"]
            n_suf = it.condition_suffix.num_frames if it.condition_suffix else mk["n_suffix"]
            gb, gen_frames = process_video_file(
                REPO_ROOT / it.generated_video, cache_dir, extractor, tracker=None,
                short_side=short, n_prefix=n_pre, n_suffix=n_suf)
            row = {"item_id": it.item_id, "arm": it.arm, "notes": it.notes,
                   "scalar_depth": gb.scalars["depth"],
                   "scalar_cross": gb.profile["cross"],
                   "scalar_cross_high": gb.profile["cross_high"]}
            others = {s: F for s, F in style_frames.items() if s != it.style}
            row.update({f"leak_{k}": v for k, v in
                        leakage(gb.feats[gb.core], style_frames[it.style], others).items()})
            core = gb.feats[gb.core]
            row["appearance_core_best"] = float(max(
                set_similarity(core, bundles[f"{it.style}/{v.stem}"].feats[
                    bundles[f"{it.style}/{v.stem}"].core]) for v in refs[it.style]))
            if it.condition_prefix:
                pre, _ = load_frames(REPO_ROOT / it.condition_prefix.video, short_side=short)
                row.update(endpoint_fidelity(gen_frames, gb.feats, pre[:n_pre],
                                             lambda f: extractor.extract(f), lpips_scorer, "prefix"))
            if it.condition_suffix:
                suf, _ = load_frames(REPO_ROOT / it.condition_suffix.video, short_side=short)
                row.update(endpoint_fidelity(gen_frames, gb.feats, suf[-n_suf:],
                                             lambda f: extractor.extract(f), lpips_scorer, "suffix"))
            row.update(seam_scores(temporal_lpips(gen_frames, lpips_scorer), n_pre, n_suf))
            adv_rows.append(row)
            print(f"[checkC] {it.item_id}: leak_max={row['leak_max_sim_target']:.3f} "
                  f"excess={row['leak_excess']:.3f}")
        extractor.free(); lpips_scorer.free()

        adv_cfg = cfg["checks"]["adversarial"]
        hard = [r for r in adv_rows if r["arm"] in adv_cfg["hard_bar_arms"]]
        misses = [r["item_id"] for r in hard
                  if r["leak_max_sim_target"] < adv_cfg["copy_min_leak"]]
        checkC = {"hard_bar": adv_cfg["copy_min_leak"], "n_hard_items": len(hard),
                  "misses": misses,
                  "verdict": "M6_OK" if not misses else "M6_BROKEN_FIX_BEFORE_USE"}
        print(f"[checkC] hard-bar items {len(hard)}, misses {len(misses)} -> {checkC['verdict']}")

        # ---------------- figure + outputs ------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arms = sorted({r["arm"] for r in adv_rows})
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for k, arm in enumerate(arms):
            ys = [r["leak_max_sim_target"] for r in adv_rows if r["arm"] == arm]
            ax.scatter(np.full(len(ys), k) + np.linspace(-0.12, 0.12, len(ys)), ys, s=22)
        ax.axhline(adv_cfg["copy_min_leak"], color="red", ls="--", lw=1, label="hard bar 0.88")
        ax.axhline(0.78, color="gray", ls=":", lw=1, label="exp_051 honest max 0.78")
        ax.set_xticks(range(len(arms)), arms, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("leak max sim")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "leak_scatter.png", dpi=140)
        plt.close(fig)

        with open(run_dir / "adversarial.jsonl", "w") as f:
            for r in adv_rows:
                f.write(json.dumps(r) + "\n")
        (run_dir / "checks.json").write_text(json.dumps(
            {"checkA": checkA, "checkB": checkB, "checkC": checkC}, indent=2))

        arm_rows = []
        for arm in arms:
            sub = [r for r in adv_rows if r["arm"] == arm]
            arm_rows.append([arm, len(sub),
                             float(np.mean([r["leak_max_sim_target"] for r in sub])),
                             float(np.min([r["leak_max_sim_target"] for r in sub])),
                             float(np.mean([r["leak_excess"] for r in sub])),
                             float(np.mean([r.get("prefix_dino", np.nan) for r in sub])),
                             float(np.mean([r["max_seam_z"] for r in sub]))])
        rep = ["# exp_053 checks\n",
               f"## A — M3 all-frames vs core-mask -> **{checkA['verdict']}**\n",
               md_table(["variant", "1-NN acc", "wilson95", "cohens_d", "sep_frac", "sep_d"],
                        [[v,
                          checkA[v]["retrieval"]["accuracy_1nn"],
                          str([round(x, 2) for x in checkA[v]["retrieval"]["accuracy_wilson95"]]),
                          checkA[v]["retrieval"]["separation_cohens_d"],
                          checkA[v]["floor_ceiling"]["sep_frac"],
                          checkA[v]["floor_ceiling"]["sep_d"]] for v in ("core", "all")]),
               f"\n## B — motion within>cross: {checkB['styles_ok']} styles, "
               f"per-clip {checkB['clips_within_gt_cross']:.2f}\n",
               md_table(["style", "within", "cross", "ok"],
                        [[s, v["within"], v["cross"], str(v["ok"])]
                         for s, v in checkB["per_style"].items()]),
               f"\n## C — adversarial near-copies -> **{checkC['verdict']}** "
               f"(bar {adv_cfg['copy_min_leak']}, misses: {misses or 'none'})\n",
               md_table(["arm", "n", "leak max sim (mean)", "(min)", "excess", "prefix_dino",
                         "max seam z"], arm_rows)]
        (run_dir / "checks_report.md").write_text("\n".join(rep))

        if cfg["wandb"]["enabled"]:
            try:
                import wandb
                run = wandb.init(project=cfg["wandb"]["project"], name="exp053_checks",
                                 tags=["exp_053", "eval-harness", "checks"])
                run.summary.update({
                    "checkA/verdict": checkA["verdict"],
                    "checkA/acc_all": a["retrieval"]["accuracy_1nn"],
                    "checkA/acc_core": checkA["core"]["retrieval"]["accuracy_1nn"],
                    "checkB/styles_ok": checkB["styles_ok"],
                    "checkC/verdict": checkC["verdict"]})
                run.log({"leak_scatter": wandb.Image(str(run_dir / "leak_scatter.png"))})
                run.finish()
            except Exception as e:
                print(f"[warn] wandb logging failed: {e}")
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
