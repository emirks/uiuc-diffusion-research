"""exp_052 — standalone rubric-judge pass.

Runs in the LTX-2-official venv ($LAB/LTX-2-official/.venv/bin/python): the
diffusion conda env's torch 2.5.1 predates the torch>=2.6 attention-mask path
transformers 4.57 uses for Gemma 3. Only needs torch/transformers/av/numpy +
the transition_eval package via PYTHONPATH.

Reads the eval manifest, judges each generated video against the style's
canonical reference clip, writes judge_results.json + judge_summary.json into
the given results run dir (alongside items.jsonl).
"""

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from diffusion.transition_eval.judge import RubricJudge, judge_pass_rate  # noqa: E402
from diffusion.transition_eval.manifest import load_manifest, scan_references  # noqa: E402
from diffusion.transition_eval.video_io import load_frames  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def resolve(p: str) -> pathlib.Path:
    p = os.path.expandvars(p)
    return pathlib.Path(p) if os.path.isabs(p) else REPO_ROOT / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="results run dir (holds items.jsonl)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = resolve(args.out)

    items = load_manifest(resolve(args.manifest))
    refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                           exclude=tuple(cfg["data"]["exclude"]))
    short = cfg["features"]["short_side"]

    print(f"[info] judging {len(items)} items")
    judge = RubricJudge(os.path.expandvars(cfg["judge"]["model_path"]))
    ref_cache = {}
    results = {}
    for it in items:
        if it.style not in ref_cache:
            ref_cache[it.style], _ = load_frames(refs[it.style][0], short_side=short)
        gen_frames, _ = load_frames(resolve(it.generated_video), short_side=short)
        try:
            res = judge.judge(ref_cache[it.style], gen_frames, n_frames=cfg["judge"]["n_frames"])
        except Exception as e:
            res = {"parse_error": True, "_raw": f"EXCEPTION: {e}"}
        results[it.item_id] = res
        print(f"[info] {it.item_id}: " + json.dumps(
            {q: (res.get(q, {}) or {}).get("answer") for q in
             ("q1_same_type", "q2_dynamics", "q3_endpoints", "q4_leakage", "q5_artifacts")}))
    judge.free()

    (out_dir / "judge_results.json").write_text(json.dumps(results, indent=2))
    arms = sorted({it.arm for it in items})
    summary = {arm: judge_pass_rate([results[it.item_id] for it in items if it.arm == arm])
               for arm in arms}
    (out_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    print("[summary]")
    for arm, s in summary.items():
        print(f"  {arm}: " + json.dumps({k: round(v, 2) if isinstance(v, float) else v
                                         for k, v in s.items()}))
    print(f"[done] -> {out_dir}/judge_summary.json")


if __name__ == "__main__":
    main()
