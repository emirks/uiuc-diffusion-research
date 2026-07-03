"""exp_053 — rubric judge over a manifest, Gemini API backend (native video).

Login-node safe: no GPU, no torch. API key from $GEMINI_API_KEY or
~/.config/gemini/api_key. Every raw response is cached under the config's
judge cache dir, so re-runs cost nothing and every verdict is auditable.
"""

import argparse
import json
import pathlib
import os
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.exp_utils import TeeLogger, load_config, next_run_dir  # noqa: E402
from diffusion.transition_eval.judge_gemini import GeminiJudge  # noqa: E402
from diffusion.transition_eval.manifest import load_manifest, scan_references  # noqa: E402
from diffusion.transition_eval.rubric import RUBRIC_QUESTIONS, judge_pass_rate  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def api_key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    key_file = pathlib.Path.home() / ".config" / "gemini" / "api_key"
    if key_file.exists():
        return key_file.read_text().strip()
    sys.exit("no GEMINI_API_KEY and no ~/.config/gemini/api_key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", default="ladder")
    args = ap.parse_args()
    cfg = load_config(pathlib.Path(args.config))
    jc = cfg["judge_gemini"]

    items = load_manifest(REPO_ROOT / args.manifest if not os.path.isabs(args.manifest)
                          else pathlib.Path(args.manifest))
    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / f"judge_gemini_{args.label}"
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        refs = scan_references(REPO_ROOT / cfg["data"]["transitions_root"],
                               exclude=tuple(cfg["data"]["exclude"]))
        judge = GeminiJudge(api_key=api_key(), model=jc["model"], fps=jc["fps"],
                            cache_dir=REPO_ROOT / jc["cache_dir"] / args.label)
        print(f"[info] {run_id}: {len(items)} items, model {jc['model']} @ {jc['fps']} fps")

        results = {}
        for it in items:
            ref_path = refs[it.style][0]  # fixed canonical reference per style
            gen_path = REPO_ROOT / it.generated_video
            try:
                res = judge.judge(ref_path, gen_path, item_id=it.item_id)
            except Exception as e:  # judge stays advisory; never sink the run
                res = {"parse_error": True, "_raw": f"EXCEPTION: {e}"}
            results[it.item_id] = {**res, "_arm": it.arm}
            answers = {q: (res.get(q, {}) or {}).get("answer") for q in RUBRIC_QUESTIONS}
            print(f"[info] {it.item_id}: {answers}"
                  f"{' (cached)' if res.get('_cached') else ''}")

        summary = {}
        for arm in sorted({it.arm for it in items}):
            summary[arm] = judge_pass_rate(
                [r for r in results.values() if r["_arm"] == arm])
        (run_dir / "judge_results.json").write_text(json.dumps(results, indent=2))
        (run_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2))
        print("[summary]")
        for arm, s in summary.items():
            print(f"  {arm}: " + " ".join(f"{q}={s[q]:.2f}" for q in RUBRIC_QUESTIONS)
                  + f" all_pass={s['all_pass']:.2f} parsed={s['n_parsed']}")
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
