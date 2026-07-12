"""Near-duplicate audit of split v1 (GPU job; see scripts/job_audit_split_v1.sbatch).

For each class with test items, computes the M2a copy score (certified
machinery: diffusion.transition_eval.m2_integrity.copy_score over per-frame
DINOv2 features, ALL frames on both sides) between every cross-boundary
(train, test) clip pair within the class. A pair flags at
copy_max >= 0.858 (tau_copy, certification record v3.0.0-amendment-1).

Pre-registered remediation is applied mechanically INSIDE this script by
re-invoking scripts/build_split_v1.py's split logic with the accumulated flag
set (same RNG stream, next sample; all-flagged class -> all-train) and
re-scanning replacements, until convergence. Outputs:

  data/processed/transitions_std121/split_v1_flagged.json   (flag set, may be empty)
  outputs/eval/split_v1_audit/audit_results.json            (every pair + score, per iteration)

Run AFTER this job: python scripts/build_split_v1.py  (writes the final split_v1.json).

Feature cache: outputs/eval/split_v1_audit/dino_cache — deliberately NOT the
shared harness cache (outputs/eval/cache).

Requires PYTHONPATH to include the certified worktree src
(.claude/worktrees/eval-v3-spec/src) BEFORE this repo's src.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_split_v1 import CORPUS, STD, build_split  # noqa: E402

from diffusion.transition_eval.features import DinoExtractor, video_features  # noqa: E402
from diffusion.transition_eval.m2_integrity import copy_score  # noqa: E402

TAU = 0.858  # certified tau_copy (v3.0.0-amendment-1)
CACHE = REPO_ROOT / "outputs/eval/split_v1_audit/dino_cache"
RESULTS = REPO_ROOT / "outputs/eval/split_v1_audit/audit_results.json"
FLAGGED_OUT = STD / "split_v1_flagged.json"

MAX_ITERS = 12


def main() -> None:
    corpus = json.loads(CORPUS.read_text())
    extractor = DinoExtractor()
    feats_cache: dict[str, np.ndarray] = {}

    def feats(rel: str) -> np.ndarray:
        if rel not in feats_cache:
            f, _fps = video_features(REPO_ROOT / rel, CACHE, extractor)
            feats_cache[rel] = f
        return feats_cache[rel]

    flagged: dict[str, list[str]] = {}
    scanned: set[tuple[str, str, str]] = set()  # (class, test, train) already scored
    iterations = []

    for it in range(1, MAX_ITERS + 1):
        split = build_split(corpus, flagged)
        pairs_this_iter = []
        new_flags: dict[str, set[str]] = {}
        for cls, v in split.items():
            if not v["test"]:
                continue
            for t in v["test"]:
                for tr in v["train"]:
                    key = (cls, t, tr)
                    if key in scanned:
                        continue
                    scanned.add(key)
                    ft, ftr = feats(v["paths"][t]), feats(v["paths"][tr])
                    r = copy_score(
                        ft, np.ones(len(ft), dtype=bool),          # all test frames
                        ftr, np.zeros(len(ftr), dtype=bool),       # all train frames (non-core)
                        tau=TAU,
                    )
                    flag = bool(r["copy_max"] >= TAU)
                    pairs_this_iter.append({
                        "class": cls, "test": t, "train": tr,
                        "copy_max": round(r["copy_max"], 4),
                        "test_frame": r["copy_gen_frame"],
                        "train_frame": r["copy_ref_frame"],
                        "flagged": flag,
                    })
                    if flag:
                        new_flags.setdefault(cls, set()).add(t)
        n_flagged = sum(len(s) for s in new_flags.values())
        print(f"[iter {it}] scored {len(pairs_this_iter)} new pairs, "
              f"{n_flagged} newly flagged test clips: "
              f"{ {c: sorted(s) for c, s in new_flags.items()} or '{}' }")
        iterations.append({"iteration": it, "pairs": pairs_this_iter,
                           "new_flags": {c: sorted(s) for c, s in new_flags.items()}})
        if not new_flags:
            break
        for cls, s in new_flags.items():
            flagged.setdefault(cls, [])
            flagged[cls] = sorted(set(flagged[cls]) | s)
    else:
        raise RuntimeError(f"audit did not converge in {MAX_ITERS} iterations")

    FLAGGED_OUT.write_text(json.dumps({
        "tau_copy": TAU,
        "written_by": "scripts/audit_split_v1.py",
        "date": datetime.date.today().isoformat(),
        "flagged": flagged,
    }, indent=2))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "tau_copy": TAU,
        "date": datetime.date.today().isoformat(),
        "n_pairs_scored": len(scanned),
        "n_iterations": len(iterations),
        "final_flagged": flagged,
        "iterations": iterations,
    }, indent=2))
    n_flag_pairs = sum(p["flagged"] for i in iterations for p in i["pairs"])
    print(f"[done] {len(scanned)} pairs scored over {len(iterations)} iteration(s); "
          f"{n_flag_pairs} flagged pairs; final flag set: {flagged or 'EMPTY'}")
    print(f"[done] -> {FLAGGED_OUT} ; {RESULTS}")
    print("[next] run: python scripts/build_split_v1.py   # writes final split_v1.json")


if __name__ == "__main__":
    main()
