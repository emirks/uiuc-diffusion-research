"""ladder2 — Amendment-1 Pass A: clip-level dominance diagnostic.

WHY THIS EXISTS. The pre-registered generalist claim was a donor-pool margin vs the base twin.
That statistic is invalid for every reference-bearing cell: without the adapter, LTX-2 treats the
in-context clip as CONTENT TO CONTINUE and reproduces the demo's own scene instead of the
conditioned endpoint. Copying a donor-class clip scores well against the donor pool, so base is
rewarded for ignoring the task and ic_gen is penalised for doing it. The harness's absolute
`near_copy` flag (tau 0.858, calibrated for verbatim frame copies) cannot catch scene-level
reproduction sitting around cos 0.5.

WHAT THIS MEASURES (per Amendment-1, locked before any recipient-pool score existed):

    ep_align  = mean over GEN MIDDLE frames of ( max over ENDPOINT-clip frames of cos )
    ref_align = mean over GEN MIDDLE frames of ( max over REFERENCE NON-CORE frames of cos )
    ref_dominated := ref_align > ep_align          (threshold 0, pre-committed)

MEAN-of-best-match, not max-of-max: max aggregation is exactly why a whole reproduced scene slides
under tau_copy. The mean asks "what is the middle of this video actually made of?".

Masks follow the certified harness verbatim: `mid_mask` for the generation (everything outside the
conditioned windows) and `~core_mask` for the reference (the demo's own scenes A/B). `copy_max` is
a certified metric and is NOT touched — these are new columns beside it.

    python experiments/ladder2/dominance.py --mode passA
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WT = REPO_ROOT / ".claude/worktrees/eval-v4-cert/src"
sys.path.insert(0, str(WT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import prompts  # noqa: E402
from diffusion.transition_eval.features import DinoExtractor, video_features  # noqa: E402
from diffusion.transition_eval.m2_integrity import mid_mask  # noqa: E402
from diffusion.transition_eval.morph import core_mask, morph_profile  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
GENS = REPO_ROOT / "outputs/videos/ladder2"
CACHE = REPO_ROOT / "outputs/eval/cache"
OUT = HERE / "dominance_passA.jsonl"
N_PREFIX, N_SUFFIX = 9, 8


def feats_of(path: Path, ext: DinoExtractor) -> np.ndarray:
    f, _ = video_features(path, CACHE, ext)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["passA"], default="passA")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in (HERE / "registry.jsonl").read_text().splitlines() if x.strip()]
    base_by_key = {r["input_key"]: r for r in rows if r["arm"] == "base"}
    # every reference-bearing row, BOTH arms (advisor: apply symmetrically, including to cells
    # whose result looked flattering — G-ref-control's +4.1pp came from the same mechanism)
    work = []
    for r in rows:
        if r["arm"] != "ic_gen" or not r.get("reference"):
            continue
        twin = base_by_key.get(r["input_key"])
        for arm_row, arm in ((r, "ic_gen"), (twin, "base")):
            if arm_row is None:
                continue
            for seed in (42, 43):
                p = GENS / arm_row["arm"] / f"{arm_row['item_id']}__s{seed}.mp4"
                if p.exists():
                    work.append((r, arm, seed, p))
    if args.limit:
        work = work[: args.limit]
    print(f"[passA] {len(work)} (generation, arm) rows to measure")

    ext = DinoExtractor()
    out, n_err = [], 0
    for i, (row, arm, seed, gen_path) in enumerate(work, 1):
        try:
            ep_clip = STD / prompts.clip_class(row["endpoint"]) / f"{row['endpoint']}.mp4"
            ref_clip = STD / prompts.clip_class(row["reference"]) / f"{row['reference']}.mp4"
            g = feats_of(gen_path, ext)
            e = feats_of(ep_clip, ext)
            f = feats_of(ref_clip, ext)
            n_suf = N_SUFFIX if row["sided"] == "two" else 0
            gm = mid_mask(len(g), N_PREFIX, n_suf)
            rprof = morph_profile(f, n_prefix=N_PREFIX,
                                  n_suffix=N_SUFFIX if row["sided"] == "two" else N_SUFFIX,
                                  n_endpoints=2 if row["sided"] == "two" else 1)
            rnon = ~core_mask(rprof)
            gi = np.flatnonzero(gm)
            if len(gi) == 0 or not rnon.any():
                continue
            ep_align = float((g[gi] @ e.T).max(axis=1).mean())
            ref_align = float((g[gi] @ f[np.flatnonzero(rnon)].T).max(axis=1).mean())
            out.append({"item_id": row["item_id"], "cell": row["cell"], "arm": arm, "seed": seed,
                        "donor": row["donor_class"], "endpoint_class": row["endpoint_class"],
                        "ep_align": ep_align, "ref_align": ref_align,
                        "ref_dominated": bool(ref_align > ep_align)})
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                print(f"  [err] {gen_path.name}: {type(exc).__name__}: {exc}")
        if i % 50 == 0:
            print(f"  {i}/{len(work)}")

    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"[passA] wrote {len(out)} rows -> {OUT.relative_to(REPO_ROOT)} ({n_err} errors)")

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for o in out:
        agg[(o["cell"], o["arm"])]["ep"].append(o["ep_align"])
        agg[(o["cell"], o["arm"])]["ref"].append(o["ref_align"])
        agg[(o["cell"], o["arm"])]["dom"].append(o["ref_dominated"])
    print(f"\n{'cell':18s} {'arm':7s} {'n':>4s} {'ep_align':>9s} {'ref_align':>10s} {'ref_dominated':>14s}")
    print("-" * 68)
    for (cell, arm) in sorted(agg):
        d = agg[(cell, arm)]
        rate = sum(d["dom"]) / len(d["dom"])
        print(f"{cell:18s} {arm:7s} {len(d['ep']):4d} {st.mean(d['ep']):9.3f} "
              f"{st.mean(d['ref']):10.3f} {rate:13.0%}")
    print("\nref_dominated = the generation's middle frames resemble the DEMO more than the ENDPOINT")
    print("Advisor gate: if base's rate is below ~50%, the leakage story is weaker than believed")
    print("             -> re-consult BEFORE publishing the amendment's interpretation.")


if __name__ == "__main__":
    main()
