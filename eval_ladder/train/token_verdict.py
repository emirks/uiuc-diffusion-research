"""ladder2 — decide the transition-slot token from the probe (pre-registered rule).

The token holds the transition slot in every prompt. It must be INERT on the base model:
if the base model already does transition-ish work when it sees the token, then the
`text_floor` arm stops being a leak-proof and every arm's causal story gets muddier.

Pre-registered decision rule (fixed before any probe output was looked at):

    noise  = d( notok@s42 , notok@s43 )          # same prompt, different seed
    effect = d( notok@s42 , TOKEN@s42 )          # same seed, token added
    TOKEN is INERT  iff  median over clips of (effect / noise)  <= 1.0

i.e. adding the token must move the base model no more than simply changing the seed does.
`sksz` is the default; `qvtr` the fallback; if both fail, stop and re-consult the advisor.

Distance d = mean absolute difference over decoded frames (both clips share the conditioning
and the resolution, so a plain pixel distance is the honest measure of "did the output move").

    python eval_ladder/train/token_verdict.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROBE = REPO_ROOT / "outputs/videos/ladder2_token_probe/probe"
ARMS = HERE.parent / "arms.yaml"
VERDICT = HERE / "token_verdict.json"

CANDIDATES = ["sksz", "qvtr"]
SEEDS = (42, 43)


def read_frames(path: Path):
    import numpy as np

    try:
        import imageio.v3 as iio

        return np.asarray(iio.imread(path, plugin="pyav"), dtype=np.float32)
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[:, :, ::-1])
        cap.release()
        return np.asarray(frames, dtype=np.float32)


def distance(a: Path, b: Path) -> float:
    import numpy as np

    fa, fb = read_frames(a), read_frames(b)
    n = min(len(fa), len(fb))
    return float(np.abs(fa[:n] - fb[:n]).mean() / 255.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the winning token into arms.yaml")
    args = ap.parse_args()

    clips = sorted({p.name.split("__")[0] for p in PROBE.glob("*__notok__s42.mp4")})
    if not clips:
        sys.exit(f"[token] no probe outputs yet under {PROBE}")

    report = {"rule": "effect/noise <= 1.0 (median over clips)", "clips": clips, "candidates": {}}
    winner = None
    for token in CANDIDATES:
        ratios = []
        for clip in clips:
            paths = {
                "n42": PROBE / f"{clip}__notok__s42.mp4",
                "n43": PROBE / f"{clip}__notok__s43.mp4",
                "t42": PROBE / f"{clip}__{token}__s42.mp4",
            }
            if not all(p.exists() for p in paths.values()):
                continue
            noise = distance(paths["n42"], paths["n43"])
            effect = distance(paths["n42"], paths["t42"])
            ratios.append(effect / noise if noise > 0 else float("inf"))
            print(f"  {clip:22s} {token}: noise={noise:.4f} effect={effect:.4f} "
                  f"ratio={ratios[-1]:.2f}")
        if not ratios:
            print(f"[token] {token}: no complete triples yet")
            continue
        med = st.median(ratios)
        inert = med <= 1.0
        report["candidates"][token] = {"n": len(ratios), "median_ratio": med, "inert": inert,
                                       "ratios": ratios}
        print(f"[token] {token}: median effect/noise = {med:.2f} over {len(ratios)} clips "
              f"-> {'INERT' if inert else 'NOT INERT'}")
        if inert and winner is None:
            winner = token

    report["winner"] = winner
    VERDICT.write_text(json.dumps(report, indent=2))
    if winner is None:
        print("[token] NO CANDIDATE PASSED — stop and re-consult the advisor before training")
        sys.exit(3)
    print(f"[token] VERDICT: {winner}")
    if args.apply:
        text = ARMS.read_text()
        import re

        new = re.sub(r"^token: \S+", f"token: {winner}", text, count=1, flags=re.M)
        ARMS.write_text(new)
        print(f"[token] arms.yaml token set to {winner!r}")


if __name__ == "__main__":
    main()
