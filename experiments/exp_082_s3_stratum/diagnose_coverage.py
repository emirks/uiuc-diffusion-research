"""Why did the general coverage metric escape? Measure a SMALL set of principled alternatives.

The v1 metric asked, per patch, "does this resemble either source stream under local search?".
It escaped: best reachable 11/23 recall at 1/40 FP, BAD p95 median 0.199 vs GOOD 0.112 — heavy
overlap. The likely reason is a design flaw I can name: **legitimate optical effects also fail
to resemble either source.** Fog, rack focus, world-space dissolves and crossfades all transform
the frame, so a correlation-based "explained" test penalises them exactly like fabrication. That
is consistent with the escape: the metric fires on effect strength, not on defect presence.

The observed defect is much more specific than "doesn't resemble the sources": it is BLACK
CONTENT that appears where the mesh has left the frame. So this measures a small number of
targeted statistics alongside the general one, and reports separation for each.

DISCIPLINE NOTE, to carry into the advisor briefing: these are evaluated on the same 63 labels
that the v1 threshold was fitted on, so the comparison is in-sample and a winner here is a
HYPOTHESIS, not a validated instrument. The re-pilot is its out-of-sample test, exactly as the
advisor specified. The candidate set is deliberately small (4) and each is motivated by the
observed failure mode rather than swept.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402

from engine3d import videoio  # noqa: E402

cv2.setNumThreads(2)

BLACK = 24          # luma below this is "black"
DILATE = 9          # local window (px, full-res) within which a source black pixel excuses one
N_SAMPLES = 14


def stats_for(clip: np.ndarray, A: np.ndarray, B: np.ndarray,
              onset: int, release: int) -> dict:
    lo, hi = onset + 1, release
    if hi - lo < 2:
        return {}
    idx = np.unique(np.linspace(lo, hi - 1, min(N_SAMPLES, hi - lo)).astype(int))
    k = np.ones((DILATE, DILATE), np.uint8)

    novel, rblack, dblack = [], [], []
    for t in idx:
        gr = cv2.cvtColor(clip[t], cv2.COLOR_RGB2GRAY)
        ga = cv2.cvtColor(A[t], cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(B[t], cv2.COLOR_RGB2GRAY)
        r_b = gr < BLACK
        # a render-black pixel is EXCUSED if either source is black anywhere nearby: that is
        # real dark content that merely moved, not a hole in the mesh
        src_b = ((ga < BLACK) | (gb < BLACK)).astype(np.uint8)
        src_b = cv2.dilate(src_b, k).astype(bool)
        novel.append(float((r_b & ~src_b).mean()))
        rblack.append(float(r_b.mean()))
        dblack.append(float(r_b.mean() - max((ga < BLACK).mean(), (gb < BLACK).mean())))
    return {"novel_black_max": float(np.max(novel)),
            "novel_black_p95": float(np.percentile(novel, 95)),
            "novel_black_mean": float(np.mean(novel)),
            "render_black_max": float(np.max(rblack)),
            "delta_black_max": float(np.max(dblack))}


def sep(v: np.ndarray, bad: np.ndarray, max_fp: int = 3) -> dict:
    """Best recall achievable at <= max_fp false positives, plus AUC."""
    order = np.argsort(-v)
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    nb, ng = int(bad.sum()), int((~bad).sum())
    auc = float((ranks[~bad].sum() - ng * (ng - 1) / 2) / (nb * ng))
    auc = 1 - auc if auc < 0.5 else auc
    best = {"recall": 0, "fp": 0, "thr": None}
    for thr in np.unique(v):
        f = v > thr
        fp = int((f & ~bad).sum())
        if fp <= max_fp and int((f & bad).sum()) > best["recall"]:
            best = {"recall": int((f & bad).sum()), "fp": fp, "thr": float(thr)}
    return {"auc": round(auc, 3), **best}


def main() -> None:
    audit = json.loads((HERE / "PILOT_VISUAL_AUDIT.json").read_text())
    pilot = json.loads((HERE / "PILOT_RESULT.json").read_text())
    by_stem = {c["stem"]: c for c in pilot["clips"]}
    v1 = REPO_ROOT / "data/processed/synth_endpoints"
    bank = {c["clip_id"]: str((v1 / c["mp4"]).resolve())
            for c in json.loads((v1 / "bank_tightened.json").read_text())["clips"]}
    vid_dir = REPO_ROOT / "outputs/videos/ctt_v2_s3/pilot/videos"

    src: dict = {}

    def get(p: str, n: int):
        if p not in src:
            src[p] = videoio.read_clip(Path(p))
            while len(src) > 40:
                src.pop(next(iter(src)))
        return src[p][:n]

    rows, t0 = [], time.time()
    for n, a in enumerate(audit, 1):
        c = by_stem[a["stem"]]
        clip = videoio.read_clip(vid_dir / f"{a['stem']}.mp4")
        rows.append({"stem": a["stem"], "bad": a["bad"], "family": a["family"], "tag": a["tag"],
                     **stats_for(clip, get(bank[c["A"]], len(clip)),
                                 get(bank[c["B"]], len(clip)), c["onset"], c["release"])})
        if n % 15 == 0:
            print(f"[diag] {n}/{len(audit)} ({time.time()-t0:.0f}s)", flush=True)

    # fold in the v1 general metric for a like-for-like comparison
    cal = {r["stem"]: r for r in json.loads((HERE / "COVERAGE_CALIB.json").read_text())["per_clip"]}
    for r in rows:
        for k in ("unexplained_p95", "unexplained_max"):
            r[k] = cal[r["stem"]][k]

    bad = np.array([r["bad"] for r in rows], bool)
    cands = ["novel_black_max", "novel_black_p95", "novel_black_mean",
             "render_black_max", "delta_black_max", "unexplained_p95", "unexplained_max"]
    print(f"\n{'STATISTIC':<22} {'AUC':>6} {'BADmed':>8} {'GOODmed':>8} "
          f"{'recall@<=3FP':>13} {'thr':>8}")
    out = {}
    for k in cands:
        v = np.array([r[k] for r in rows])
        s = sep(v, bad)
        out[k] = {**s, "bad_median": round(float(np.median(v[bad])), 4),
                  "good_median": round(float(np.median(v[~bad])), 4)}
        print(f"{k:<22} {s['auc']:>6.3f} {np.median(v[bad]):>8.4f} {np.median(v[~bad]):>8.4f} "
              f"{str(s['recall'])+'/23':>13} {str(round(s['thr'],4)) if s['thr'] is not None else '-':>8}")

    json.dump({"created": "2026-07-25", "in_sample_caveat":
               "evaluated on the same 63 labels the v1 threshold was fitted on — a winner here "
               "is a HYPOTHESIS; the re-pilot is the out-of-sample test",
               "black_luma_thr": BLACK, "dilate_px": DILATE, "n_samples": N_SAMPLES,
               "separation": out, "per_clip": rows},
              open(HERE / "COVERAGE_DIAGNOSE.json", "w"), indent=1)
    print(f"\n[diag] -> {HERE / 'COVERAGE_DIAGNOSE.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
