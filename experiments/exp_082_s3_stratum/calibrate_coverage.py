"""Calibrate + FREEZE the S3 coverage threshold on the 63 hand-labelled pilot clips.

Advisor's pre-committed instrument bar:
    ">=20/23 BAD caught at <=3/40 GOOD falsely flagged, threshold frozen on these labels BEFORE
     the re-pilot renders. The re-pilot then serves as its out-of-sample test — same discipline
     as every frozen policy in this campaign."

If no threshold on the declared grid meets that bar, the instrument ESCAPES and I report that
rather than shipping a detector that does not detect — exactly as the dsx degenerate-frame gate
was killed by its own pre-committed bar.

    python calibrate_coverage.py
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

import coverage as cov  # noqa: E402
from engine3d import videoio  # noqa: E402

MIN_RECALL = 20      # of 23 BAD
MAX_FP = 3           # of 40 GOOD


def main() -> None:
    audit = json.loads((HERE / "PILOT_VISUAL_AUDIT.json").read_text())
    pilot = json.loads((HERE / "PILOT_RESULT.json").read_text())
    by_stem = {c["stem"]: c for c in pilot["clips"]}
    pool = json.loads((REPO_ROOT / "data/processed/ctt_v2_strata/CONTENT_POOL.json").read_text())
    # the pilot ran before the letterbox exclusion, so resolve paths from the FULL v1 bank
    v1 = REPO_ROOT / "data/processed/synth_endpoints"
    bank = {c["clip_id"]: str((v1 / c["mp4"]).resolve())
            for c in json.loads((v1 / "bank_tightened.json").read_text())["clips"]}
    for e in pool["training"] + pool["reserved"]:
        bank.setdefault(e["clip_id"], e["mp4"])

    vid_dir = REPO_ROOT / "outputs/videos/ctt_v2_s3/pilot/videos"
    rows, t0 = [], time.time()
    for n, a in enumerate(audit, 1):
        c = by_stem[a["stem"]]
        clip = videoio.read_clip(vid_dir / f"{a['stem']}.mp4")
        A = videoio.read_clip(Path(bank[c["A"]]))[:len(clip)]
        B = videoio.read_clip(Path(bank[c["B"]]))[:len(clip)]
        m = cov.coverage(clip, A, B, c["onset"], c["release"])
        rows.append({"stem": a["stem"], "bad": a["bad"], "family": a["family"],
                     "tag": a["tag"], **{k: v for k, v in m.items() if k != "per_frame"}})
        if n % 10 == 0:
            print(f"[cal] {n}/{len(audit)}  ({time.time()-t0:.0f}s)", flush=True)

    bad = np.array([r["bad"] for r in rows], bool)
    print(f"\n[cal] labels: {bad.sum()} BAD / {(~bad).sum()} GOOD")

    grid = [round(x, 3) for x in np.arange(0.02, 0.45, 0.005)]
    results = []
    for stat in ("unexplained_p95", "unexplained_max", "unexplained_mean"):
        v = np.array([r[stat] for r in rows])
        print(f"\n[cal] {stat}:  BAD median {np.median(v[bad]):.3f}  "
              f"GOOD median {np.median(v[~bad]):.3f}  "
              f"BAD p10 {np.percentile(v[bad],10):.3f}  GOOD p90 {np.percentile(v[~bad],90):.3f}")
        for thr in grid:
            flag = v > thr
            rec = int((flag & bad).sum())
            fp = int((flag & ~bad).sum())
            results.append({"stat": stat, "thr": thr, "recall": rec, "fp": fp,
                            "meets_bar": rec >= MIN_RECALL and fp <= MAX_FP})

    ok = [r for r in results if r["meets_bar"]]
    chosen = None
    if ok:
        # among configs meeting the bar, prefer max recall, then fewest false positives, then
        # the most permissive threshold (largest thr) so the gate is as gentle as it can be
        chosen = sorted(ok, key=lambda r: (-r["recall"], r["fp"], -r["thr"]))[0]
        print(f"\n[cal] BAR MET: {chosen['stat']} > {chosen['thr']} -> "
              f"recall {chosen['recall']}/23, FP {chosen['fp']}/40")
    else:
        best = sorted(results, key=lambda r: (-(r["recall"] - 3 * r["fp"])))[0]
        print(f"\n[cal] *** ESCAPE *** no config on the grid meets >={MIN_RECALL}/23 recall at "
              f"<={MAX_FP}/40 FP. Best reachable: {best['stat']} > {best['thr']} -> "
              f"recall {best['recall']}/23, FP {best['fp']}/40")

    out = {
        "created": "2026-07-25",
        "authority": "fable-advisor: pre-committed instrument bar >=20/23 recall at <=3/40 FP",
        "labels": {"n_bad": int(bad.sum()), "n_good": int((~bad).sum()),
                   "source": "operator blind visual audit of the 63-clip pilot"},
        "bar": {"min_recall_of_23": MIN_RECALL, "max_fp_of_40": MAX_FP},
        "metric_params": {"GRAY_HW": cov.GRAY_HW, "PATCH": cov.PATCH, "SEARCH": cov.SEARCH,
                          "FLAT_STD": cov.FLAT_STD, "NCC_OK": cov.NCC_OK,
                          "LUMA_OK": cov.LUMA_OK, "N_RAMP_SAMPLES": cov.N_RAMP_SAMPLES},
        "verdict": "FROZEN" if chosen else "ESCAPE",
        "frozen": chosen,
        "grid": results,
        "per_clip": rows,
    }
    (HERE / "COVERAGE_CALIB.json").write_text(json.dumps(out, indent=1))
    print(f"[cal] -> {HERE / 'COVERAGE_CALIB.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
