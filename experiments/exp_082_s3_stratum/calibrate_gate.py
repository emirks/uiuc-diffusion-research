"""Re-render the 63 pilot operators capturing GEOMETRIC COVERAGE, and freeze the S3 gate.

WHY THIS REPLACES THE IMAGE-SPACE ATTEMPT. The first coverage metric (coverage.py) asked, per
patch, "does this resemble either source stream?". It ESCAPED its pre-committed bar: best
reachable 11/23 recall at 1/40 FP. A targeted black-void statistic did better (AUC 0.832) but
still only 12/23 — because the 23 BAD clips are not one defect class. Inspecting the 11 it
missed showed tears and smears, not black.

Then reading the renderer settled it. `composite()` computes `den`, the total available alpha
per pixel, and hands it to `_fill_holes`, a push-pull inpainter. So there is ONE root cause with
two faces:

    den ~ 0 over a small region  -> inpainted plausibly            (fine)
    den ~ 0 over a medium region -> filled by a wide-radius blur    -> "smear / melt / tear"
    den ~ 0 over a large region  -> even the 81px blur cannot reach -> "black void"

Both symptoms I hand-labelled are the same failure: the camera looked past the edge of the
world, or stretched geometry so far that almost no real signal remains. `den` measures that
CAUSE exactly and geometrically — no image heuristic, no threshold on appearance — and it is
already computed inside the render at zero extra cost.

So the gate candidate is the fraction of ramp pixels with den below a coverage threshold. This
script recomputes it for the 63 already-labelled pilot operators (identical ops, identical
content pairs, identical timing, read back from PILOT_RESULT.json) and evaluates it against the
advisor's pre-committed instrument bar: **>=20/23 BAD caught at <=3/40 GOOD falsely flagged.**
If nothing clears the bar, the instrument ESCAPES and I report that.

    sbatch experiments/exp_082_s3_stratum/job_calgate.sbatch
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

from diffusion.exp_utils import load_config  # noqa: E402

from engine3d import ops3d, videoio  # noqa: E402
from engine3d.render3d import MeshRenderer  # noqa: E402

MIN_RECALL = 20      # of 23 BAD
MAX_FP = 3           # of 40 GOOD


def main() -> None:
    cfg = load_config(HERE / "config_s3.yaml")
    inf = cfg["inference"]
    NF = inf["num_frames"]

    audit = json.loads((HERE / "PILOT_VISUAL_AUDIT.json").read_text())
    pilot = json.loads((HERE / "PILOT_RESULT.json").read_text())
    by_stem = {c["stem"]: c for c in pilot["clips"]}

    v1 = REPO_ROOT / "data/processed/synth_endpoints"
    bank = {c["clip_id"]: (v1 / c["mp4"]).resolve()
            for c in json.loads((v1 / "bank_tightened.json").read_text())["clips"]}
    lb = set(json.loads(
        (REPO_ROOT / "data/processed/ctt_v2_strata/LETTERBOX_AUDIT.json").read_text())["excluded"])
    cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]

    renderer = MeshRenderer(inf["width"], inf["height"], step=inf["mesh_step"])
    print(f"[gate] GL: {renderer.renderer_name()}", flush=True)

    fr: dict = {}
    dp: dict = {}

    def frames(cid):
        if cid not in fr:
            fr[cid] = videoio.read_clip(bank[cid])[:NF]
            while len(fr) > 16:
                fr.pop(next(iter(fr)))
        return fr[cid]

    def depth(cid):
        if cid not in dp:
            dp[cid] = np.load(cache_dir / f"{cid}.npy").astype(np.float32)
            while len(dp) > 16:
                dp.pop(next(iter(dp)))
        return dp[cid]

    rows, t0 = [], time.time()
    for n, a in enumerate(audit, 1):
        c = by_stem[a["stem"]]
        params = dict(c["params"])
        params["fog_color"] = tuple(params["fog_color"])
        op = ops3d.Operator3D(**params)
        cov: list = []
        ops3d.render_transition_stream(
            renderer, op, frames(c["A"]), frames(c["B"]), depth(c["A"]), depth(c["B"]),
            c["onset"], c["release"], coverage_out=cov)
        unc = np.array([x["uncovered"] for x in cov])
        weak = np.array([x["weak"] for x in cov])
        hr = np.array([x["hole_radius"] for x in cov])
        wr = np.array([x["weak_radius"] for x in cov])
        rows.append({
            "stem": a["stem"], "bad": a["bad"], "family": a["family"], "tag": a["tag"],
            "clean_contents": (c["A"] not in lb) and (c["B"] not in lb),
            "unc_max": float(unc.max()), "unc_p95": float(np.percentile(unc, 95)),
            "unc_mean": float(unc.mean()),
            "weak_max": float(weak.max()), "weak_p95": float(np.percentile(weak, 95)),
            "weak_mean": float(weak.mean()),
            "hole_r_max": float(hr.max()), "hole_r_p95": float(np.percentile(hr, 95)),
            "weak_r_max": float(wr.max()), "weak_r_p95": float(np.percentile(wr, 95)),
        })
        if n % 10 == 0:
            print(f"[gate] {n}/{len(audit)} ({time.time()-t0:.0f}s)", flush=True)

    bad = np.array([r["bad"] for r in rows], bool)
    cands = ["unc_max", "unc_p95", "unc_mean", "weak_max", "weak_p95", "weak_mean",
             "hole_r_max", "hole_r_p95", "weak_r_max", "weak_r_p95"]

    def sweep(v, mask=None):
        m = np.ones(len(v), bool) if mask is None else mask
        vv, bb = v[m], bad[m]
        best = {"recall": 0, "fp": 0, "thr": None,
                "n_bad": int(bb.sum()), "n_good": int((~bb).sum())}
        for thr in np.unique(vv):
            f = vv > thr
            fp = int((f & ~bb).sum())
            if fp <= MAX_FP and int((f & bb).sum()) > best["recall"]:
                best = {**best, "recall": int((f & bb).sum()), "fp": fp, "thr": float(thr)}
        order = np.argsort(-vv)
        ranks = np.empty(len(vv))
        ranks[order] = np.arange(len(vv))
        nb, ng = int(bb.sum()), int((~bb).sum())
        auc = float((ranks[~bb].sum() - ng * (ng - 1) / 2) / max(nb * ng, 1))
        best["auc"] = round(1 - auc if auc < 0.5 else auc, 3)
        return best

    clean = np.array([r["clean_contents"] for r in rows], bool)
    print(f"\n[gate] {len(rows)} clips | {bad.sum()} BAD | "
          f"{clean.sum()} on clean (non-letterboxed) contents")
    print(f"\n{'STAT':<12} {'AUC':>6} {'BADmed':>9} {'GOODmed':>9} {'recall@<=3FP':>13} {'thr':>9}")
    res = {}
    for k in cands:
        v = np.array([r[k] for r in rows])
        s = sweep(v)
        res[k] = {"all": s, "clean_only": sweep(v, clean),
                  "bad_median": round(float(np.median(v[bad])), 5),
                  "good_median": round(float(np.median(v[~bad])), 5)}
        print(f"{k:<12} {s['auc']:>6.3f} {np.median(v[bad]):>9.5f} {np.median(v[~bad]):>9.5f} "
              f"{str(s['recall'])+'/'+str(int(bad.sum())):>13} "
              f"{(round(s['thr'],5) if s['thr'] is not None else '-')!s:>9}")

    ok = [(k, res[k]["all"]) for k in cands if res[k]["all"]["recall"] >= MIN_RECALL]
    chosen = None
    if ok:
        k, s = sorted(ok, key=lambda t: (-t[1]["recall"], t[1]["fp"], -(t[1]["thr"] or 0)))[0]
        chosen = {"stat": k, **s}
        print(f"\n[gate] BAR MET: {k} > {s['thr']:.5f} -> recall {s['recall']}/{int(bad.sum())}, "
              f"FP {s['fp']}/{int((~bad).sum())}")
    else:
        k, s = sorted(((k, res[k]["all"]) for k in cands),
                      key=lambda t: -(t[1]["recall"] - 3 * t[1]["fp"]))[0]
        print(f"\n[gate] *** ESCAPE *** best reachable {k} > "
              f"{s['thr'] if s['thr'] is None else round(s['thr'],5)} -> "
              f"recall {s['recall']}/{int(bad.sum())}, FP {s['fp']}")

    out = {"created": "2026-07-25",
           "authority": "fable-advisor pre-committed instrument bar: >=20/23 recall, <=3/40 FP",
           "signal": "geometric coverage — fraction of ramp pixels whose total composite alpha "
                     "(`den`) falls below a threshold, i.e. pixels no mesh actually covered and "
                     "`_fill_holes` had to invent",
           "thresholds": {"COVER_HARD": ops3d.COVER_HARD, "COVER_WEAK": ops3d.COVER_WEAK},
           "bar": {"min_recall": MIN_RECALL, "max_fp": MAX_FP},
           "verdict": "FROZEN" if chosen else "ESCAPE", "frozen": chosen,
           "separation": res, "per_clip": rows}
    (HERE / "GATE_CALIB.json").write_text(json.dumps(out, indent=1))
    print(f"[gate] -> {HERE / 'GATE_CALIB.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
