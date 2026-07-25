"""The ONE permitted new S3 statistic: hole-SALIENCE overlap. Definition frozen before any value
is computed.

Advisor ruling: "The permitted addition is the one your own diagnosis (a) predicts: a hole-
salience overlap — unfillable-hole/weak-coverage mass intersected with the subject/saliency
region, max or p95 over ramp frames. Define it precisely and freeze the definition BEFORE
computing a single value."

WHY. Ten quantity statistics all failed, and the reason is in their own numbers: a clip judged
GOOD typically has 12% of ramp pixels uncovered with a 76-px-radius largest hole. Quantity is
not the discriminator. The hypothesis this tests is that *location* is: a hole over a plain wall
is invisible after inpainting, the same hole over a face is not.

DEFINITION (frozen). For each sampled ramp frame:
  * `den` is the compositor's total available alpha; hole = den < COVER_HARD, weak = den < COVER_WEAK.
  * SALIENCE is the spectral-residual saliency (numpy; this OpenCV build has no cv2.saliency) of the COMPOSITED frame, normalised to
    [0,1]. Spectral residual is used because it is the same family the endpoint funnel already
    relies on for its compact-foreground fallback, it needs no model download, and it is
    computed on the rendered frame so no source-space-to-render-space warping is involved.
  * salience_hole  = sum(salience * hole) / sum(salience)     — the share of the frame's total
                     visual interest that sits on invented pixels.
  * salience_weak  = same with the weak mask.
  * peak_hole      = max salience value anywhere inside the hole mask — "is any of this hole
                     somewhere the eye actually goes".
Clip statistic = max and p95 over sampled ramp frames. Six values, no free parameters beyond
the two coverage thresholds already frozen.

Scored against the SAME adjudicated labels and the SAME operating point (>=87% recall at <=7.5%
FP). No threshold tuning beyond the declared sweep. If it fails, S3 drops per the pre-committed
tree.
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

N_SAMPLES = 12


def main() -> None:
    cfg = load_config(HERE / "config_s3.yaml")
    inf = cfg["inference"]
    NF = inf["num_frames"]

    lab = json.loads((HERE / "GATE_ADJUDICATED.json").read_text())["per_clip_label"]
    pilot = json.loads((HERE / "PILOT_RESULT.json").read_text())
    by_stem = {c["stem"]: c for c in pilot["clips"]}
    v1 = REPO_ROOT / "data/processed/synth_endpoints"
    bank = {c["clip_id"]: (v1 / c["mp4"]).resolve()
            for c in json.loads((v1 / "bank_tightened.json").read_text())["clips"]}
    cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]

    renderer = MeshRenderer(inf["width"], inf["height"], step=inf["mesh_step"])
    print(f"[sal] GL: {renderer.renderer_name()}", flush=True)
    fr: dict = {}
    dp: dict = {}

    def frames(c):
        if c not in fr:
            fr[c] = videoio.read_clip(bank[c])[:NF]
            while len(fr) > 14:
                fr.pop(next(iter(fr)))
        return fr[c]

    def depth(c):
        if c not in dp:
            dp[c] = np.load(cache_dir / f"{c}.npy").astype(np.float32)
            while len(dp) > 14:
                dp.pop(next(iter(dp)))
        return dp[c]

    rows, t0 = [], time.time()
    for n, (stem, y) in enumerate(sorted(lab.items()), 1):
        c = by_stem[stem]
        params = dict(c["params"])
        params["fog_color"] = tuple(params["fog_color"])
        op = ops3d.Operator3D(**params)
        cov: list = []
        clip = ops3d.render_transition_stream(
            renderer, op, frames(c["A"]), frames(c["B"]), depth(c["A"]), depth(c["B"]),
            c["onset"], c["release"], coverage_out=cov)
        # re-derive the masks on the sampled ramp frames alongside their salience
        lo, hi = c["onset"] + 1, c["release"]
        idx = np.unique(np.linspace(lo, hi - 1, min(N_SAMPLES, hi - lo)).astype(int))
        tmap = {x["t"]: x for x in cov}
        sh = [tmap[t]["sal_hole"] for t in idx if t in tmap]
        sw = [tmap[t]["sal_weak"] for t in idx if t in tmap]
        ph = [tmap[t]["sal_peak"] for t in idx if t in tmap]
        rows.append({"stem": stem, "bad": bool(y),
                     "sal_hole_max": float(np.nanmax(sh)) if sh else 0.0,
                     "sal_hole_p95": float(np.nanpercentile(sh, 95)) if sh else 0.0,
                     "sal_weak_max": float(np.nanmax(sw)) if sw else 0.0,
                     "sal_weak_p95": float(np.nanpercentile(sw, 95)) if sw else 0.0,
                     "sal_peak_max": float(np.nanmax(ph)) if ph else 0.0,
                     "sal_peak_p95": float(np.nanpercentile(ph, 95)) if ph else 0.0})
        if n % 10 == 0:
            print(f"[sal] {n}/{len(lab)} ({time.time()-t0:.0f}s)", flush=True)

    y = np.array([r["bad"] for r in rows], bool)
    MIN_REC = int(np.ceil(0.87 * y.sum()))
    MAX_FP = int(np.floor(0.075 * (~y).sum()))
    cands = ["sal_hole_max", "sal_hole_p95", "sal_weak_max", "sal_weak_p95",
             "sal_peak_max", "sal_peak_p95"]
    print(f"\n[sal] {y.sum()} BAD / {(~y).sum()} GOOD | bar >={MIN_REC} recall at <={MAX_FP} FP")
    print(f"\n{'STAT':<14} {'AUC':>6} {'BADmed':>9} {'GOODmed':>9} {'recall@bar':>12} {'thr':>9}")
    res, passing = {}, []
    for k in cands:
        v = np.array([r[k] for r in rows])
        o = np.argsort(-v)
        rk = np.empty(len(v))
        rk[o] = np.arange(len(v))
        auc = (rk[~y].sum() - (~y).sum() * ((~y).sum() - 1) / 2) / (y.sum() * (~y).sum())
        auc = 1 - auc if auc < 0.5 else auc
        b = {"recall": 0, "fp": 0, "thr": None}
        for t in np.unique(v):
            f = v > t
            fp = int((f & ~y).sum())
            if fp <= MAX_FP and int((f & y).sum()) > b["recall"]:
                b = {"recall": int((f & y).sum()), "fp": fp, "thr": float(t)}
        res[k] = {"auc": round(auc, 3), **b}
        if b["recall"] >= MIN_REC:
            passing.append(k)
        print(f"{k:<14} {auc:>6.3f} {np.median(v[y]):>9.4f} {np.median(v[~y]):>9.4f} "
              f"{b['recall']:>8d}/{y.sum():<3d} "
              f"{(round(b['thr'],4) if b['thr'] is not None else '-')!s:>9}"
              f"{'  <== MEETS BAR' if b['recall'] >= MIN_REC else ''}")
    print(f"\n[sal] VERDICT: {'PASS ' + str(passing) if passing else 'FAIL'}")
    (HERE / "GATE_SALIENCE.json").write_text(json.dumps(
        {"created": "2026-07-25", "definition_frozen_before_computation": True,
         "bar": {"min_recall": MIN_REC, "max_fp": MAX_FP},
         "verdict": "PASS" if passing else "FAIL", "passing": passing,
         "results": res, "per_clip": rows}, indent=1))
    print(f"[sal] -> {HERE / 'GATE_SALIENCE.json'}")


if __name__ == "__main__":
    main()
