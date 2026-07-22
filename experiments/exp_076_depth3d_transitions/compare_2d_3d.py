"""Measure the Parallax Index on the exp_075 (2D shader) clips and the exp_076
(2.5D depth) clips, and on the real human-made transitions.

The point: a 2D operator has no notion of depth, so near and far pixels move by
the same amount and PI collapses to ~1 by construction. If the depth bank does
not separate cleanly from the shader bank on this measure, it is not actually
delivering 3D and the whole premise of exp_076 fails.

Usage: python compare_2d_3d.py <exp075_run_dir> <exp076_run_dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine3d import depth, metrics, videoio  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "outputs/analysis/exp_076_depth_cache"


def summarise(name: str, rows: list[dict]) -> dict:
    pis = np.array([r["pi"] for r in rows])
    rhos = np.array([r["rho"] for r in rows])
    out = {
        "set": name, "n": len(rows),
        "pi_median": round(float(np.median(pis)), 3),
        "pi_p10": round(float(np.percentile(pis, 10)), 3),
        "pi_p90": round(float(np.percentile(pis, 90)), 3),
        "rho_median": round(float(np.median(rhos)), 3),
        "frac_pi_gt_1_3": round(float((pis > 1.3).mean()), 3),
    }
    print(f"{name:28s} n={out['n']:3d}  PI median {out['pi_median']:5.2f} "
          f"[p10 {out['pi_p10']:5.2f}, p90 {out['pi_p90']:5.2f}]  "
          f"rho {out['rho_median']:+.2f}  frac(PI>1.3)={out['frac_pi_gt_1_3']:.2f}")
    return out


def depth_for(clip: np.ndarray, key: str) -> np.ndarray:
    """Depth of the frame the transition starts from, cached where possible."""
    npy = CACHE / f"{key}.npy"
    d = np.load(npy) if npy.exists() else depth.disparity(clip[0][None])[0]
    return depth.to_view_depth(d, 1.0, 3.5)


def main() -> None:
    d75, d76 = (pathlib.Path(a).resolve() for a in sys.argv[1:3])
    results = {}

    # -- exp_076: 2.5D depth transitions (PI is recorded during the run) ----
    m76 = json.load(open(d76 / "manifest.json"))
    results["depth3d"] = summarise("exp_076 depth-mesh 3D",
                                   [m["parallax"] for m in m76])

    # -- exp_076 by camera family ------------------------------------------
    fams: dict[str, list] = {}
    for m in m76:
        fams.setdefault(m["family"], []).append(m["parallax"])
    for fam in sorted(fams):
        results[f"depth3d_{fam}"] = summarise(f"  family: {fam}", fams[fam])

    # -- exp_075: 2D shader transitions ------------------------------------
    m75 = json.load(open(d75 / "manifest.json"))
    rows = []
    for m in [x for x in m75 if x["tag"] != "real"][:40]:
        clip = videoio.read_clip(d75 / "videos" / f"{m['stem']}.mp4")
        mid = len(clip) // 2
        seg = clip[mid: mid + 6]
        rows.append(metrics.parallax_index(seg, depth_for(seg, f"_75_{m['stem']}")))
    results["shader2d"] = summarise("exp_075 gl-transitions 2D", rows)

    # -- the real human-made transitions -----------------------------------
    real = [x for x in m75 if x["tag"] == "real"]
    if real:
        rows = []
        for m in real:
            clip = videoio.read_clip(d75 / "videos" / f"{m['stem']}.mp4")
            mid = len(clip) // 2
            seg = clip[mid: mid + 6]
            rows.append(metrics.parallax_index(seg, depth_for(seg, f"_r_{m['stem']}")))
        results["real"] = summarise("real human transitions", rows)

    out = REPO_ROOT / "outputs/analysis/exp_076_parallax_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
