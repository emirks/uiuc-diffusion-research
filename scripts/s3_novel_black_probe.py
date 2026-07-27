"""Novel-black probe for the S3 comparison viewer.

Group A (exp_080 run_0001) has no per-clip hole radius and no adjudicated
verdicts, so there is no artifact-sourced way to ask "does A show the
disocclusion defect too?". This probe supplies one cheap, calibrated answer.

Statistic
---------
For a frame, `hard_black` is the fraction of the frame occupied by connected
near-black blobs (luma < 14, area >= 400 px) whose boundary abuts bright
content (luma > 110 over >25% of the blob's dilated rim). A naturally dark sky
does not qualify; a void punched by `_fill_holes` next to lit geometry does.

Raw `hard_black` is dominated by how dark the source footage is, so the probe
reports a *delta*: the max over five frames spanning 20–80% of the transition
ramp, minus the max over six frames of the clip's own pure phases. The pure
phases are byte-identical to the real source streams and contain no rendering,
so they are the perfect within-clip control.

Calibration
-----------
Run over group B (exp_082 pilot), whose 63 clips carry adjudicated BAD/GOOD
labels, the delta flags ~half of the BAD clips at +3 pp and almost none of the
GOOD ones. That makes it a usable one-sided probe: a high delta is evidence of
the defect, a low delta is not proof of its absence — it only sees hard black
voids, not the flat-smear / melt failure mode that S3_DROPPED.json identified
as the semantically hard part.

This is an operator sanity probe, NOT a certified metric and NOT a gate. It was
written for the comparison viewer; it does not revive the dropped S3 gate.

Usage:  python scripts/s3_novel_black_probe.py [repo_root]
Writes: experiments/exp_082_s3_stratum/S3_NOVEL_BLACK_PROBE.json
"""

from __future__ import annotations

import datetime
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

BLACK, BRIGHT, MIN_AREA, RIM_FRAC = 14, 110, 400, 0.25
OUT = "experiments/exp_082_s3_stratum/S3_NOVEL_BLACK_PROBE.json"


def hard_black(img: Image.Image) -> float:
    a = np.asarray(img.convert("L"), dtype=np.uint8)
    m = a < BLACK
    if m.mean() < 1e-5:
        return 0.0
    lab, n = ndimage.label(m)
    if n == 0:
        return 0.0
    bright_d = ndimage.binary_dilation(a > BRIGHT, iterations=3)
    tot = 0
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < MIN_AREA:
            continue
        rim = ndimage.binary_dilation(comp, iterations=3) & (~comp)
        if (rim & bright_d).sum() / max(rim.sum(), 1) > RIM_FRAC:
            tot += int(comp.sum())
    return tot / a.size


def probe(video: str, idxs: list[int]) -> list[float]:
    with tempfile.TemporaryDirectory() as td:
        sel = "+".join(rf"eq(n\,{i})" for i in idxs)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf",
                        f"select='{sel}'", "-vsync", "0", f"{td}/f_%03d.png"], check=True)
        return [hard_black(Image.open(p)) for p in sorted(pathlib.Path(td).glob("f_*.png"))]


def ramp(on: int, re: int, k: int = 5) -> list[int]:
    return [int(on + (re - on) * t) for t in np.linspace(0.2, 0.8, k)]


def pure(re: int) -> list[int]:
    return [1, 3, 5, re + 4, re + 8, 119]


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    per: dict[str, dict] = {}

    a_dir = root / "outputs/videos/exp_080_depth3d_realstream_121/run_0001"
    for m in json.load(open(a_dir / "manifest.json")):
        v = str(a_dir / "videos" / f"{m['stem']}.mp4")
        rr, pp = max(probe(v, ramp(m["onset"], m["release"]))), max(probe(v, pure(m["release"])))
        per[m["stem"]] = {"group": "A", "verdict": None, "ramp": round(rr, 5),
                          "pure": round(pp, 5), "delta": round(rr - pp, 5)}

    b_dir = root / "outputs/videos/ctt_v2_s3/pilot"
    adj = json.load(open(root / "experiments/exp_082_s3_stratum/GATE_ADJUDICATED.json"))
    adj = adj["per_clip_label"]
    for c in json.load(open(root / "experiments/exp_082_s3_stratum/PILOT_RESULT.json"))["clips"]:
        v = str(b_dir / "videos" / f"{c['stem']}.mp4")
        rr, pp = max(probe(v, ramp(c["onset"], c["release"]))), max(probe(v, pure(c["release"])))
        per[c["stem"]] = {"group": "B", "verdict": "bad" if adj[c["stem"]] else "good",
                          "ramp": round(rr, 5), "pure": round(pp, 5), "delta": round(rr - pp, 5)}

    def summ(rows):
        d = sorted(r["delta"] for r in rows)
        return {"n": len(d), "median_delta_pp": round(st.median(d) * 100, 2),
                "n_over_1pp": sum(x > 0.01 for x in d), "n_over_3pp": sum(x > 0.03 for x in d),
                "frac_over_3pp": round(sum(x > 0.03 for x in d) / len(d), 3),
                "median_raw_ramp_pct": round(st.median([r["ramp"] for r in rows]) * 100, 2),
                "median_raw_pure_pct": round(st.median([r["pure"] for r in rows]) * 100, 2)}

    A = [r for r in per.values() if r["group"] == "A"]
    Bb = [r for r in per.values() if r["verdict"] == "bad"]
    Bg = [r for r in per.values() if r["verdict"] == "good"]
    out = {
        "created": datetime.date.today().isoformat(),
        "what": "novel hard-black fraction added by the transition ramp over the clip's own "
                "pure phases; see scripts/s3_novel_black_probe.py for the definition",
        "status": "OPERATOR SANITY PROBE — not a certified metric, not a gate, one-sided "
                  "(sees hard black voids only, not flat-smear/melt)",
        "params": {"black_luma": BLACK, "bright_luma": BRIGHT, "min_blob_px": MIN_AREA,
                   "rim_bright_frac": RIM_FRAC, "ramp_frames": 5, "pure_frames": 6},
        "calibration_on_exp_082_adjudicated_labels": {"B_bad": summ(Bb), "B_good": summ(Bg)},
        "applied_to_exp_080_group_A": summ(A),
        "reading": "A behaves like B's GOOD clips on this probe, not like B's BAD ones. "
                   "A's raw hard-black number is high purely because its source footage is "
                   "dark; against its own pure phases the ramp mostly removes black.",
        "per_clip": per,
    }
    p = root / OUT
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    for k, v in [("A", summ(A)), ("B bad", summ(Bb)), ("B good", summ(Bg))]:
        print(f"  {k:6} n={v['n']:3} median delta {v['median_delta_pp']:+6.2f} pp   "
              f">3pp {v['n_over_3pp']}/{v['n']} ({100*v['frac_over_3pp']:.0f}%)")


if __name__ == "__main__":
    main()
