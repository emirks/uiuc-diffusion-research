"""Blind visual audit: build anonymised contact sheets, then score the grades.

Judging one's own renders is the weak point of this experiment, so the grading
is made as hard to fudge as it cheaply can be:

  * the sample is drawn at random from a fixed seed, before anything is looked
    at, and every clip drawn is graded — no dropping;
  * each clip becomes a 3-frame strip at progress ~= 0.3 / 0.5 / 0.7 labelled
    with an opaque id (`c07`), shuffled across arms, so the sheet carries no
    arm, no map name and no compositor;
  * the id -> (arm, map) key is written to a separate file that is only joined
    in the `score` step, after the grades exist.

    python sample_audit.py sheets [run_dir]     # writes audit_sheets/ + KEY.json
    python sample_audit.py score  [run_dir]     # joins GRADES.json with KEY.json

GRADES.json format: {"c07": "BAD", "c08": "OK", ...}
Pre-registered rubric (fixed before any sheet was looked at):
  BAD = reads as a digital artefact — a hard alpha cut with no material at the
        boundary, a visible synthetic primitive (stripe/checker/spiral/perfect
        ring), an undirected crossfade, or speckle dirt along the front.
  OK  = reads as a physical material event (ink, paint, burn, frost, leak).
Borderline goes to BAD.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

import numpy as np
import PIL.Image
import PIL.ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mattes import clipio  # noqa: E402

N_PER_ARM = 16
SEED = 84084
TILE_W = 150
COLS = 3          # clips per sheet row
ROWS = 4


def latest_run() -> pathlib.Path:
    return sorted((REPO_ROOT / "outputs" / "videos"
                   / "exp_084_luma_matte_viewer").glob("run_*"))[-1]


def progress_frames(total: int, k: int) -> list[int]:
    sys.path.insert(0, str(REPO_ROOT / "experiments"
                           / "exp_075_procedural_transition_engine"))
    from engine import streams
    p = streams.progress_ramp(total, k, k, "smoothstep")
    return [int(np.argmin(np.abs(p - q))) for q in (0.30, 0.50, 0.70)]


def build_sheets(run_dir: pathlib.Path) -> None:
    man = json.load(open(run_dir / "manifest.json"))
    rng = random.Random(SEED)
    by_arm: dict[str, list[dict]] = {}
    for m in man:
        by_arm.setdefault(m["arm"], []).append(m)

    picked: list[dict] = []
    for arm in sorted(by_arm):
        rows = sorted(by_arm[arm], key=lambda r: r["stem"])
        picked += rng.sample(rows, min(N_PER_ARM, len(rows)))
    rng.shuffle(picked)

    key = {}
    out_dir = run_dir / "audit_sheets"
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    tiles = []
    for i, rec in enumerate(picked):
        cid = f"c{i:02d}"
        key[cid] = {"stem": rec["stem"], "arm": rec["arm"], "map": rec["map"],
                    "compositor": rec["compositor_key"], "pair": rec["pair_id"]}
        clip = clipio.read_clip(run_dir / "videos" / f"{rec['stem']}.mp4")
        idx = progress_frames(len(clip), 6)
        h, w = clip.shape[1:3]
        tw, th = TILE_W, int(TILE_W * h / w)
        strip = PIL.Image.new("RGB", (3 * tw + 4, th + 16), (255, 255, 255))
        for j, fi in enumerate(idx):
            strip.paste(PIL.Image.fromarray(clip[fi]).resize((tw, th),
                                            PIL.Image.LANCZOS), (j * (tw + 2), 16))
        d = PIL.ImageDraw.Draw(strip)
        d.rectangle([0, 0, 3 * tw + 4, 15], fill=(20, 20, 20))
        d.text((4, 3), cid, fill=(255, 235, 120))
        tiles.append(strip)

    per = COLS * ROWS
    for s in range((len(tiles) + per - 1) // per):
        chunk = tiles[s * per:(s + 1) * per]
        tw, th = chunk[0].size
        sheet = PIL.Image.new("RGB", (COLS * (tw + 6), ROWS * (th + 6)), (245, 245, 245))
        for k, t in enumerate(chunk):
            sheet.paste(t, ((k % COLS) * (tw + 6) + 3, (k // COLS) * (th + 6) + 3))
        sheet.save(out_dir / f"sheet_{s:02d}.jpg", quality=90)

    json.dump(key, open(run_dir / "AUDIT_KEY.json", "w"), indent=1)
    print(f"{len(tiles)} clips -> {out_dir} ({len(list(out_dir.glob('*.jpg')))} sheets)")
    print("KEY written to AUDIT_KEY.json — do not open before grading")


def score(run_dir: pathlib.Path) -> None:
    key = json.load(open(run_dir / "AUDIT_KEY.json"))
    grades = json.load(open(run_dir / "GRADES.json"))
    missing = sorted(set(key) - set(grades))
    assert not missing, f"ungraded clips (no dropping allowed): {missing}"

    per_arm: dict[str, list[int]] = {}
    per_map: dict[tuple[str, str], list[int]] = {}
    for cid, g in grades.items():
        assert g in ("BAD", "OK"), f"{cid}: bad grade {g}"
        k = key[cid]
        bad = int(g == "BAD")
        per_arm.setdefault(k["arm"], []).append(bad)
        per_map.setdefault((k["arm"], k["map"]), []).append(bad)

    res = {"n_graded": len(grades), "rubric": "BAD = reads as a digital artefact",
           "per_arm": {}, "per_clip": {c: {**key[c], "grade": g}
                                       for c, g in sorted(grades.items())}}
    print(f"{'arm':22s} {'n':>3s} {'BAD':>4s} {'rate':>7s}")
    for arm in sorted(per_arm):
        v = per_arm[arm]
        res["per_arm"][arm] = {"n": len(v), "bad": int(sum(v)),
                               "bad_rate": round(float(np.mean(v)), 3)}
        print(f"{arm:22s} {len(v):3d} {sum(v):4d} {np.mean(v):6.1%}")
    json.dump(res, open(run_dir / "AUDIT_RESULT.json", "w"), indent=1)
    print(f"-> {run_dir / 'AUDIT_RESULT.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheets"
    rd = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else latest_run()
    (build_sheets if cmd == "sheets" else score)(rd)
