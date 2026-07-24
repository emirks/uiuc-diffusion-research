"""exp_077 D2 Task 4 — the OWNER SHEET: 20 random GATE-PASSING audit clips as filmstrips.

Deliverable (the training run is blocked on the owner's OK):
    outputs/endpoint_candidates/d2_audit/owner_sheet.png
    outputs/endpoint_candidates/d2_audit/filmstrips/<stem>.png   (per-clip, larger)
    outputs/endpoint_candidates/d2_audit/videos/<stem>.mp4       (the same 20 clips)
    outputs/endpoint_candidates/d2_audit/owner_sheet_index.json

Frames are sampled across 0..120; the pinned endpoint blocks (0-8 from clip A, 112-120 from
clip B) are drawn with a red border and labelled, and each row is captioned with shader,
easing/flip/swap, timing (onset/release/duration) and the four gate scores.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

import d2_metrics  # noqa: E402
import d2_sheets  # noqa: E402
from calibrate_d2 import load_rows  # noqa: E402
from engine import videoio  # noqa: E402

CONFIG_PATH = HERE / "config_d2.yaml"
N_SHEET = 20
SEED = 7


def caption(r: dict, tau: float) -> list[str]:
    a1 = max(r["assert1"]["mae_pure_a"], r["assert1"]["mae_pure_b"])
    return [
        f"{r['stem']}   shader={r['shader']}   ease={r['easing']}  flip={r['flip']}  "
        f"swap={r['swap']}   aux=none  ext=none (REAL streams)   |   "
        f"onset={r['timing']['onset']:.1f}  release={r['timing']['release']:.1f}  "
        f"duration={r['timing']['duration']:.1f}",
        f"GATE  pure-phase MAE={a1:.4f}   seam={max(r['assert2']['seam_ratio']):.2f} (<=2.0)   "
        f"M1 p10={r['m1_p10']:.3f} (>={tau})   M2 max dq={r['m2_max_dq']:.3f} (<=0.5)"
        f"        A={r['A']}   B={r['B']}",
    ]


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    gate = cfg["gate"]
    tau = json.loads((HERE / "D2_TAU.json").read_text())["tau"]
    root = REPO_ROOT / cfg["outputs"]["dir"] / "audit"
    out = REPO_ROOT / cfg["outputs"]["sheet_dir"]
    (out / "filmstrips").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows(root / "meta")
    passing = [r for r in rows
               if d2_metrics.verdict(r, tau, assert1_tol=gate["assert1_tol"],
                                     seam_max=gate["seam_max"],
                                     m2_max=gate["m2_max_dq"])["pass"]]
    print(f"[owner] {len(passing)}/{len(rows)} clips pass the frozen gate (tau={tau})")
    rng = random.Random(SEED)
    pick = rng.sample(passing, min(N_SHEET, len(passing)))
    pick.sort(key=lambda r: r["stem"])

    blocks, index = [], []
    for r in pick:
        clip = videoio.read_clip(root / "videos" / f"{r['stem']}.mp4")
        T = len(clip)
        i0, j0 = r["phase"]["i0"], r["phase"]["j0"]
        marks = {0: "A anchor 0-8", 8: "A anchor 0-8", 112: "B anchor 112-120",
                 T - 1: "B anchor 112-120"}
        row_idx = [0, 8] + [int(round(i0 + (j0 - i0) * k / 10)) for k in range(1, 10)] \
            + [112, T - 1]
        row_idx = sorted(dict.fromkeys(min(T - 1, max(0, i)) for i in row_idx))
        blocks.append(d2_sheets.clip_block(clip, row_idx, cols=len(row_idx), frame_w=200,
                                           caption=caption(r, tau), marks=marks,
                                           cap_size=15))
        big_idx = d2_sheets.pick_frames(i0, j0, T, n_ramp=10)
        d2_sheets.save(out / "filmstrips" / f"{r['stem']}.png",
                       d2_sheets.clip_block(clip, big_idx, cols=7, frame_w=340,
                                            caption=caption(r, tau), marks=marks))
        shutil.copy(root / "videos" / f"{r['stem']}.mp4", out / "videos" / f"{r['stem']}.mp4")
        index.append({"stem": r["stem"], "shader": r["shader"], "easing": r["easing"],
                      "flip": r["flip"], "swap": r["swap"], "A": r["A"], "B": r["B"],
                      "timing": r["timing"], "m1_p10": r["m1_p10"],
                      "m2_max_dq": r["m2_max_dq"],
                      "seam_ratio": r["assert2"]["seam_ratio"],
                      "pure_phase_mae": [r["assert1"]["mae_pure_a"],
                                         r["assert1"]["mae_pure_b"]],
                      "frames_shown": row_idx})

    d2_sheets.save(out / "owner_sheet.png", d2_sheets.stack_blocks(blocks, gap=14))
    (out / "owner_sheet_index.json").write_text(json.dumps(
        {"n_clips": len(index), "sampled_from": "gate-PASSING D2 audit clips",
         "sample_seed": SEED, "frozen_tau": tau, "n_passing": len(passing),
         "n_audit_clips": len(rows), "clips": index}, indent=2))
    print(f"[owner] wrote {out / 'owner_sheet.png'}  ({len(blocks)} clips)")


if __name__ == "__main__":
    main()
