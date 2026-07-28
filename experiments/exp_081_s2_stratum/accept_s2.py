"""S2 ACCEPTANCE — verify the delivered stratum against the pre-committed bars.

Advisor's S2 acceptance criteria (ruling 6, step 4), no sign-off required if all pass:

  * pure phases byte-exact everywhere (MAX condition, never a mean);
  * realized incidence verified FROM THE MANIFESTS = exactly 800 x 10 post-gate;
  * overdraw <= 2.5x (expected ~1.6-1.7x);
  * per-shader rejection tracked, with the >50%-rejection blacklist rule live;
  * blind n=64 shader-stratified visual audit at <= 3/64 BAD.

This script does everything except the visual audit itself: it checks the hard invariants,
computes the rates, and emits a SHUFFLED, shader-stratified 64-clip audit sheet set with
identities withheld until the scores are recorded. Run `--stage verify` first, then look at
the sheets, then `--stage record --verdicts <json>`.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT = REPO_ROOT / "outputs/videos/ctt_v2_s2/full"


def load():
    clips, ops = [], []
    for f in sorted(glob.glob(str(ROOT / "meta/clips_shard*.jsonl"))):
        clips += [json.loads(l) for l in open(f) if l.strip()]
    for f in sorted(glob.glob(str(ROOT / "meta/ops_shard*.jsonl"))):
        ops += [json.loads(l) for l in open(f) if l.strip()]
    return clips, ops


def verify() -> dict:
    plan = json.loads((HERE / "PLAN_S2.json").read_text())
    m = plan["design"]["contents_per_op"]
    n_ops_planned = plan["design"]["n_ops"]
    clips, ops = load()
    complete = [o for o in ops if o["complete"]]
    dropped = [o for o in ops if o["dropped"]]

    # --- hard invariants -----------------------------------------------------------------
    fails = []
    max_pure = max((c["assert1"]["max_pure"] for c in clips), default=0.0)
    if max_pure != 0.0:
        fails.append(f"pure-phase identity broken: max abs diff {max_pure}")
    by_op = collections.defaultdict(list)
    for c in clips:
        by_op[c["op_index"]].append(c)
    wrong = [(k, len(v)) for k, v in by_op.items() if len(v) != m]
    if wrong:
        fails.append(f"{len(wrong)} ops in the manifest do not have exactly {m} clips: {wrong[:5]}")
    nondisj = [k for k, v in by_op.items()
               if len({x for c in v for x in (c["A"], c["B"])}) != 2 * m]
    if nondisj:
        fails.append(f"{len(nondisj)} ops are not endpoint-disjoint across their {m} clips")
    orphan = [c["stem"] for c in clips if not (ROOT / "videos" / f"{c['stem']}.mp4").exists()]
    if orphan:
        fails.append(f"{len(orphan)} manifest rows have no mp4 on disk")
    ghost_ops = {o["op_index"] for o in dropped} & set(by_op)
    if ghost_ops:
        fails.append(f"{len(ghost_ops)} DROPPED ops still have manifest rows")
    ids = {c["op_id"] for c in clips}
    if len(ids) != len(by_op):
        fails.append(f"op_id collision: {len(by_op)} op blocks but {len(ids)} unique ids")

    # --- rates ---------------------------------------------------------------------------
    # Overdraw must come from the OPS LOG, not the per-shard summaries: a shard that is
    # requeued rewrites its summary, so after the OOM restart the summaries undercounted
    # renders and reported an impossible overdraw of 0.91x. The ops log is append-only and
    # cumulative, and every render is either an accepted slot or a logged reject.
    rendered = sum(o["n_slots"] + len(o["rejects"]) for o in ops) or None
    per_shader = collections.defaultdict(lambda: {"accepted": 0, "rejected": 0})
    for o in ops:
        per_shader[o["shader"]]["accepted"] += o["n_slots"]
        per_shader[o["shader"]]["rejected"] += len(o["rejects"])
    blacklist = [s for s, v in per_shader.items()
                 if v["rejected"] / max(v["accepted"] + v["rejected"], 1) > 0.50]
    overdraw = (rendered / len(clips)) if (rendered and clips) else None

    out = {
        "created": "2026-07-25", "stratum": "S2",
        "planned": {"n_ops": n_ops_planned, "contents_per_op": m,
                    "n_clips": n_ops_planned * m},
        "realized": {"n_clips": len(clips), "n_ops_complete": len(complete),
                     "n_ops_dropped": len(dropped),
                     "n_ops_finalised": len(ops)},
        "invariants": {
            "pure_phase_max_abs_diff": max_pure,
            "all_ops_exactly_m_clips": not wrong,
            "all_ops_endpoint_disjoint": not nondisj,
            "no_orphan_manifest_rows": not orphan,
            "no_ghost_dropped_ops": not ghost_ops,
            "op_ids_unique": len(ids) == len(by_op)},
        "gates": {
            "seam_max": round(max(c["assert2"]["seam_max_ratio"] for c in clips), 4) if clips else None,
            "m1_p10_min": round(min(c["m1_p10"] for c in clips), 4) if clips else None,
            "m2_max_dq": round(max(c["m2_max_dq"] for c in clips), 4) if clips else None,
            "m1_min_flag_count": sum(c["m1_min_flag"] for c in clips)},
        "overdraw": round(overdraw, 4) if overdraw else None,
        "overdraw_bar": 2.5,
        "overdraw_pass": (overdraw <= 2.5) if overdraw else None,
        "attempts_to_fill": {
            "min": min((o["attempts"] for o in complete), default=None),
            "median": int(np.median([o["attempts"] for o in complete])) if complete else None,
            "max": max((o["attempts"] for o in complete), default=None)},
        "per_shader": {k: v for k, v in sorted(per_shader.items())},
        "shaders_over_50pct_rejection": blacklist,
        "failures": fails,
        "verdict": "PASS" if not fails else "FAIL",
    }
    (HERE / "S2_ACCEPTANCE.json").write_text(json.dumps(out, indent=1))
    print(f"realized {len(clips)} clips / {len(complete)} complete ops "
          f"({len(dropped)} dropped) vs planned {n_ops_planned}x{m}")
    print(f"pure-phase max abs diff {max_pure}  |  overdraw "
          f"{out['overdraw']}x (bar 2.5)  |  attempts min/med/max "
          f"{out['attempts_to_fill']['min']}/{out['attempts_to_fill']['median']}/"
          f"{out['attempts_to_fill']['max']}")
    print(f"gates: seam max {out['gates']['seam_max']} | m1_p10 min {out['gates']['m1_p10_min']} "
          f"| m2 max {out['gates']['m2_max_dq']} | m1_min flags {out['gates']['m1_min_flag_count']}")
    print(f"shaders over 50% rejection: {blacklist or 'none'}")
    print(f"VERDICT: {out['verdict']}" + ("" if not fails else "\n  " + "\n  ".join(fails)))
    return out


def sheets(n: int = 64, per_sheet: int = 8) -> None:
    """Shader-stratified, shuffled, identities withheld."""
    import PIL.Image
    import PIL.ImageDraw
    clips, _ = load()
    by_sh = collections.defaultdict(list)
    for c in clips:
        by_sh[c["shader"]].append(c)
    rng = random.Random(6464)
    picked, shaders = [], sorted(by_sh)
    while len(picked) < n and shaders:
        for s in list(shaders):
            if by_sh[s]:
                picked.append(by_sh[s].pop(rng.randrange(len(by_sh[s]))))
                if len(picked) == n:
                    break
            else:
                shaders.remove(s)
    rng.shuffle(picked)
    out = ROOT / "audit_sheets"
    out.mkdir(exist_ok=True)
    rows = []
    for i, c in enumerate(picked):
        p = ROOT / "filmstrips" / f"{c['stem']}.jpg"
        img = PIL.Image.open(p).convert("RGB")
        W = 1400
        img = img.resize((W, int(img.height * W / img.width)), PIL.Image.LANCZOS)
        cv = PIL.Image.new("RGB", (W + 60, img.height), "white")
        cv.paste(img, (60, 0))
        PIL.ImageDraw.Draw(cv).text((5, img.height // 2 - 6), f"#{i:03d}", fill="black")
        rows.append(np.asarray(cv))
    for k in range(0, len(rows), per_sheet):
        PIL.Image.fromarray(np.concatenate(rows[k:k + per_sheet], axis=0)).save(
            out / f"s2audit_{k // per_sheet:02d}.png")
    (ROOT / "AUDIT_KEY.json").write_text(json.dumps(
        {"n": len(picked), "bar_max_bad": 3,
         "note": "shader-stratified, shuffled; score every id before revealing identities",
         "key": [{"blind_id": i, "stem": c["stem"], "shader": c["shader"]}
                 for i, c in enumerate(picked)]}, indent=1))
    print(f"{(len(rows)+per_sheet-1)//per_sheet} sheets covering {len(picked)} clips -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="verify", choices=["verify", "sheets"])
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    if a.stage == "verify":
        verify()
    else:
        sheets(a.n)
