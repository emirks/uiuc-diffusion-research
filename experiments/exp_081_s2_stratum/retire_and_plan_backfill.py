"""S2 backfill — retire the blacklisted shaders, then plan the ops that restore 800 x 10.

Advisor ruling (option A). Two standing rules pointed at this independently:

  * the >50%-rejection blacklist rule, which fires on its own terms; and
  * ruling 1c's replacement clause, which my renderer never implemented (it dropped ops
    without resampling), leaving the stratum at 755 ops instead of 800.

They are the same problem: 36 of the 45 dropped ops came from the 6 shaders that breach the
blacklist line.

WHY REJECTION RATE JUSTIFIES RETIREMENT even though every surviving clip passed all four gates —
the advisor's argument, recorded because it is the strongest one and nobody had voiced it:

    "For a 93%-rejection shader, the 10 surviving pairs are extreme gate-selections on content
     — the pairs where the content happened to mask the shader's problems. That entangles op and
     content BY SELECTION, which is precisely the confound this factorial stratum exists to
     eliminate. An op whose 10 contents were chosen by survivorship is a degraded instance of the
     same-op-any-content invariance signal even when every surviving clip is individually fine."

Plus the campaign's own history: gates are imperfect instruments (the M1 floor escape, the S3
coverage escape, D2's original mean-vs-max assert), and in D2 six of seven baseline-BAD clips
came from shaders later blacklisted on rejection rate.

This script:
  1. writes D2_POLICY_V2.json — a NEW versioned file, never a silent mutation of the frozen one;
  2. retires every clip and manifest row belonging to the 6 blacklisted shaders;
  3. plans the backfill: same-shader replacements for drops from clean shaders, and the
     blacklisted shaders' op budget redistributed across the survivors to keep per-shader counts
     balanced (balance for diversity, not for ease);
  4. appends the new ops to PLAN_S2.json at indices >= 800 so existing clips keep their indices.
"""

from __future__ import annotations

import collections
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "experiments/exp_077_synth_stratum"))

from diffusion.exp_utils import load_config  # noqa: E402

import streams_real as sr  # noqa: E402
from engine import operators, shaders  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402
from plan_s2 import HOLDOUT_S2, s2_op_id  # noqa: E402

ROOT = REPO_ROOT / "outputs/videos/ctt_v2_s2/full"
REJECT_RATE_BAR = 0.50
TARGET_OPS = 800


def main() -> None:
    cfg = load_config(HERE / "config_s2.yaml")
    smp, s2 = cfg["sampling"], cfg["s2"]
    H, W = cfg["inference"]["height"], cfg["inference"]["width"]
    T, K = cfg["inference"]["num_frames"], cfg["inference"]["anchor_frames"]
    rng = random.Random(cfg["runtime"]["seed"] + 999)

    plan = json.loads((HERE / "PLAN_S2.json").read_text())
    pairs, m = plan["pairs"], plan["design"]["contents_per_op"]

    ops_log = []
    for f in sorted(glob.glob(str(ROOT / "meta/ops_shard*.jsonl"))):
        ops_log += [json.loads(l) for l in open(f) if l.strip()]
    clips = []
    for f in sorted(glob.glob(str(ROOT / "meta/clips_shard*.jsonl"))):
        clips += [json.loads(l) for l in open(f) if l.strip()]

    # ---- 1. who breaches the bar --------------------------------------------------------
    per = collections.defaultdict(lambda: {"accepted": 0, "rejected": 0, "ops": 0, "dropped": 0})
    for o in ops_log:
        d = per[o["shader"]]
        d["accepted"] += o["n_slots"]
        d["rejected"] += len(o["rejects"])
        d["ops"] += 1
        d["dropped"] += int(o["dropped"])
    rates = {k: v["rejected"] / max(v["accepted"] + v["rejected"], 1) for k, v in per.items()}
    black = sorted(k for k, r in rates.items() if r > REJECT_RATE_BAR)
    assert not (set(black) & set(HOLDOUT_S2)), \
        f"a blacklisted shader is also in the held-out registry: {set(black) & set(HOLDOUT_S2)}"
    survivors = sorted(k for k in per if k not in set(black))
    print(f"[bf] blacklisting {len(black)}: {black}")
    print(f"[bf] surviving trainable shaders: {len(survivors)}")

    policy = json.loads((REPO_ROOT / cfg["inputs"]["policy"]).read_text())
    v2 = {
        "version": "D2_POLICY v2", "created": "2026-07-25",
        "supersedes": "D2_POLICY_FINAL.json (v1, unmodified — this is an additive version, not a "
                      "mutation of a frozen file)",
        "authority": "fable-advisor: >50% per-shader rejection blacklist, fired on the S2 build",
        "change": "6 shaders retired on measured rejection rate; trainable families 62 -> 56, "
                  "blacklist 40 -> 46",
        "rationale_selection_confound":
            "For a high-rejection shader the surviving pairs are extreme gate-selections on "
            "content — the pairs whose content happened to mask the shader's problems. That "
            "entangles operator and content BY SELECTION, degrading the same-op-any-content "
            "invariance signal even when every surviving clip is individually fine.",
        "rejection_table": {k: {"accepted": per[k]["accepted"], "rejected": per[k]["rejected"],
                                "rate": round(rates[k], 4), "ops": per[k]["ops"],
                                "dropped": per[k]["dropped"]} for k in sorted(per)},
        "newly_blacklisted": black,
        "blacklist": sorted(set(policy["blacklist"]) | set(black)),
        "keep_shaders": sorted(set(policy["keep_shaders"]) - set(black)),
        "holdout_unaffected": sorted(HOLDOUT_S2),
        "eval_stock_note": "the 6 are flagged DEAD in the eval-stock manifest too — a "
                           "93%-rejection shader is not a fair eval probe either; these families "
                           "are dead, not held out.",
    }
    (HERE / "D2_POLICY_V2.json").write_text(json.dumps(v2, indent=2))
    print(f"[bf] wrote D2_POLICY_V2.json (blacklist {len(policy['blacklist'])} -> "
          f"{len(v2['blacklist'])}, keep {len(policy['keep_shaders'])} -> {len(v2['keep_shaders'])})")

    # ---- 2. retire the blacklisted shaders' media + rows ---------------------------------
    retire_ops = {o["op_index"] for o in ops_log if o["shader"] in set(black)}
    retire_stems = [c["stem"] for c in clips if c["op_index"] in retire_ops]
    ret_dir = ROOT / "retired_blacklisted"
    (ret_dir / "videos").mkdir(parents=True, exist_ok=True)
    (ret_dir / "filmstrips").mkdir(parents=True, exist_ok=True)
    moved = 0
    for s in retire_stems:
        for sub, ext in (("videos", "mp4"), ("filmstrips", "jpg")):
            p = ROOT / sub / f"{s}.{ext}"
            if p.exists():
                p.rename(ret_dir / sub / f"{s}.{ext}")
                moved += 1
    # rewrite the manifests without the retired rows (originals kept as .v1)
    for f in sorted(glob.glob(str(ROOT / "meta/clips_shard*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        keep = [r for r in rows if r["op_index"] not in retire_ops]
        Path(f + ".v1").write_text("".join(json.dumps(r) + "\n" for r in rows))
        Path(f).write_text("".join(json.dumps(r) + "\n" for r in keep))
    for f in sorted(glob.glob(str(ROOT / "meta/ops_shard*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        keep = [r for r in rows if r["op_index"] not in retire_ops]
        Path(f + ".v1").write_text("".join(json.dumps(r) + "\n" for r in rows))
        Path(f).write_text("".join(json.dumps(r) + "\n" for r in keep))
    remaining = [c for c in clips if c["op_index"] not in retire_ops]
    complete_left = {c["op_index"] for c in remaining}
    print(f"[bf] retired {len(retire_ops)} ops / {len(retire_stems)} clips "
          f"({moved} media files moved to retired_blacklisted/)")
    print(f"[bf] remaining: {len(complete_left)} complete ops / {len(remaining)} clips")

    # ---- 3. how many new ops, and from which shaders --------------------------------------
    need = TARGET_OPS - len(complete_left)
    have = collections.Counter(c["shader"] for c in remaining)
    have = {s: have[s] // m for s in survivors}
    # balance: fill the shaders furthest below the post-backfill mean first
    alloc = collections.Counter()
    for _ in range(need):
        s = min(survivors, key=lambda x: (have[x] + alloc[x], x))
        alloc[s] += 1
    print(f"[bf] need {need} new ops -> mean per surviving shader "
          f"{(len(complete_left)+need)/len(survivors):.1f}; "
          f"alloc min/max {min(alloc.values())}/{max(alloc.values())}")

    # ---- 4. sample them, gate-2 pre-validated, and append at indices >= 800 ---------------
    z = np.load(REPO_ROOT / cfg["inputs"]["endpoint_frame_cache"])
    fa = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
    fb = {k[2:]: z[k] for k in z.files if k.startswith("b_")}
    runner = GLRunner(W, H)
    raw = shaders.load_bank(Path(cfg["model"]["shader_bank"]))
    gated, _ = operators.validate_bank(runner, raw, tol=smp["endpoint_tol"])
    bank = {k: v for k, v in gated.items() if k in set(survivors)}
    easings = [e for e in sr.d2_easings() if e in policy["keep_easings"]]
    seen = {o["op_id"] for o in plan["ops"]}

    # pair usage from the SURVIVING grid, so the backfill keeps ops-per-pair regular
    pdeg = collections.Counter()
    for c in remaining:
        pdeg[c["pair_id"]] += 1
    tol, n_probe = smp["endpoint_tol_operator"], s2["probe_pairs"]
    new_ops, tries, t0 = [], 0, time.time()
    for shader, k in sorted(alloc.items()):
        made = 0
        while made < k:
            op = sr.make_operator(bank, rng, shader, easings=easings, p_flip=smp["p_flip"],
                                  p_swap=smp["p_swap"], p_vary=smp["p_vary"], param_filter=None)
            tries += 1
            if tries > need * 200:
                raise RuntimeError("backfill operator pre-validation is rejecting almost everything")
            probe = rng.sample(range(len(pairs)), n_probe)
            res = [operators.check_operator(runner, bank, op, fa[pairs[i]["A"]], fb[pairs[i]["B"]])
                   for i in probe]
            if not all(max(d0, d1) <= tol for d0, d1 in res):
                continue
            timing = sr.sample_timing(rng, T, K, frac=smp["timing_frac"])
            oid = s2_op_id(shader, op.params, op.easing, op.flip, op.swap, timing)
            if oid in seen:
                continue
            seen.add(oid)
            i0, j0 = sr.phase_indices(T, timing["onset"], timing["release"])
            # candidates: least-used pairs first, endpoint-disjointness enforced at render time
            cand = sorted(range(len(pairs)), key=lambda i: (pdeg[i], rng.random()))
            chosen, used = [], set()
            for pi in cand:
                a, b = pairs[pi]["A"], pairs[pi]["B"]
                if a in used or b in used:
                    continue
                chosen.append(pi)
                used |= {a, b}
                if len(chosen) == m:
                    break
            for pi in chosen:
                pdeg[pi] += 1
            spares = [pi for pi in cand if pi not in set(chosen)][: s2["candidates_per_op"] - m]
            new_ops.append({
                "op_id": oid, "shader": shader, "params": op.params, "easing": op.easing,
                "flip": op.flip, "swap": op.swap, "extension": "none", "aux_kind": None,
                "timing": {kk: timing[kk] for kk in ("onset", "release", "duration", "u1", "u2")},
                "phase": {"i0": i0, "j0": j0},
                "probe_gate2_max": round(max(max(d) for d in res), 4),
                "backfill": True, "candidates": chosen + spares})
            made += 1
    print(f"[bf] sampled {len(new_ops)} backfill ops in {time.time()-t0:.0f}s "
          f"({tries} draws, {100*(tries-len(new_ops))/max(tries,1):.0f}% gate-2 rejected)")

    plan["ops"] = plan["ops"] + new_ops
    plan["design"]["n_ops_original"] = TARGET_OPS
    plan["design"]["n_ops_in_file"] = len(plan["ops"])
    plan["backfill"] = {
        "created": "2026-07-25", "authority": "fable-advisor option (A)",
        "blacklisted": black, "retired_ops": sorted(retire_ops),
        "retired_clips": len(retire_stems),
        "surviving_ops_before_backfill": len(complete_left),
        "new_ops": len(new_ops), "allocation": dict(alloc),
        "target_total_ops": TARGET_OPS,
        "policy": "D2_POLICY_V2.json"}
    (HERE / "PLAN_S2.json").write_text(json.dumps(plan, indent=1))
    (HERE / "BACKFILL_PLAN.json").write_text(json.dumps(
        {**plan["backfill"], "rejection_rates": {k: round(v, 4) for k, v in sorted(rates.items())}},
        indent=1))
    print(f"[bf] PLAN_S2.json now holds {len(plan['ops'])} ops "
          f"({len(complete_left)} rendered + {len(new_ops)} to render = {TARGET_OPS})")


if __name__ == "__main__":
    main()
