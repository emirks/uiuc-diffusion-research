"""ctt_v2 S3 RE-PILOT — 63 clips with the coverage gate LIVE and pair-swap retry.

The first pilot failed 23/63 on blind audit with every gate green. This re-pilot is the
advisor's single permitted engine iteration, and it changes exactly two things:

  1. the frozen COVERAGE gate rejects a clip whose ramp contains pixels no mesh covered
     (GATE_CALIB.json, calibrated on the 63 hand labels and frozen before this ran);
  2. rejection triggers a CONTENT-PAIR SWAP, never an amplitude or timing change — both are
     part of exact-op identity, and adapting them per content would silently destroy the
     same-op invariance the whole stratum exists to provide.

Contents come from the CLEAN pool (the 20 letterboxed bank clips are excluded), which alone
removes several of the original 23.

PRE-COMMITTED PASS BARS (advisor, frozen before this run):
    (i)   blind audit of all ACCEPTED clips <= 5% BAD
    (ii)  no family with >= 1/3 of its accepted clips BAD
    (iii) visible-parallax audit >= 90% of accepted
    (iv)  projected overdraw <= 3x and op-drop <= 30% after any blacklisting
    (v)   audit ~15 REJECTED clips too — if most rejects look fine the gate is over-rejecting
          (non-gating, but reported)
Any family or arm with > 50% gate rejection is BLACKLISTED and its budget redistributed —
mechanically, so the decision is data rather than taste.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import TeeLogger, load_config  # noqa: E402

from engine3d import cameras, metrics, ops3d, videoio  # noqa: E402
from engine3d.depth import to_view_depth  # noqa: E402
from engine3d.render3d import MeshRenderer  # noqa: E402

log = logging.getLogger("s3repilot")

FORCED = [
    ("dissolve_fbm", dict(dissolve="fbm", dissolve_freq=1.6)),
    ("rackfocus",    dict(focus=12.0)),
    ("fog",          dict(fog=1.6)),
    ("dollyzoom",    dict(dolly_zoom=1.0)),
    ("depthwipe",    dict(blend="depth_wipe", wipe_band=0.2)),
]
DZ_FALLBACK = ("handheld_mblur", dict(handheld=0.7, motion_blur=3))
MAX_PAIR_ATTEMPTS = 6          # per slot, before that slot is abandoned


def op_id_of(op: ops3d.Operator3D, onset: int, release: int) -> str:
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in op.__dict__.items()}
    d.pop("seed", None)
    return f"{op.path}_{hashlib.sha1((json.dumps(d, sort_keys=True) + f'|{onset}|{release}').encode()).hexdigest()[:8]}"


def main() -> None:
    cfg = load_config(HERE / "config_s3.yaml")
    inf, s3, tim = cfg["inference"], cfg["s3"], cfg["timing"]
    NF = inf["num_frames"]
    rng = random.Random(cfg["runtime"]["seed"] + 7)

    cal = json.loads((HERE / "GATE_CALIB.json").read_text())
    if cal["verdict"] != "FROZEN":
        sys.exit("[repilot] REFUSING TO RUN: the coverage gate ESCAPED calibration "
                 f"({cal['verdict']}). Shipping an uncalibrated gate is not an option.")
    GATE_STAT, GATE_THR = cal["frozen"]["stat"], cal["frozen"]["thr"]
    log.info("gate: %s > %.5f (recall %d, FP %d on the 63 labels)",
             GATE_STAT, GATE_THR, cal["frozen"]["recall"], cal["frozen"]["fp"])

    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / "repilot"
    for d in (out_dir / "videos", out_dir / "filmstrips", out_dir / "rejected"):
        d.mkdir(parents=True, exist_ok=True)

    with TeeLogger(out_dir / "repilot.log"):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout, force=True)
        log.info("gate FROZEN: %s > %.5f", GATE_STAT, GATE_THR)

        pool = json.loads((REPO_ROOT / cfg["inputs"]["content_pool"]).read_text())
        train_ids = [e["clip_id"] for e in pool["training"]]
        clip_path = {e["clip_id"]: Path(e["mp4"]) for e in pool["training"] + pool["reserved"]}
        cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]
        log.info("contents: %d CLEAN training endpoints (letterboxed excluded)", len(train_ids))

        renderer = MeshRenderer(inf["width"], inf["height"], step=inf["mesh_step"])
        log.info("GL: %s", renderer.renderer_name())

        fr: dict = {}
        dp: dict = {}

        def frames(cid):
            if cid not in fr:
                f = videoio.read_clip(clip_path[cid])[:NF]
                assert f.shape[0] == NF, f"{cid}: {f.shape[0]} frames"
                fr[cid] = f
                while len(fr) > 16:
                    fr.pop(next(iter(fr)))
            return fr[cid]

        def depth(cid):
            if cid not in dp:
                dp[cid] = np.load(cache_dir / f"{cid}.npy").astype(np.float32)
                while len(dp) > 16:
                    dp.pop(next(iter(dp)))
            return dp[cid]

        def draw_timing():
            w0, w1 = tim["window"]
            span = w1 - w0
            return (int(round(w0 + rng.random() * tim["jitter_frac"] * span)),
                    int(round(w1 - rng.random() * tim["jitter_frac"] * span)))

        def base_op(family, **over):
            op = ops3d.sample_operator(rng)
            op.path = family
            op.easing = rng.choice(ops3d.PATH_EASINGS)
            op.amplitude = rng.uniform(0.55, 1.6) * s3["amplitude_scale"]
            op.handheld = op.fog = op.focus = op.dolly_zoom = 0.0
            op.motion_blur, op.dissolve, op.blend = 1, "none", "crossfade"
            for k, v in over.items():
                setattr(op, k, v)
            return op

        accepted, rejected = [], []
        t0 = time.time()
        n_render = 0
        for family in sorted(cameras.PATHS):
            for slot in range(s3["pilot_per_family"]):
                tag, over = ("random", {}) if slot < 4 else FORCED[slot - 4]
                if tag == "dollyzoom" and family not in ("dolly", "spiral"):
                    tag, over = DZ_FALLBACK
                op = base_op(family, **over)
                onset, release = draw_timing()      # frozen for this op across all attempts
                oid = op_id_of(op, onset, release)
                got = False
                for attempt in range(MAX_PAIR_ATTEMPTS):
                    a, b = rng.sample(train_ids, 2)     # SWAP THE PAIR, never the operator
                    o = ops3d.Operator3D(**op.__dict__)
                    cov: list = []
                    t = time.time()
                    clip = ops3d.render_transition_stream(
                        renderer, o, frames(a), frames(b), depth(a), depth(b),
                        onset, release, coverage_out=cov)
                    render_s = time.time() - t
                    n_render += 1
                    unc = np.array([x["uncovered"] for x in cov])
                    weak = np.array([x["weak"] for x in cov])
                    stat = {"unc_max": float(unc.max()), "unc_p95": float(np.percentile(unc, 95)),
                            "unc_mean": float(unc.mean()), "weak_max": float(weak.max()),
                            "weak_p95": float(np.percentile(weak, 95)),
                            "weak_mean": float(weak.mean())}[GATE_STAT]
                    assert np.array_equal(clip[:onset + 1], frames(a)[:onset + 1])
                    assert np.array_equal(clip[release:], frames(b)[release:])
                    r_in = ops3d.join_ratio(clip, onset + 1)
                    r_out = ops3d.join_ratio(clip, release)
                    rec = {"family": family, "tag": tag, "op_id": oid, "A": a, "B": b,
                           "onset": onset, "release": release, "attempt": attempt,
                           "gate_stat": GATE_STAT, "gate_value": round(stat, 5),
                           "join_in": round(r_in, 3), "join_out": round(r_out, 3),
                           "render_s": round(render_s, 2), "describe": o.describe()}
                    if stat > GATE_THR or max(r_in, r_out) > cfg["gate"]["join_max"]:
                        rec["reason"] = ("coverage" if stat > GATE_THR else "join")
                        stem = f"s3r_REJ_{len(rejected):03d}_{family}_{tag}"
                        rec["stem"] = stem
                        if len(rejected) < 20:      # keep a bounded sample for the reject audit
                            ramp = np.linspace(onset, release, 7).astype(int).tolist()
                            PIL.Image.fromarray(videoio.filmstrip(
                                clip, [0, 8] + ramp + [112, 120])).save(
                                out_dir / "rejected" / f"{stem}.jpg", quality=88)
                        rejected.append(rec)
                        continue
                    za0 = to_view_depth(depth(a)[onset + 1], o.depth_near, o.depth_far,
                                        o.depth_gamma)
                    rec["parallax"] = metrics.parallax_index(clip[onset + 1: onset + 7], za0)
                    stem = f"s3r_{len(accepted):03d}_{family}_{tag}"
                    rec["stem"] = stem
                    videoio.write_clip(out_dir / "videos" / f"{stem}.mp4", clip, fps=inf["fps"])
                    ramp = np.linspace(onset, release, 7).astype(int).tolist()
                    PIL.Image.fromarray(videoio.filmstrip(
                        clip, [0, 8] + ramp + [112, 120])).save(
                        out_dir / "filmstrips" / f"{stem}.jpg", quality=88)
                    accepted.append(rec)
                    got = True
                    break
                log.info("%-8s slot %d %-15s %s (%d attempts)", family, slot, tag,
                         "ACCEPT" if got else "ABANDONED", attempt + 1)

        # mechanical blacklisting: >50% gate rejection at family or arm granularity
        def rate(key):
            out = {}
            for v in sorted({r[key] for r in accepted + rejected}):
                na = sum(1 for r in accepted if r[key] == v)
                nr = sum(1 for r in rejected if r[key] == v)
                out[v] = {"accepted": na, "rejected": nr,
                          "reject_rate": round(nr / max(na + nr, 1), 3),
                          "blacklist": nr / max(na + nr, 1) > 0.50}
            return out

        fam_stats, tag_stats = rate("family"), rate("tag")
        overdraw = n_render / max(len(accepted), 1)
        result = {
            "created": "2026-07-25", "phase": "RE-PILOT",
            "gate": {"stat": GATE_STAT, "threshold": GATE_THR,
                     "calibration": cal["frozen"], "source": "GATE_CALIB.json (frozen)"},
            "n_accepted": len(accepted), "n_rejected": len(rejected),
            "n_rendered": n_render, "overdraw": round(overdraw, 3),
            "overdraw_bar": 3.0, "overdraw_pass": overdraw <= 3.0,
            "by_family": fam_stats, "by_arm": tag_stats,
            "blacklisted_families": [k for k, v in fam_stats.items() if v["blacklist"]],
            "blacklisted_arms": [k for k, v in tag_stats.items() if v["blacklist"]],
            "minutes": round((time.time() - t0) / 60, 1),
            "visual_audit": {"status": "PENDING — operator blind review of ACCEPTED clips, "
                                       "plus ~15 REJECTED for over-rejection check",
                             "bars": {"max_bad_frac": 0.05, "max_bad_frac_per_family": 1 / 3,
                                      "min_visible_parallax_frac": 0.90}},
            "accepted": accepted, "rejected": rejected,
        }
        (HERE / "REPILOT_RESULT.json").write_text(json.dumps(result, indent=1))

        order = list(range(len(accepted)))
        random.Random(9191).shuffle(order)
        (out_dir / "BLIND_ORDER.json").write_text(json.dumps(
            {"note": "score GOOD/BAD and visible-parallax yes/no before revealing identities",
             "order": [{"blind_id": i, "stem": accepted[j]["stem"]}
                       for i, j in enumerate(order)]}, indent=1))

        log.info("RE-PILOT: %d accepted, %d rejected, %d rendered (overdraw %.2fx), %.1f min",
                 len(accepted), len(rejected), n_render, overdraw, result["minutes"])
        for k, v in fam_stats.items():
            log.info("  %-8s accept %2d reject %2d (%.0f%%)%s", k, v["accepted"], v["rejected"],
                     100 * v["reject_rate"], "  BLACKLIST" if v["blacklist"] else "")
        for k, v in tag_stats.items():
            log.info("  %-16s accept %2d reject %2d (%.0f%%)%s", k, v["accepted"], v["rejected"],
                     100 * v["reject_rate"], "  BLACKLIST" if v["blacklist"] else "")


if __name__ == "__main__":
    main()
