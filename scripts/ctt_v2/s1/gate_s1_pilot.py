#!/usr/bin/env python
"""CTT v2 / S1 -- the pilot gate. A5 RULING 3(i)+(ii), bars pre-registered in S1_GRID.json.

Two independent stages, deliberately separable because they have different failure modes and
different dependencies:

  --stage mechanical   per-clip HARD rejects, CPU only, no API, no learned substrate:
                         * decode corruption   (ffmpeg fails / wrong frame count / wrong geometry)
                         * frozen or black     (no inter-frame motion after the anchor; dark frames)
                         * endpoint identity   (prefix rel-L2 > tau, tau measured on Day 0)
                       These are the ONLY per-clip rejects Ruling 3 allows. Batch-level DINOv2
                       and every other harness substrate are banned from data selection, so
                       nothing here loads a model.

  --stage gemini       the batch gate: BLIND 11-way class identification, gemini-3.5-flash,
                       temp 0, maxOutputTokens 2000, bar top-1 >= 80% (chance 9.1%), run
                       ALONGSIDE a control arm of real corpus clips of the same 11 classes.
                       The control is not decoration: if the judge cannot name these manners
                       off real footage either, a low score on the generated batch says nothing
                       about S1 and the instrument is what failed.

Verdict (pre-registered, from S1_GRID.json -- do not re-derive it here):
    PASS               batch top-1 >= 80% AND control top-1 >= 80% AND mechanical rejects <= 10%
    FAIL_S1_DROPS      batch top-1 < 80% with a PASSING control -> S1 drops, mix renormalizes
                       to S0 15 / S2 85, schedule does not slip
    INSTRUMENT_INVALID control top-1 < 80% -> re-adjudicate the instrument, do not blame S1

Blindness, concretely: the judge is shown one clip and the 11 class names in a FIXED
seed-42 shuffle (same order for every item, so no positional cue tracks the truth), is told
nothing about which specialist made it, and generated and control items go through a single
identical prompt. Filenames never reach the API -- only bytes.

Usage
-----
  PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python
  $PY scripts/ctt_v2/s1/gate_s1_pilot.py --stage mechanical \
      --grid $LAB/misc/ctt_v2_final/S1_GRID.json --out outputs/ctt_v2/s1/gate

  source $LAB/secrets/gemini_transition.env
  $PY scripts/ctt_v2/s1/gate_s1_pilot.py --stage gemini  ...same args...
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_tau_endpoint import H, PX_PREFIX, W, decode, rel_l2  # noqa: E402

STD_FRAMES = 121
JUDGE_MODEL = "gemini-3.5-flash"
JUDGE_TEMPERATURE = 0.0
#: thinking models emit ~110 thought tokens before any visible text, so a low cap returns an
#: EMPTY string rather than an answer (measured, DOSSIER 5.1). Ruling 3 pins >= 2000.
JUDGE_MAX_TOKENS = 2000
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

#: frozen: mean abs inter-frame delta below this (0-1 scale) over the post-anchor window
FROZEN_DELTA = 1.0 / 255.0
#: black: mean luma below this on at least BLACK_FRAC of frames
BLACK_LUMA, BLACK_FRAC = 8.0 / 255.0, 0.10


# --------------------------------------------------------------------- mechanical
def decode_all(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={proc.returncode}: {proc.stderr.decode()[:200]}")
    arr = np.frombuffer(proc.stdout, np.uint8)
    if arr.size == 0 or arr.size % (H * W * 3):
        raise ValueError(f"unexpected geometry: {arr.size} bytes (want multiples of {H*W*3})")
    return arr.reshape(-1, H, W, 3).astype(np.float32) / 255.0


def mechanical_check(row: dict, tau: float, conds_dir: Path) -> dict:
    out = {"row_id": row["row_id"], "arm": row["arm"], "bank": row["endpoint_a_bank"],
           "out_path": row["out_path"], "rejects": [], "prefix_rel_l2": None}
    path = REPO_ROOT / row["out_path"]
    if not path.exists():
        out["rejects"].append("missing")
        return out
    try:
        vid = decode_all(path)
    except Exception as exc:  # noqa: BLE001
        out["rejects"].append(f"decode_corruption:{type(exc).__name__}")
        return out

    if len(vid) != STD_FRAMES:
        out["rejects"].append(f"frame_count:{len(vid)}")
    out["frames"] = len(vid)

    # frozen / black are judged on the GENERATED part, after the copied prefix anchor
    tail = vid[PX_PREFIX:]
    if len(tail) > 1:
        delta = float(np.abs(np.diff(tail, axis=0)).mean())
        out["mean_interframe_delta"] = delta
        if delta < FROZEN_DELTA:
            out["rejects"].append(f"frozen:{delta:.5f}")
    luma = vid.mean(axis=(1, 2, 3))
    dark = float((luma < BLACK_LUMA).mean())
    out["dark_frame_fraction"] = dark
    if dark >= BLACK_FRAC:
        out["rejects"].append(f"black:{dark:.3f}")

    anchor = conds_dir / f"{row['endpoint_a']}_start9.mp4"
    try:
        got = rel_l2(vid[:PX_PREFIX], decode(anchor, PX_PREFIX))
        out["prefix_rel_l2"] = got
        if got > tau:
            out["rejects"].append(f"endpoint_identity:{got:.4f}>tau{tau:.4f}")
    except Exception as exc:  # noqa: BLE001
        out["rejects"].append(f"anchor_unreadable:{type(exc).__name__}")

    return out


# --------------------------------------------------------------------- gemini gate
def judge_prompt(classes: list[str]) -> str:
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(classes))
    return (
        "You are shown one short video clip in which a visual transition or transformation "
        "effect happens to the opening scene. Your task is to identify WHICH KIND of effect "
        "it is, from this fixed list of 11 possibilities:\n\n"
        f"{listing}\n\n"
        "Judge only the manner of the transformation -- what visually happens and how -- not "
        "the subject matter or setting of the footage. Answer with exactly one line:\n"
        "ANSWER: <one label copied verbatim from the list above>"
    )


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def ask_judge(path: Path, classes: list[str], timeout: int = 300) -> dict:
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "video/mp4", "data": _b64(path)}},
            {"text": judge_prompt(classes)},
        ]}],
        "generationConfig": {"temperature": JUDGE_TEMPERATURE,
                             "maxOutputTokens": JUDGE_MAX_TOKENS},
    }
    url = f"{API_ROOT}/{JUDGE_MODEL}:generateContent"
    key = os.environ["GEMINI_API_KEY"]
    for attempt in range(4):
        r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                          json=body, timeout=timeout)
        if r.status_code == 429:
            # A4/operator directive: 429 is a HARD STOP, never grind the key
            return {"error": f"HTTP429: {r.text[:200]}", "hard_stop": True}
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            return {"error": f"HTTP{r.status_code}: {r.text[:200]}"}
        data = r.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        return {"text": text, "raw": data}
    return {"error": "exhausted retries"}


def parse_answer(text: str, classes: list[str]) -> str | None:
    """Exact-label match, longest first so `shadow_smoke` never resolves to `shadow`."""
    low = (text or "").lower()
    if "answer:" in low:
        low = low.split("answer:", 1)[1]
    for cls in sorted(classes, key=len, reverse=True):
        if cls.lower() in low or cls.replace("_", " ").lower() in low:
            return cls
    return None


def run_gemini(items: list[dict], classes: list[str], workers: int) -> list[dict]:
    def one(it: dict) -> dict:
        res = ask_judge(REPO_ROOT / it["path"], classes)
        return {**it, "response": res.get("text"), "error": res.get("error"),
                "hard_stop": res.get("hard_stop", False),
                "predicted": parse_answer(res.get("text", ""), classes),
                "correct": parse_answer(res.get("text", ""), classes) == it["truth"]}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, items))


def summarise(rows: list[dict], label: str) -> dict:
    scored = [r for r in rows if not r.get("error")]
    correct = sum(1 for r in scored if r["correct"])
    unparsed = sum(1 for r in scored if r["predicted"] is None)
    return {"arm_label": label, "n": len(rows), "scored": len(scored),
            "errors": len(rows) - len(scored), "unparsed": unparsed,
            "top1": (correct / len(scored)) if scored else None,
            "correct": correct}


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--stage", choices=("mechanical", "gemini", "all"), default="mechanical")
    ap.add_argument("--out", default="outputs/ctt_v2/s1/gate")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    grid = json.loads(Path(args.grid).read_text())
    tau = grid["tau_endpoint"]["value"]
    conds_dir = REPO_ROOT / grid["conds_dir"]
    gate = grid["batch_gate"]
    pilot = [r for r in grid["rows"] if r["in_pilot"]]
    classes = sorted(r["specialist_class"] for r in grid["rows"])
    classes = sorted(set(classes))
    assert len(classes) == 11, f"expected 11 classes, got {len(classes)}"

    outdir = REPO_ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": args.stage, "grid": args.grid, "tau_endpoint": tau,
                    "bars": {k: gate[k] for k in ("bar_top1", "chance", "model", "temperature",
                                                  "max_output_tokens")},
                    "verdict_rule": grid["verdict_rule"]}

    if args.stage in ("mechanical", "all"):
        with ThreadPoolExecutor(4) as pool:
            checks = list(pool.map(lambda r: mechanical_check(r, tau, conds_dir), pilot))
        rejected = [c for c in checks if c["rejects"]]
        by_bank: dict[str, list] = {}
        for c in checks:
            by_bank.setdefault(c["bank"], []).append(c)
        bank_rates = {b: sum(1 for c in v if c["rejects"]) / len(v) for b, v in by_bank.items()}
        diff_pp = (abs(bank_rates.get("synth", 0) - bank_rates.get("humanvid", 0)) * 100
                   if len(bank_rates) > 1 else 0.0)
        prefix_vals = [c["prefix_rel_l2"] for c in checks if c["prefix_rel_l2"] is not None]
        report["mechanical"] = {
            "n": len(checks), "rejected": len(rejected),
            "reject_rate": len(rejected) / len(checks) if checks else None,
            "bar_reject_rate": 0.10,
            "reject_rate_by_bank": bank_rates,
            "bank_differential_pp": diff_pp,
            "bank_differential_flag_bar_pp": gate["acceptance_by_bank_audit"]["flag_if_difference_pp_gt"],
            "bank_differential_FLAGGED": diff_pp > gate["acceptance_by_bank_audit"]["flag_if_difference_pp_gt"],
            "prefix_rel_l2": ({"n": len(prefix_vals), "p50": float(np.percentile(prefix_vals, 50)),
                               "p95": float(np.percentile(prefix_vals, 95)),
                               "max": float(max(prefix_vals))} if prefix_vals else None),
            "rejected_rows": rejected,
            "checks": checks,
        }
        m = report["mechanical"]
        print(f"[gate] mechanical: {m['rejected']}/{m['n']} rejected "
              f"({(m['reject_rate'] or 0) * 100:.1f}%, bar <= 10%)")
        if m["prefix_rel_l2"]:
            print(f"[gate]   prefix rel-L2 p50={m['prefix_rel_l2']['p50']:.4f} "
                  f"p95={m['prefix_rel_l2']['p95']:.4f} max={m['prefix_rel_l2']['max']:.4f} "
                  f"(tau={tau:.4f})")
        print(f"[gate]   by-bank reject differential {diff_pp:.1f} pp "
              f"(flag > {m['bank_differential_flag_bar_pp']} pp)")
        for c in rejected:
            print(f"[gate]   REJECT {c['row_id']}: {c['rejects']}")

    if args.stage in ("gemini", "all"):
        rng = random.Random(grid["seed"])
        shuffled = classes[:]
        rng.shuffle(shuffled)                     # fixed seed-42 option order, identical per item
        treat = [{"path": r["out_path"], "truth": r["specialist_class"],
                  "bank": r["endpoint_a_bank"], "arm_label": "S1_generated"}
                 for r in pilot if (REPO_ROOT / r["out_path"]).exists()]
        ctrl = [{"path": c["mp4"], "truth": c["class"], "bank": "corpus",
                 "arm_label": "control_real_corpus"}
                for c in gate["control_arm"]["clips"]
                if (REPO_ROOT / c["mp4"]).exists()]
        print(f"[gate] gemini: {len(treat)} generated + {len(ctrl)} control items, "
              f"judge={JUDGE_MODEL} temp={JUDGE_TEMPERATURE} maxOutputTokens={JUDGE_MAX_TOKENS}")
        print(f"[gate] option order (seed {grid['seed']}): {shuffled}")
        rows = run_gemini(treat + ctrl, shuffled, args.workers)
        if any(r.get("hard_stop") for r in rows):
            print("[gate] HTTP 429 -- HARD STOP. The Gemini project's prepayment credits were "
                  "depleted on 2026-07-28; top up billing and re-run. Nothing is adjudicated.",
                  file=sys.stderr)
        gen_rows = [r for r in rows if r["arm_label"] == "S1_generated"]
        ctl_rows = [r for r in rows if r["arm_label"] == "control_real_corpus"]
        report["gemini"] = {
            "option_order": shuffled,
            "generated": summarise(gen_rows, "S1_generated"),
            "control": summarise(ctl_rows, "control_real_corpus"),
            "generated_top1_by_bank": {
                b: summarise([r for r in gen_rows if r["bank"] == b], b)["top1"]
                for b in sorted({r["bank"] for r in gen_rows})},
            "rows": rows,
        }
        g, c = report["gemini"]["generated"], report["gemini"]["control"]
        print(f"[gate] generated top-1 = {g['top1']} ({g['correct']}/{g['scored']}), "
              f"bar {gate['bar_top1']}, chance {gate['chance']:.3f}")
        print(f"[gate] control   top-1 = {c['top1']} ({c['correct']}/{c['scored']})")
        if c["top1"] is not None and c["top1"] < gate["bar_top1"]:
            report["VERDICT"] = "INSTRUMENT_INVALID"
        elif g["top1"] is not None and g["top1"] >= gate["bar_top1"]:
            report["VERDICT"] = "PASS_pending_mechanical"
        elif g["top1"] is not None:
            report["VERDICT"] = "FAIL_S1_DROPS"
        print(f"[gate] VERDICT (gemini stage) = {report.get('VERDICT', 'UNDETERMINED')}")

    dest = outdir / f"gate_{args.stage}.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"[gate] -> {dest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
