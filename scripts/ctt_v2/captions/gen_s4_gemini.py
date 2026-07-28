#!/usr/bin/env python3
"""S4 first-frame captions via Gemini — the SAME instrument the locked store used.

Why Gemini and not another Sonnet round: gate 8a is driven by grammatical register, not
length. Measured on the Sonnet store, stripping length and punctuation out of the probe
entirely moves it 0.8849 -> 0.8672, still far above the 0.73 bar; the separating features are
preposition habits (Sonnet: against/with/beneath/by · corpus: on/in/under/while). A length fix
therefore cannot close 8a. The only lever with a real chance is using the captioner that
already sits at 0.6819 with this exact prompt.

Everything register-bearing is IMPORTED from `generate_descriptions`, never retyped:
`build_system_prompt(role="A", n, variant="v2")`, `sample_length`'s per-item draw over the 171
corpus word counts, `calibrate_ask`'s inversion of the realised-length fit, `postprocess`'s
trailing-period strip, and `_post`'s retry/429 policy. If the prompt drifts, it drifts for both
stores at once, which is the point.

Two deliberate departures, both owner-directed (2026-07-28, "do it very very quickly, do not
gate everything"):
  * ONE image, not a 9-frame video clip -- S4 conditions on video frame 0 alone.
  * NO per-item Layer-2 audit pass. Generation only.

Concurrency, not the Batch API: Batch is half price but async with up to a 24h SLA, and the
directive was minutes.

Usage:
    source $LAB/secrets/gemini_transition.env
    python scripts/ctt_v2/captions/gen_s4_gemini.py --workers 32
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2/captions"))
import generate_descriptions as gd  # noqa: E402
from caption_common import load_length_empirical  # noqa: E402

ROSTER = REPO / "outputs/ctt_v2/encodes/S4/ROSTER.json"
FRAMES = REPO / "outputs/ctt_v2/captions/s4_frame0_384"
OUTDIR = REPO / "outputs/ctt_v2/captions/s4_gemini"

_lock = threading.Lock()
_done = {"n": 0, "err": 0}


def caption_one(stem: str, empirical, seed: int, round_id: int) -> dict:
    n = gd.sample_length(stem, "A", round_id, empirical, seed)
    sysp = gd.build_system_prompt("A", n, variant="v2")
    img = base64.b64encode((FRAMES / f"{stem}.jpg").read_bytes()).decode()
    body = {
        "systemInstruction": {"parts": [{"text": sysp}]},
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": img}},
            {"text": gd.USER_TEXT},
        ]}],
        "generationConfig": {
            "temperature": gd.GEN_TEMPERATURE,
            "maxOutputTokens": gd.GEN_MAX_TOKENS,
            "thinkingConfig": {"thinkingLevel": gd.GEN_THINKING_LEVEL},
        },
    }
    resp, err = gd._post(gd.GEN_MODEL, body)
    text = gd._extract_text(resp) if resp else None
    rec = {
        "clip_id": stem, "role": "A", "bank": "s4_refvfx",
        "N_target": n, "N_asked": gd.calibrate_ask(n),
        "prompt_variant": "v2", "model": gd.GEN_MODEL,
        "temperature": gd.GEN_TEMPERATURE, "error": err,
        "description": gd.postprocess(text) if text else None,
        "raw_text": text, "raw_response": resp,
    }
    with _lock:
        _done["n"] += 1
        if not rec["description"]:
            _done["err"] += 1
        if _done["n"] % 200 == 0:
            print(f"  [{_done['n']}/2000] errors {_done['err']}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test a few first")
    ap.add_argument("--resume", action="store_true",
                    help="caption only the stems that have no description yet, and MERGE into "
                         "the existing outputs. 32 workers is ~2,800 req/min and trips the "
                         "per-minute quota at ~960 calls; the 429 policy hard-stops rather "
                         "than grinding, so a resume at lower concurrency is the intended path.")
    args = ap.parse_args()

    stems = json.loads(ROSTER.read_text())["stems"]
    prior: dict[str, str] = {}
    if args.resume:
        dpath = OUTDIR / "descriptions.json"
        prior = json.loads(dpath.read_text()) if dpath.exists() else {}
        stems = [s for s in stems if s not in prior]
        print(f"[resume] {len(prior)} already captioned; {len(stems)} to go")
    if args.limit:
        stems = stems[: args.limit]
    empirical = load_length_empirical()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    recs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(caption_one, s, empirical, args.seed, args.round): s for s in stems}
        for f in as_completed(futs):
            try:
                r = f.result()
            except gd.HardStop as e:      # 429 -> hard stop, never grind
                print(f"[HARD STOP] {e}", flush=True)
                break
            except Exception as e:        # noqa: BLE001
                print(f"[error] {futs[f]}: {type(e).__name__}: {e}", flush=True)
                continue
            recs[r["clip_id"]] = r

    dt = time.time() - t0
    ok = {k: v for k, v in recs.items() if v.get("description")}
    # raw archive APPENDS on resume -- the cost ledger reads usageMetadata out of it, so a
    # truncating write would silently erase the spend record of the earlier partial run.
    mode = "a" if args.resume else "w"
    with (OUTDIR / "raw_generation_responses.jsonl").open(mode) as fh:
        for r in recs.values():
            fh.write(json.dumps(r) + "\n")
    rpath = OUTDIR / "records.json"
    merged_rec = json.loads(rpath.read_text()) if (args.resume and rpath.exists()) else {}
    merged_rec.update({f"{k}|A": {kk: vv for kk, vv in v.items() if kk != "raw_response"}
                       for k, v in ok.items()})
    rpath.write_text(json.dumps(merged_rec, indent=1))
    merged = dict(prior)
    merged.update({k: v["description"] for k, v in ok.items()})
    (OUTDIR / "descriptions.json").write_text(
        json.dumps({k: merged[k] for k in sorted(merged)}, indent=1))
    print(f"[store] descriptions.json now holds {len(merged)} captions")
    print(f"\n[done] {len(ok)}/{len(stems)} captioned in {dt:.0f}s "
          f"({len(stems) / max(dt, 1):.1f}/s, {args.workers} workers)")
    print(f"[done] failures {len(stems) - len(ok)}  -> {OUTDIR.relative_to(REPO)}")
    miss = [s for s in stems if s not in ok]
    if miss:
        print(f"[retry-needed] {len(miss)} stems: {miss[:10]}")
        (OUTDIR / "missing.json").write_text(json.dumps(miss, indent=1))


if __name__ == "__main__":
    main()
