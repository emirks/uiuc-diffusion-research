#!/usr/bin/env python
"""grid v3 — EFFECT_DESC clauses for every NEW reference, the Lane-B effect text (TEXT_LIFECYCLE §3.2).

Same instrument, prompt and house style as misc/refvfx_baseline/build_ref_effects.py (which produced the 36
clauses of store/captions/003_effect_clauses): gemini-3.6-flash watches the reference's STANDARD clip and
returns one subject-generic clause, 12-35 words, no trailing period. Idempotent and append-only into
misc/refvfx_baseline/reference_effects.json — an existing clause is never re-rolled, so no prompt an earlier
family used can change. Raw responses archived beside the originals (misc/refvfx_baseline/raw_ref_effects/).

References are read from the v3 registries (renderable + pending); a reference whose standard clip does not
exist yet is skipped and reported (re-run after the media lane lands).

Run:  source $LAB/secrets/gemini_transition.env; python scripts/grid_v3/build_ref_effects_v3.py [--workers 6]
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
STD = REPO / "data/processed/transitions_std121"
DST = REPO / "misc/refvfx_baseline/reference_effects.json"
RAW = REPO / "misc/refvfx_baseline/raw_ref_effects"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = "gemini-3.6-flash"
sys.path.insert(0, str(REPO / "misc/refvfx_baseline"))
sys.path.insert(0, str(REPO / "eval_ladder"))

# VERBATIM from build_ref_effects.py (the 003 shelf's producer) — never edited here.
SYSTEM = """You are writing the text prompt for RefVFX, a reference-based video visual-effect \
transfer model. You will watch a REFERENCE video that demonstrates one temporal visual effect.

Write ONE clause describing ONLY the effect as it unfolds over time, so that it can be slotted \
into this sentence:

    "Make it so that the beginning of the scene is unchanged, but during the video <YOUR CLAUSE>."

Hard requirements:
- Describe ONLY the transformation/effect. Never describe who the subject is, what they wear, \
the room, the background, the lighting setup, or any detail specific to THIS clip. The effect \
will be applied to a completely different scene.
- Refer to the subject generically: "the subject", "the person", or "the scene".
- Present tense, one clause, 12-35 words. No trailing period.
- Do not invent a name or trigger word for the effect.
- Describe the visible progression (how it starts, spreads, resolves), not a label.

Return STRICT JSON: {"effect": "<clause>"}"""


def std_path(clip: str) -> Path | None:
    hits = sorted(STD.glob(f"*/{clip}.mp4"))
    return hits[0] if hits else None


def post(body: dict, tries: int = 5):
    key = os.environ["GEMINI_API_KEY"]
    url = f"{API_ROOT}/{MODEL}:generateContent"
    last = None
    for a in range(tries):
        try:
            r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=240)
        except Exception as e:
            last = f"EXC:{type(e).__name__}"
            time.sleep(min(2 ** a, 20) + random.random())
            continue
        if r.status_code == 200:
            return r.json(), None
        if r.status_code >= 500 or r.status_code in (408, 409, 429):
            last = f"HTTP{r.status_code}"
            time.sleep(min(2 ** a, 20) + random.random())
            continue
        return None, f"HTTP{r.status_code}:{r.text[:300]}"
    return None, f"exhausted:{last}"


def one(clip: str) -> tuple[str, str | None, str]:
    path = std_path(clip)
    if path is None:
        return clip, None, "no_std_clip"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(path.read_bytes()).decode()}},
            {"text": "Describe the effect this reference video demonstrates."},
        ]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    resp, err = post(body)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{clip}.json").write_text(json.dumps({"clip": clip, "err": err, "resp": resp, "model": MODEL,
                                                  "std_clip": str(path.relative_to(REPO)), "grid": "v3"}, indent=1))
    if err:
        return clip, None, err
    try:
        text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
        effect = json.loads(text)["effect"].strip().rstrip(".").strip()
    except Exception as e:
        return clip, None, f"parse:{type(e).__name__}"
    return clip, effect, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    refs: set[str] = set()
    for f in ("registry_v3.jsonl", "registry_v3_pending.jsonl"):
        for l in (REPO / "eval_ladder" / f).read_text().splitlines():
            r = json.loads(l)
            if r.get("reference") and r["arm"] == "ic_gen":
                refs.add(r["reference"])
    out = json.loads(DST.read_text()) if DST.exists() else {}
    todo = sorted(c for c in refs if c not in out)
    ready = [c for c in todo if std_path(c) is not None]
    print(f"{len(refs)} references in v3; {len(out)} already have a clause; {len(todo)} to do, {len(ready)} with a standard clip on disk -> {MODEL}")
    if args.dry_run or not ready:
        print("skipped (no standard clip yet):", [c for c in todo if c not in ready][:10])
        return
    fails = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for clip, effect, status in ex.map(one, ready):
            if effect is None:
                fails.append((clip, status))
                print(f"  FAIL {clip}: {status}")
            else:
                out[clip] = effect
                print(f"  {clip:40s} {effect}")
    DST.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")     # append-only: successes land, failures re-run
    wc = sorted(len(v.split()) for v in out.values())
    print(f"\nwrote {DST.relative_to(REPO)}  n={len(out)}  words p10/p50/p90 = {wc[len(wc)//10]}/{wc[len(wc)//2]}/{wc[-max(1, len(wc)//10)]}"
          f"  | failed {len(fails)} | still without a standard clip {len(todo) - len(ready)}")


if __name__ == "__main__":
    main()
