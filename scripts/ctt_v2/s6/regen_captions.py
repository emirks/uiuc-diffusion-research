#!/usr/bin/env python3
"""Regenerate ALL EffectData (S6) first-frame captions -- ONE IMAGE PER MODEL CALL.

Why this exists
---------------
The current S6 captions (store/captions/004_effectdata) were written by
claude-opus-4-8 agents in 80-image BATCHES, and a Layer-2 vision audit proved
SYSTEMATIC caption<->subject MISALIGNMENT (e.g. batch_14 subjects
0013de12117b8c30 and 1ffc4d7b5990890f had their captions SWAPPED).  The batch
method loses the key<->image binding.

The fix is STRUCTURAL: each caption is produced in a Gemini call that contains
EXACTLY ONE image, and is written keyed to that exact subject.  With one image
per call, misalignment is impossible.

Method (reuses the proven S4 machinery -- NOT reinvented)
--------------------------------------------------------
* Generator model / decode config: gemini-3.6-flash, temp 0.7, maxOutputTokens
  120, thinkingLevel "minimal" -- imported verbatim from
  ``scripts/ctt_v2/captions/generate_descriptions.py`` (GEN_MODEL, GEN_*).
* Prompt: the v2-s4f0 system prompt = ``build_system_prompt("A", N, "v2")``
  (role A, the S4 role-A v2 prompt the audited captions and the S4 store used),
  adapted from the "9-frame snippet" register to the "single still frame"
  register by a small set of ASSERTED string replacements (see
  ``build_still_frame_prompt``).  Every replacement must fire or the module
  raises -- a prompt drift can never pass silently.
* Length: per-subject deterministic draw from the 171-value corpus empirical
  (``sample_length``), calibrated inside ``build_system_prompt`` via
  ``calibrate_ask`` exactly as the S4 store did.  Seeded on the subject id so
  it is reproducible and resume-stable.
* Image: the FIRST FRAME jpg is attached as inline_data image/jpeg -- ONE
  subject per call, never a batch.

Robustness
----------
* ThreadPool (~8), exponential backoff on 429/403/5xx.  The 429 handling is the
  TOLERANT variant from ``s6/audit_captions.py`` (retry with backoff), NOT the
  generator's global hard-stop -- a concurrent pro-tier AUDIT job may be sharing
  this key and we must ride out its rate limits rather than crash.
* Checkpointed + resumable: subjects with an accepted caption are skipped on
  restart.  Working files under $CLAUDE_JOB_DIR/tmp/s6_regen/.
* Sanity pass mirrors ``s6/build_caption_store.py`` HARD_LEAK + format checks
  (single sentence, ends with a period, no transformation/effect/frame/sound
  leak words) plus a 12..56 word band; a failing caption is re-drawn ONCE with
  a fresh length draw.

Output
------
* data/processed/effectdata/captions/regen/gemini_captions.json  {subject: caption}
  -- a NEW file.  Does NOT touch the store, the old out_*.json, or conditions.

Usage
-----
  source $LAB/secrets/gemini_transition.env
  OMP_NUM_THREADS=1 python scripts/ctt_v2/s6/regen_captions.py --workers 8
  OMP_NUM_THREADS=1 python scripts/ctt_v2/s6/regen_captions.py --limit 5   # smoke
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Repo-anchored paths
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json"
FRAMES = REPO / "data/processed/effectdata/first_frames"
OUT_DIR = REPO / "data/processed/effectdata/captions/regen"
OUT_JSON = OUT_DIR / "gemini_captions.json"
OUT_META = OUT_DIR / "regen_meta.json"
LENGTH_EMPIRICAL = REPO / "misc/ctt_v2_final/M1_length_empirical.json"

# --------------------------------------------------------------------------
# Reuse the proven generator machinery (imports, no /projects coupling used).
# We deliberately do NOT import generate_descriptions._post -- it treats HTTP 429
# as a GLOBAL hard-stop, and this job must tolerate a concurrent audit job's rate
# limits.  Everything else (prompt, length, decode config) is imported verbatim.
# --------------------------------------------------------------------------
sys.path.insert(0, str(REPO / "scripts/ctt_v2/captions"))
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
sys.path.insert(0, str(REPO / "scripts/ctt_v2/s6"))
from generate_descriptions import (  # noqa: E402
    GEN_MODEL, GEN_TEMPERATURE, GEN_MAX_TOKENS, GEN_THINKING_LEVEL,
    build_system_prompt, calibrate_ask, sample_length,
)
from build_caption_store import HARD_LEAK  # noqa: E402  authoritative store leak gate

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
SEED = 42
USER_TEXT_IMG = "Describe this still frame."
WORD_LO, WORD_HI = 12, 56  # re-draw band (task-specified; store gate is 10..56)

# --------------------------------------------------------------------------
# v2-s4f0 "single still frame" prompt.
# Start from the exact v2 role-A prompt (build_system_prompt), then convert the
# "9-frame snippet / footage" register to "single still frame".  Each replacement
# is ASSERTED to fire so a silent prompt drift is impossible.
# --------------------------------------------------------------------------
_SNIPPET_TO_FRAME = [
    ("short video snippets for a film-production shot list",
     "still frames for a film-production shot list"),
    ("You will receive a 9-frame snippet of ordinary footage.",
     "You will receive a single still frame of ordinary footage."),
    ("describing only what is visible in the snippet.",
     "describing only what is visible in the frame."),
    ("Mention the camera only if the snippet itself shows",
     "Mention the camera only if the frame itself shows"),
    ("do not mention anything that happens before or after the snippet,",
     "do not mention anything that happens before or after this frame,"),
    ("Do not refer to the video, image, frames, snippet, or footage as objects.",
     "Do not refer to the image, frame, or footage as objects."),
    ("but describe only what is in your own snippet:",
     "but describe only what is in your own frame:"),
]


def build_still_frame_prompt(n: int) -> str:
    """The v2-s4f0 role-A prompt, re-registered for a single still frame."""
    p = build_system_prompt("A", n, variant="v2")
    for old, new in _SNIPPET_TO_FRAME:
        assert old in p, f"prompt drift: expected substring not found: {old!r}"
        p = p.replace(old, new)
    assert "snippet" not in p, "prompt still references 'snippet' after adaptation"
    return p


# --------------------------------------------------------------------------
# HTTP -- tolerant _post adapted from s6/audit_captions.py (429/403/5xx retryable).
# --------------------------------------------------------------------------
_lock = threading.Lock()
_counters = {"calls": 0, "retries": 0, "http429": 0, "http403": 0, "http5xx": 0}
_RETRY_STATUS = {403, 408, 409, 425, 429, 500, 502, 503, 504}


def _post(model: str, body: dict, timeout: int = 240, max_tries: int = 8):
    key = os.environ["GEMINI_API_KEY"]
    url = f"{API_ROOT}/{model}:generateContent"
    last = None
    for attempt in range(max_tries):
        try:
            r = requests.post(
                url,
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
        except Exception as e:  # transient network
            last = f"EXC:{type(e).__name__}:{e}"
            with _lock:
                _counters["retries"] += 1
            time.sleep(min(2 ** attempt, 30) + random.random())
            continue
        with _lock:
            _counters["calls"] += 1
        if r.status_code == 200:
            return r.json(), None
        if r.status_code in _RETRY_STATUS:
            with _lock:
                if r.status_code == 429:
                    _counters["http429"] += 1
                elif r.status_code == 403:
                    _counters["http403"] += 1
                elif r.status_code >= 500:
                    _counters["http5xx"] += 1
                _counters["retries"] += 1
            last = f"HTTP{r.status_code}:{r.text[:160]}"
            time.sleep(min(2 ** attempt, 30) + random.random())
            continue
        return None, f"HTTP{r.status_code}:{r.text[:300]}"
    return None, f"exhausted_retries:{last}"


def _extract_text(resp: dict):
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        return None


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# --------------------------------------------------------------------------
# Normalisation + sanity pass (mirrors build_caption_store.py HARD_LEAK/format).
# Captions are kept WITH their trailing period (out_*.json convention -- the store
# rebuild validates endswith('.') ), unlike the S4 store's period-stripped form.
# --------------------------------------------------------------------------
def normalize(raw: str | None) -> str:
    return (raw or "").strip().strip("﻿").strip()


def sanity_fails(cap: str) -> list[str]:
    f: list[str] = []
    if not cap:
        return ["empty"]
    if "\n" in cap:
        f.append("newline")
    if cap.startswith(("-", "*", '"', "`", "'")):
        f.append("markup_start")
    if any(ch in cap for ch in ("*", "`", "#", "[", "]", "<", ">")):
        f.append("markup_char")
    if not cap[0].isupper():
        f.append("not_uppercase_initial")
    if not cap.endswith("."):
        f.append("no_terminal_period")
    body = cap[:-1] if cap.endswith(".") else cap
    if ". " in body or "! " in body or "? " in body:
        f.append("multi_sentence")
    wc = len(cap.split())
    if wc < WORD_LO or wc > WORD_HI:
        f.append(f"word_band:{wc}")
    m = HARD_LEAK.search(cap)
    if m:
        f.append(f"hard_leak:{m.group(0).strip()}")
    return f


# --------------------------------------------------------------------------
# Per-subject generation: 1 image -> 1 call, 1 re-draw on sanity/format failure.
# --------------------------------------------------------------------------
def gen_call(subject: str, n: int):
    body = {
        "systemInstruction": {"parts": [{"text": build_still_frame_prompt(n)}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": _b64(FRAMES / f"{subject}.jpg")}},
                {"text": USER_TEXT_IMG},
            ],
        }],
        "generationConfig": {
            "temperature": GEN_TEMPERATURE,
            "maxOutputTokens": GEN_MAX_TOKENS,
            "thinkingConfig": {"thinkingLevel": GEN_THINKING_LEVEL},
        },
    }
    resp, err = _post(GEN_MODEL, body)
    text = _extract_text(resp) if resp else None
    return text, err, (resp or {}).get("modelVersion")


def process_subject(subject: str, empirical: list[int]) -> dict:
    history: list[dict] = []
    best: dict | None = None
    for attempt in (1, 2):
        n = sample_length(subject, "A", attempt, empirical, SEED)
        text, err, mver = gen_call(subject, n)
        if err:
            history.append({"attempt": attempt, "n_target": n, "error": err})
            continue
        cap = normalize(text)
        fails = sanity_fails(cap)
        rec = {"attempt": attempt, "n_target": n, "n_asked": calibrate_ask(n),
               "raw_text": text, "caption": cap, "words": len(cap.split()) if cap else 0,
               "fails": fails, "model_version": mver}
        history.append(rec)
        best = rec
        if not fails:
            return {"subject": subject, "caption": cap, "n_target": n,
                    "n_asked": calibrate_ask(n), "accepted_attempt": attempt,
                    "words": rec["words"], "redrawn": attempt > 1, "sanity_ok": True,
                    "model_version": mver, "history": history}
    if best and best.get("caption"):
        return {"subject": subject, "caption": best["caption"], "n_target": best["n_target"],
                "n_asked": best["n_asked"], "accepted_attempt": best["attempt"],
                "words": best["words"], "redrawn": True, "sanity_ok": False,
                "sanity_fails": best["fails"], "model_version": best.get("model_version"),
                "history": history}
    return {"subject": subject, "unfinished": True,
            "error": "; ".join(h.get("error", "") for h in history if h.get("error"))[:250],
            "history": history}


# --------------------------------------------------------------------------
# Checkpoint (atomic, resumable)
# --------------------------------------------------------------------------
def load_ckpt(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def save_ckpt(p: Path, data: dict):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def pctl(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round((q / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N subjects")
    default_work = (Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/s6_regen"
                    if os.environ.get("CLAUDE_JOB_DIR") else REPO / "outputs/s6_regen")
    ap.add_argument("--workdir", default=str(default_work))
    args = ap.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY not set -- source $LAB/secrets/gemini_transition.env")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    ckpt_path = work / ("checkpoint_smoke.json" if args.limit else "checkpoint.json")

    store = json.loads(STORE.read_text())
    subjects = sorted(k[:-2] for k in store["descriptions"] if k.endswith("|A"))
    assert len(subjects) == len(set(subjects))
    if not args.limit:
        assert len(subjects) == 2000, f"expected 2000 subjects, got {len(subjects)}"
    if args.limit:
        subjects = subjects[:args.limit]

    miss = [s for s in subjects if not (FRAMES / f"{s}.jpg").exists()]
    if miss:
        raise SystemExit(f"{len(miss)} subjects missing first frames, e.g. {miss[:5]}")

    empirical = json.loads(LENGTH_EMPIRICAL.read_text())
    assert isinstance(empirical, list) and empirical, "length empirical must be a non-empty list"

    results = load_ckpt(ckpt_path)
    todo = [s for s in subjects if not results.get(s, {}).get("caption")]
    print(f"subjects={len(subjects)}  already_done={len(subjects)-len(todo)}  todo={len(todo)}",
          flush=True)
    print(f"generator={GEN_MODEL} temp={GEN_TEMPERATURE} maxtok={GEN_MAX_TOKENS} "
          f"thinking={GEN_THINKING_LEVEL}  workers={args.workers}  seed={SEED}", flush=True)

    t0 = time.time()
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_subject, s, empirical): s for s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # never let one subject kill the pool
                    r = {"subject": s, "unfinished": True, "error": f"EXC:{type(e).__name__}:{e}"}
                results[s] = r
                done += 1
                if done % 50 == 0:
                    save_ckpt(ckpt_path, results)
                    ok = sum(1 for v in results.values() if v.get("caption"))
                    print(f"  {done}/{len(todo)}  accepted_total={ok}  "
                          f"{time.time()-t0:.0f}s  counters={_counters}", flush=True)
    save_ckpt(ckpt_path, results)

    # ---- assemble output {subject: caption} ----
    captions = {s: results[s]["caption"] for s in subjects if results.get(s, {}).get("caption")}
    unfinished = [s for s in subjects if s not in captions]
    redrawn = [s for s in subjects if results.get(s, {}).get("redrawn")]
    sanity_bad = [s for s in subjects
                  if results.get(s, {}).get("caption") and not results[s].get("sanity_ok")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(sorted(captions.items())), indent=1, ensure_ascii=True) + "\n")

    wc = sorted(len(c.split()) for c in captions.values())
    meta = {
        "task": "S6 EffectData first-frame captions, regenerated ONE IMAGE PER CALL",
        "generator_model": GEN_MODEL,
        "generator_model_versions": sorted({results[s].get("model_version") for s in captions
                                            if results[s].get("model_version")}),
        "gen_temperature": GEN_TEMPERATURE, "gen_max_output_tokens": GEN_MAX_TOKENS,
        "gen_thinking_level": GEN_THINKING_LEVEL,
        "prompt_variant": "v2-s4f0 (build_system_prompt role A variant v2, "
                          "re-registered to 'single still frame')",
        "length_source": "misc/ctt_v2_final/M1_length_empirical.json (171 vals), "
                          "sample_length seeded per-subject, calibrate_ask applied",
        "seed": SEED,
        "n_subjects": len(subjects),
        "n_captioned": len(captions),
        "n_unfinished": len(unfinished),
        "unfinished_subjects": unfinished,
        "n_redrawn": len(redrawn),
        "n_sanity_fail_after_redraw": len(sanity_bad),
        "sanity_fail_subjects": {s: results[s].get("sanity_fails") for s in sanity_bad},
        "length_words": {"n": len(wc), "min": wc[0] if wc else None, "max": wc[-1] if wc else None,
                         "p10": pctl(wc, 10), "p50": pctl(wc, 50), "p90": pctl(wc, 90),
                         "mean": round(sum(wc) / len(wc), 2) if wc else None},
        "api_counters": _counters,
        "wall_seconds": round(time.time() - t0, 1),
        "output_path": str(OUT_JSON),
        "checkpoint": str(ckpt_path),
    }
    OUT_META.write_text(json.dumps(meta, indent=1) + "\n")

    print("\n=== SUMMARY ===")
    print(f"captioned {len(captions)}/{len(subjects)}  unfinished={len(unfinished)}  "
          f"redrawn={len(redrawn)}  sanity_fail_after_redraw={len(sanity_bad)}")
    print(f"length words p10/p50/p90 = {meta['length_words']['p10']}/"
          f"{meta['length_words']['p50']}/{meta['length_words']['p90']} "
          f"(min {meta['length_words']['min']} max {meta['length_words']['max']} "
          f"mean {meta['length_words']['mean']})")
    print(f"counters={_counters}")
    if unfinished:
        print(f"UNFINISHED (re-run to resume): {unfinished[:10]}")
    if sanity_bad:
        print(f"SANITY-FAIL kept (flagged): "
              f"{[(s, results[s].get('sanity_fails')) for s in sanity_bad[:10]]}")
    print(f"out -> {OUT_JSON}")
    print(f"meta -> {OUT_META}")


if __name__ == "__main__":
    main()
