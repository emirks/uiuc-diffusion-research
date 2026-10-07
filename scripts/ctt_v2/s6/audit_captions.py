#!/usr/bin/env python3
"""Layer-2 vision audit of the EffectData (S6) first-frame captions.

Independent per-item check that each caption actually describes ITS first frame
(catches a captioner describing the wrong / mislabeled clip while still passing
the lexical gates).

INDEPENDENCE (binding):  the captions were written by ``claude-opus-4-8 vision``
(store `generator` field), so the auditor MUST be a different family.  Auditor is
pinned to the repo's A13 Layer-2 auditor ``gemini-3.1-pro-preview`` (temp 0), the
same model/temperature/thinking-level pin used in
``scripts/ctt_v2/captions/generate_descriptions.py``.  This script REUSES that
machinery -- the HTTP `_post` retry/backoff, the model+temp+thinking pin, and the
`AuditError` discipline whereby an auditor outage is NEVER scored as a clean pass
(a missing / unparseable / out-of-schema verdict is an ERROR, retried on resume,
never counted as `match`).  The judging axis is the grounding clause (b) of that
file's ``AUDIT_QUESTION`` ("does it describe anything not visible ... or get any
visible attribute wrong"), recast into the three-way match/mismatch/borderline
verdict this audit requires.

Reads (never writes/edits) the captions.  Checkpointed + resumable.

Usage:
    source $LAB/secrets/gemini_transition.env
    OMP_NUM_THREADS=1 python audit_captions.py --workers 12
    OMP_NUM_THREADS=1 python audit_captions.py --smoke 8      # tiny dry check
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
# Paths (repo-anchored)
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json"
FRAMES = REPO / "data/processed/effectdata/first_frames"
CAP_DIR = REPO / "data/processed/effectdata/captions"
REPORT = CAP_DIR / "AUDIT_REPORT.json"

# --------------------------------------------------------------------------
# Auditor pin -- REUSED from generate_descriptions.py (A13 pin, 2026-07-28).
# --------------------------------------------------------------------------
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
PRIMARY_AUDIT_MODEL = os.environ.get("CTT_AUDIT_MODEL", "gemini-3.1-pro-preview")
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash"]  # only if primary 404s
AUDIT_TEMPERATURE = 0.0
AUDIT_MAX_TOKENS = 768  # >=512 so the JSON verdict cannot truncate; headroom for pro thinking

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["match", "mismatch", "borderline"]},
        "reason": {"type": "STRING"},
        "severity": {"type": "STRING", "enum": ["none", "minor", "major"]},
    },
    "required": ["verdict", "reason", "severity"],
}

SYSTEM_INSTRUCTION = (
    "You are an independent image-caption auditor. You are given ONE image (the first "
    "frame of a short video clip) and ONE caption that was written by a different model "
    "to describe that first frame. Your only task is to check whether the caption "
    "actually describes THIS image. You did not write the caption; judge strictly by "
    "what you can see in the image."
)

# Grounding axis reused from generate_descriptions.py AUDIT_QUESTION clause (b),
# recast into the three-way verdict this Layer-2 audit requires.
def audit_question(caption: str) -> str:
    return (
        f'Caption: "{caption}"\n\n'
        "Does this caption describe the image shown? Consider the main subject (its type: "
        "person / specific animal / object), the setting or background, and the salient "
        "visible attributes (colors, clothing, hair, count, pose, action).\n"
        '- verdict "mismatch": the caption clearly describes a DIFFERENT scene or subject '
        "than the image -- wrong subject type, wrong setting, or several gross attribute "
        "contradictions -- so it plainly belongs to a different image. This is the "
        "wrong-clip / mislabeled failure this audit exists to catch.\n"
        '- verdict "borderline": mostly correct but with a notable visible-attribute error '
        "(wrong color / count / clothing / species detail), or genuinely ambiguous.\n"
        '- verdict "match": accurately describes this image; minor omissions or harmless '
        "paraphrase are fine.\n"
        "Ignore writing style, grammar, sentence length, and any mention of motion or "
        "change (a single frame cannot show motion) -- judge ONLY visual grounding of the "
        "static content.\n"
        'severity: "major" for a mismatch, "minor" for a borderline attribute error, '
        '"none" for a match.\n'
        "Return JSON {verdict, reason, severity}. reason = one short phrase citing the "
        "specific visual evidence."
    )


# --------------------------------------------------------------------------
# HTTP -- adapted from generate_descriptions.py _post.
# Difference from the original: transient 403 ("project denied access" -- observed
# to be intermittent on this key) and 429 are treated as RETRYABLE with backoff
# rather than a global hard-stop, so one transient failure never kills the run
# (task directive).  A persistent failure exhausts retries and returns an error
# for that single item, which is then left unjudged and retried on the next resume.
# --------------------------------------------------------------------------
_lock = threading.Lock()
_counters = {"calls": 0, "retries": 0, "http429": 0, "http403": 0}
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
        if r.status_code == 404:
            return None, f"HTTP404:{r.text[:200]}"  # model-missing -> caller may fall back
        if r.status_code in _RETRY_STATUS:
            with _lock:
                if r.status_code == 429:
                    _counters["http429"] += 1
                elif r.status_code == 403:
                    _counters["http403"] += 1
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


_b64_cache: dict[str, str] = {}


def _b64(path: Path) -> str:
    p = str(path)
    v = _b64_cache.get(p)
    if v is None:
        v = base64.b64encode(path.read_bytes()).decode()
        _b64_cache[p] = v
    return v


# --------------------------------------------------------------------------
# Verdict validation -- AuditError discipline reused from generate_descriptions.py.
# An unusable verdict is an ERROR, never a default/pass.
# --------------------------------------------------------------------------
_VALID_VERDICT = {"match", "mismatch", "borderline"}
_VALID_SEVERITY = {"none", "minor", "major"}


def validate_verdict(rec: dict) -> dict:
    where = rec.get("subject")
    if rec.get("error"):
        raise ValueError(f"{where}: audit call failed: {rec['error']}")
    if rec.get("raw_response") is None:
        raise ValueError(f"{where}: audit returned no response object")
    if rec.get("parse_error") is not None:
        raise ValueError(f"{where}: audit verdict is not JSON: {rec['parse_error']!r}")
    v = rec.get("verdict_obj")
    if v is None:
        raise ValueError(f"{where}: audit returned an EMPTY verdict")
    if not isinstance(v, dict):
        raise ValueError(f"{where}: audit verdict is not an object: {v!r}")
    for f in ("verdict", "reason", "severity"):
        if f not in v:
            raise ValueError(f"{where}: verdict missing field {f!r}: {v!r}")
    if v["verdict"] not in _VALID_VERDICT:
        raise ValueError(f"{where}: verdict out of domain: {v['verdict']!r}")
    if v["severity"] not in _VALID_SEVERITY:
        raise ValueError(f"{where}: severity out of domain: {v['severity']!r}")
    return v


def audit_one(model: str, subject: str, caption: str) -> dict:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": _b64(FRAMES / f"{subject}.jpg")}},
                    {"text": audit_question(caption)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": AUDIT_TEMPERATURE,
            "maxOutputTokens": AUDIT_MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingLevel": "low" if "pro" in model else "minimal"},
        },
    }
    resp, err = _post(model, body)
    rec = {
        "subject": subject, "model": model,
        "model_version_echo": (resp or {}).get("modelVersion"),
        "error": err, "raw_response": resp,
    }
    txt = _extract_text(resp) if resp else None
    verdict_obj = None
    if txt:
        try:
            verdict_obj = json.loads(txt)
        except Exception:
            rec["parse_error"] = txt[:400]
    rec["verdict_obj"] = verdict_obj
    return rec


# --------------------------------------------------------------------------
# Provenance: subject -> source batch file (for the per-batch tripwire).
# --------------------------------------------------------------------------
def build_provenance() -> dict:
    files = ["pilot_captions.json"] + [f"out_{i:02d}.json" for i in range(25)]
    subj2batch = {}
    for f in files:
        p = CAP_DIR / f
        if not p.exists():
            continue
        for s in json.loads(p.read_text()):
            subj2batch[s] = f
    return subj2batch


# --------------------------------------------------------------------------
# Checkpoint (atomic write, resumable).
# --------------------------------------------------------------------------
def load_ckpt(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_ckpt(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def resolve_model(preferred: str) -> tuple[str, str | None]:
    """Preflight the preferred auditor; fall back only on a 404 (model missing)."""
    resp, err = _post(preferred, {
        "contents": [{"role": "user", "parts": [{"text": "ok"}]}],
        "generationConfig": {"maxOutputTokens": 8,
                             "thinkingConfig": {"thinkingLevel": "low" if "pro" in preferred else "minimal"}},
    })
    if err and err.startswith("HTTP404"):
        for fb in FALLBACK_MODELS:
            resp2, err2 = _post(fb, {
                "contents": [{"role": "user", "parts": [{"text": "ok"}]}],
                "generationConfig": {"maxOutputTokens": 8,
                                     "thinkingConfig": {"thinkingLevel": "minimal"}},
            })
            if not err2:
                return fb, f"{preferred} returned 404; fell back to {fb}"
        raise SystemExit(f"auditor {preferred} 404 and no fallback reachable")
    return preferred, None


def judge_missing(model, subjects, captions, results, ckpt_path, workers, pass_label,
                  only_subjects=None):
    """Judge every subject in `only_subjects` (default: those without a final verdict)."""
    todo = only_subjects if only_subjects is not None else [
        s for s in subjects if s not in results or not results[s].get("final_verdict")]
    if not todo:
        return
    print(f"[{pass_label}] judging {len(todo)} items with {model} ...", flush=True)
    done = 0
    errs = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(audit_one, model, s, captions[s]): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            rec = fut.result()
            try:
                v = validate_verdict(rec)
            except ValueError as e:
                errs += 1
                # leave unjudged -> retried on resume; record last error for visibility
                results.setdefault(s, {})["last_error"] = str(e)[:200]
                done += 1
                if done % 50 == 0:
                    save_ckpt(ckpt_path, results)
                    print(f"[{pass_label}] {done}/{len(todo)} (errors so far {errs})", flush=True)
                continue
            entry = results.setdefault(s, {})
            entry.pop("last_error", None)
            entry[pass_label] = {
                "verdict": v["verdict"], "reason": v["reason"], "severity": v["severity"],
                "model": model, "model_version": rec.get("model_version_echo"),
            }
            done += 1
            if done % 50 == 0:
                save_ckpt(ckpt_path, results)
                print(f"[{pass_label}] {done}/{len(todo)} (errors so far {errs})", flush=True)
    save_ckpt(ckpt_path, results)
    print(f"[{pass_label}] complete: {done} processed, {errs} errors "
          f"(counters={_counters})", flush=True)


def pct(n, d):
    return round(100.0 * n / d, 4) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=0, help="audit only the first N subjects")
    default_work = (Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/s6_audit"
                    if os.environ.get("CLAUDE_JOB_DIR") else REPO / "outputs/s6_audit")
    ap.add_argument("--workdir", default=str(default_work))
    args = ap.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY not set -- source the secrets env first")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    ckpt_path = work / ("checkpoint_smoke.json" if args.smoke else "checkpoint.json")

    store = json.loads(STORE.read_text())
    captions = {k[:-2]: v for k, v in store["descriptions"].items() if k.endswith("|A")}
    subjects = sorted(captions)
    if args.smoke:
        subjects = subjects[:args.smoke]
    subj2batch = build_provenance()

    # sanity: every subject has a frame and a batch
    miss_frame = [s for s in subjects if not (FRAMES / f"{s}.jpg").exists()]
    miss_batch = [s for s in subjects if s not in subj2batch]
    if miss_frame:
        raise SystemExit(f"{len(miss_frame)} subjects missing first frames, e.g. {miss_frame[:3]}")
    if miss_batch and not args.smoke:
        print(f"WARN: {len(miss_batch)} subjects with no batch provenance, e.g. {miss_batch[:3]}")

    model, sub_note = resolve_model(PRIMARY_AUDIT_MODEL)
    print(f"auditor={model}  temp={AUDIT_TEMPERATURE}  n_subjects={len(subjects)}"
          + (f"  [{sub_note}]" if sub_note else ""), flush=True)

    results = load_ckpt(ckpt_path)

    # ---- Pass 1: judge everyone missing a pass1 verdict ----
    to_p1 = [s for s in subjects if s not in results or "pass1" not in results[s]]
    judge_missing(model, subjects, captions, results, ckpt_path, args.workers, "pass1",
                  only_subjects=to_p1)

    # Retry any pass-1 items that errored (no pass1 recorded yet) up to a couple rounds
    for rnd in range(2):
        still = [s for s in subjects if "pass1" not in results.get(s, {})]
        if not still:
            break
        print(f"[pass1-retry {rnd}] {len(still)} still unjudged", flush=True)
        judge_missing(model, subjects, captions, results, ckpt_path, args.workers,
                      "pass1", only_subjects=still)

    # Set a provisional final verdict from pass1 for all non-borderline
    for s in subjects:
        e = results.get(s, {})
        if "pass1" in e:
            e["final_verdict"] = e["pass1"]["verdict"]
            e["final_reason"] = e["pass1"]["reason"]
            e["final_severity"] = e["pass1"]["severity"]
            e["final_pass"] = "pass1"
    save_ckpt(ckpt_path, results)

    # ---- Pass 2 (adjudication): re-judge every borderline once; keep the 2nd verdict ----
    borderline = [s for s in subjects
                  if results.get(s, {}).get("pass1", {}).get("verdict") == "borderline"]
    print(f"[pass2] {len(borderline)} borderline items to re-judge", flush=True)
    to_p2 = [s for s in borderline if "pass2" not in results.get(s, {})]
    judge_missing(model, subjects, captions, results, ckpt_path, args.workers, "pass2",
                  only_subjects=to_p2)
    for rnd in range(2):
        still = [s for s in borderline if "pass2" not in results.get(s, {})]
        if not still:
            break
        judge_missing(model, subjects, captions, results, ckpt_path, args.workers,
                      "pass2", only_subjects=still)
    # keep the second verdict for borderlines
    for s in borderline:
        e = results.get(s, {})
        if "pass2" in e:
            e["final_verdict"] = e["pass2"]["verdict"]
            e["final_reason"] = e["pass2"]["reason"]
            e["final_severity"] = e["pass2"]["severity"]
            e["final_pass"] = "pass2"
    save_ckpt(ckpt_path, results)

    # ---- Tally ----
    judged = [s for s in subjects if results.get(s, {}).get("final_verdict")]
    unjudged = [s for s in subjects if not results.get(s, {}).get("final_verdict")]
    n = len(judged)
    counts = {"match": 0, "mismatch": 0, "borderline": 0}
    for s in judged:
        counts[results[s]["final_verdict"]] += 1

    mism_rate = pct(counts["mismatch"], n)
    bord_rate = pct(counts["borderline"], n)

    mismatched = []
    for s in judged:
        if results[s]["final_verdict"] == "mismatch":
            mismatched.append({
                "subject": s, "batch": subj2batch.get(s, "UNKNOWN"),
                "caption": captions[s], "reason": results[s]["final_reason"],
                "severity": results[s]["final_severity"],
                "pass1_verdict": results[s].get("pass1", {}).get("verdict"),
            })
    mismatched.sort(key=lambda d: (d["batch"], d["subject"]))

    # per-batch mismatch tally
    per_batch = {}
    for s in judged:
        b = subj2batch.get(s, "UNKNOWN")
        d = per_batch.setdefault(b, {"judged": 0, "mismatch": 0, "borderline": 0})
        d["judged"] += 1
        fv = results[s]["final_verdict"]
        if fv in ("mismatch", "borderline"):
            d[fv] += 1
    flagged_batches = sorted(b for b, d in per_batch.items() if d["mismatch"] >= 3)

    # ---- Length histogram (word counts) over the full caption corpus ----
    wc = sorted(len(captions[s].split()) for s in subjects)

    def q(p):
        if not wc:
            return None
        i = min(len(wc) - 1, max(0, int(round((p / 100.0) * (len(wc) - 1)))))
        return wc[i]

    hist = {"n": len(wc), "min": wc[0] if wc else None, "max": wc[-1] if wc else None,
            "p10": q(10), "p50": q(50), "p90": q(90),
            "mean": round(sum(wc) / len(wc), 2) if wc else None}

    # ---- Tripwire evaluations (PRE-REGISTERED) ----
    tw1 = {"name": "overall consensus-mismatch rate", "bar": "<= 2%",
           "value_pct": mism_rate, "n_mismatch": counts["mismatch"], "n_judged": n,
           "pass": mism_rate <= 2.0}
    tw2 = {"name": "any source batch with >= 3 mismatches -> flag batch",
           "bar": "0 batches with >=3 mismatches", "flagged_batches": flagged_batches,
           "per_batch_mismatch": {b: per_batch[b]["mismatch"] for b in sorted(per_batch)},
           "pass": len(flagged_batches) == 0}

    overall_pass = tw1["pass"] and tw2["pass"] and not unjudged
    verdict_line = ("PASS" if overall_pass else "FAIL-regenerate") + (
        "" if not unjudged else f" (INCOMPLETE: {len(unjudged)} unjudged)")

    report = {
        "task": "Layer-2 vision audit of EffectData (S6) first-frame captions",
        "generated_by_model": store.get("generator", "").split(",")[0],
        "independence": "generator=claude (Claude family); auditor=Gemini family (required)",
        "auditor_model": model,
        "auditor_fallback_note": sub_note,
        "auditor_temperature": AUDIT_TEMPERATURE,
        "auditor_thinking_level": "low" if "pro" in model else "minimal",
        "auditor_max_output_tokens": AUDIT_MAX_TOKENS,
        "auditor_machinery_reused_from": "scripts/ctt_v2/captions/generate_descriptions.py "
            "(_post retry/backoff, model+temp+thinking pin, AuditError no-default-verdict rule)",
        "procedure": {
            "pass1": "judge every subject once (image + caption -> match/mismatch/borderline)",
            "pass2": "adjudication -- re-judge every pass1=borderline once more; keep 2nd verdict",
            "consensus_mismatch": "final_verdict==mismatch after adjudication; pass1 mismatches "
                "are taken as final (only borderlines are re-judged, per the pre-registered spec)",
            "error_discipline": "a missing/unparseable/out-of-schema verdict is an ERROR, never "
                "counted as match; retried on resume",
        },
        "n_subjects": len(subjects),
        "n_judged": n,
        "n_unjudged": len(unjudged),
        "unjudged_subjects": unjudged,
        "verdict_counts": counts,
        "mismatch_count": counts["mismatch"],
        "mismatch_rate_pct": mism_rate,
        "borderline_count": counts["borderline"],
        "borderline_rate_pct": bord_rate,
        "tripwire_1_overall_mismatch": tw1,
        "tripwire_2_per_batch": tw2,
        "length_histogram_words": hist,
        "mismatched_subjects": mismatched,
        "per_batch_summary": {b: per_batch[b] for b in sorted(per_batch)},
        "VERDICT": verdict_line,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    print(f"auditor={model} temp={AUDIT_TEMPERATURE}")
    print(f"N judged={n}/{len(subjects)}  unjudged={len(unjudged)}")
    print(f"verdicts={counts}")
    print(f"mismatch rate={mism_rate}% vs bar 2%  -> tripwire1 {'PASS' if tw1['pass'] else 'FAIL'}")
    print(f"batches flagged (>=3 mism)={flagged_batches} -> tripwire2 {'PASS' if tw2['pass'] else 'FAIL'}")
    print(f"length words p10/p50/p90 = {hist['p10']}/{hist['p50']}/{hist['p90']} "
          f"(min {hist['min']} max {hist['max']} mean {hist['mean']})")
    print(f"VERDICT: {verdict_line}")
    print(f"report -> {REPORT}")
    if args.smoke:
        print("\n[smoke] per-item:")
        for s in subjects:
            e = results.get(s, {})
            print(f"  {s} [{subj2batch.get(s,'?')}] {e.get('final_verdict')} "
                  f"({e.get('final_severity')}): {str(e.get('final_reason'))[:90]}")


if __name__ == "__main__":
    main()
