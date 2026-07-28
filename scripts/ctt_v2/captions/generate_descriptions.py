#!/usr/bin/env python
"""CTT v2 per-(clip, role) description generator -- advisor A4 (round 8) Q1 + Q3.

Produces a description store `{clip_id: {"A": "...", "B": "..."}}` from 9-frame
byte-pure anchor VIDEOS (never the full clip, never a transition frame), runs the
A4 Layer-1 tiered lexical filter and the A4 Layer-2 independent-model semantic
audit, and archives every raw API response.

AUDITOR PIN (advisor A13, 2026-07-28): the Layer-2 auditor is
`gemini-3.1-pro-preview`.  A4 named a PRO-tier auditor and made
generator/auditor INDEPENDENCE the binding requirement; the `gemini-3.5-flash`
substitution was never a preference but a forced deviation recorded under a
factual error -- DOSSIER §5.1 read the pro tier as rate-limited when
`gemini-3-pro-preview` is in fact RETIRED (404), and `gemini-3.1-pro-preview` is
live.  Promoting it therefore REMOVES a deviation and reverts toward the written
spec, and it is strictly more independent than 3.5-flash (pro family, vs the
3.6-flash generator's own family).  A13 also pins the no-switch-back rule: if
3.5-flash returns, the auditor does NOT revert -- one recorded instrument change.
First-pass rates either side of this change are NOT a like-for-like series.

THINKING BUDGET (measured): with the default thinking level, gemini-3.x flash
spends ~111 "thoughts" tokens before emitting text, so A4's
`max_output_tokens=120` truncates the sentence mid-word.  The GENERATOR therefore
sets `thinkingConfig.thinkingLevel = "minimal"`, which makes the 120-token budget
apply to the visible text as A4 intended.  `thinkingBudget: 0` is rejected
(HTTP 400) by these models.  The pro-tier AUDITOR **rejects `"minimal"`** and is
pinned at `"low"` with a 512-token cap, so the JSON verdict can never truncate.

Usage
-----
  source /projects/illinois/eng/cs/jrehg/users/emirkisa/secrets/gemini_transition.env
  PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python

  # deterministic per-(bank x role) sample -- the M3 pilot
  $PY generate_descriptions.py --sample-per-cell 100 --seed 42 \
      --out outputs/ctt_v2/captions/pilot

  # explicit pair list: JSON [["clip_id","A"], ["clip_id","B"], ...]
  $PY generate_descriptions.py --pairs pairs.json --out <dir>
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption_common import (  # noqa: E402
    STRIPS_INDEX,
    LeakFilter,
    format_violations,
    load_length_empirical,
    word_count,
)
from root_common import load_caption_store_exclusions  # noqa: E402

# --------------------------------------------------------------------------
# Pinned model identity
# --------------------------------------------------------------------------
GEN_MODEL = "gemini-3.6-flash"
GEN_TEMPERATURE = 0.7
GEN_MAX_TOKENS = 120
GEN_THINKING_LEVEL = "minimal"

# --- A13 pinned auditor config; change only by a recorded advisor ruling ---
AUDIT_MODEL = "gemini-3.1-pro-preview"    # A4's pro tier, restored (A13 Q1)
AUDIT_TEMPERATURE = 0.0
AUDIT_MAX_TOKENS = 512                    # >= 512 so the JSON verdict cannot truncate
AUDIT_THINKING_LEVEL = "low"              # the pro tier REJECTS "minimal"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# --------------------------------------------------------------------------
# A4 Q1 prompts -- VERBATIM
# --------------------------------------------------------------------------
_PROMPT_A_TEMPLATE = (
    "You write one-sentence descriptions of short video snippets for a film-production "
    "shot list. You will receive a 9-frame snippet of ordinary footage. Write exactly ONE "
    "English sentence describing only what is visible in the snippet. Begin with the main "
    'subject and its appearance (e.g. "A woman with…", "A young man in…"), then the '
    "subject's action using simple-present or present-progressive verbs, then the setting, "
    'lighting, and colors, using plain literal terms ("red dress", not "vibrant red dress"). '
    "Mention the camera only if the snippet itself shows an obvious camera viewpoint (e.g. an "
    "overhead view). Describe only this moment: do not mention anything that happens before "
    "or after the snippet, and do not use any language about the scene changing, transforming, "
    "beginning, ending, shifting, or revealing anything. Do not mention sounds, music, or "
    "speech. Do not refer to the video, image, frames, snippet, or footage as objects. Do not "
    "name any visual effect, editing technique, or animation style. Aim for about {N} words "
    "(anywhere from {NLO} to {NHI} words is fine). Output only the sentence, ending with a "
    "period — no quotes, no markdown, no preamble."
)

# A4 Q1: "identical system prompt except the second and third sentences are
# replaced by [the noun-phrase instruction] and the final line reads
# 'Output only the phrase, ending with a period.'"
_PROMPT_B_TEMPLATE = (
    "You write one-sentence descriptions of short video snippets for a film-production "
    "shot list. You will receive a 9-frame snippet of ordinary footage. Write exactly ONE "
    "lowercase English noun phrase (not a complete sentence) describing only what is visible "
    'in the snippet, in the style: "a woman with long dark hair in a gray coat sipping coffee '
    'beside a rain-streaked window". Begin with "a", "an", or a plural noun phrase; express '
    "the action with -ing participles, never a standalone conjugated verb; then setting, "
    "lighting, and colors in plain literal terms. "
    "Mention the camera only if the snippet itself shows an obvious camera viewpoint (e.g. an "
    "overhead view). Describe only this moment: do not mention anything that happens before "
    "or after the snippet, and do not use any language about the scene changing, transforming, "
    "beginning, ending, shifting, or revealing anything. Do not mention sounds, music, or "
    "speech. Do not refer to the video, image, frames, snippet, or footage as objects. Do not "
    "name any visual effect, editing technique, or animation style. Aim for about {N} words "
    "(anywhere from {NLO} to {NHI} words is fine). Output only the phrase, ending with a "
    "period — no quotes, no markdown, no preamble."
)

# --------------------------------------------------------------------------
# Prompt variant "v2" -- OPERATOR DIAGNOSTIC, NOT A4's text.
#
# Round 1 of the M3 pilot (400 descriptions, A4's verbatim prompts) FAILED gate
# #8 at balanced accuracy 0.7139 (bar <= 0.65).  The discriminative features were
# pure style, not content:
#
#   statistic            corpus   round-1   delta
#   commas/description    1.88     0.78     -1.10   (top corpus-side feature PUNCT[,])
#   "while" rate         11.7%    27.9%    +16.2pp  (top new-side feature)
#   be-verb rate          8.8%     0.5%     -8.3pp
#   words p50              33       29        -4
#   colour terms/desc     3.158    2.309    -0.85
#
# A4 Q4's pre-registered failure action is "fix the prompt or the length sampler
# and regenerate the entire stratum ... max 3 regeneration rounds".  v2 is that
# fix, kept OPT-IN so A4's verbatim prompt remains the default and round 1 stays
# reproducible.  The delta is deliberately minimal and structural only:
#   (1) the invented style exemplar is replaced by a REAL corpus description
#       (A4's own thesis is instrument matching; A4's B exemplar has 0 commas,
#       which alone explains most of the comma deficit);
#   (2) an explicit appositive-comma instruction;
#   (3) an explicit plain-colour instruction.
# No banned-lexeme list is added -- A4 Q1's pink-elephant rule is respected, so
# "while" is displaced structurally rather than named.
# --------------------------------------------------------------------------
CORPUS_EXEMPLAR_A = (
    "A barefoot woman with long brown hair, wearing a red short-sleeved blouse and black "
    "pants, runs forward through a hazy parking garage illuminated by green overhead lights "
    "and a warm orange glow."
)
CORPUS_EXEMPLAR_B = (
    "a pale woman with a tousled black bob and winged eyeliner in a black cutout halter top, "
    "biting the fingertip of her long black latex glove beside a bright curtained window"
)
_V2_STYLE_A = (
    " Match the sentence structure of this example exactly, but describe only what is in "
    f'your own snippet: "{CORPUS_EXEMPLAR_A}" — that is, a subject noun phrase, then one or '
    "two comma-separated appositive phrases giving clothing and features, then a single "
    "finite main verb, then the setting and its lighting. Name the plain colour of every "
    "significant garment, object and light source."
)
_V2_STYLE_B = (
    " Match the structure of this example exactly, but describe only what is in your own "
    f'snippet: "{CORPUS_EXEMPLAR_B}" — that is, a subject noun phrase, then appearance and '
    "clothing, then a comma, then the -ing participle action, then the setting and its "
    "lighting. Name the plain colour of every significant garment, object and light source."
)

# Length calibration fitted on round 1 (n=391): realised = 5.533 + 0.7347 * N_asked
# (corr 0.810).  The model compresses the requested range, running ~3 words short
# on average and ~7 short at the top of the range.  v2 inverts the fit so the
# REALISED distribution matches the corpus draw.  The A4 draw+clamp itself is
# unchanged; this is a post-draw transform on what we ask for.
_LEN_FIT_A, _LEN_FIT_B = 5.533, 0.7347


def calibrate_ask(n_drawn: int) -> int:
    return int(round(max(10, min(60, (n_drawn - _LEN_FIT_A) / _LEN_FIT_B))))


USER_TEXT = "Describe this snippet."

AUDIT_QUESTION = (
    "(a) Does this sentence describe, foreshadow, or hint at any scene change, "
    "transformation, transition, ending, or visual effect? (b) Does it describe anything not "
    "visible in these frames, or get any visible attribute wrong (colors, clothing, hair, "
    "count, action)? Answer JSON {leak: YES/NO, inaccurate: YES/NO, errors: [...]}"
)


# --------------------------------------------------------------------------
# Prompt variant "v3" -- ROUND 3, ruled by the round-9 advisor.  NOT A4's text.
#
# v3 = v2 plus ONE change, A-ROLE ONLY.  A4's prompt mandates "simple-present or
# present-progressive verbs" and thereby forces be-verbs to 0.0% against a corpus
# that uses them 8.8% of the time -- a categorical, prompt-manufactured tell.  By
# A4's own core principle ("captioning is instrument matching") that is a spec
# bug: the frozen eval prompts contain copular constructions at 8.8%.
#
# Scoped as an instrument-matching defect fix, NOT gate-chasing.  Nothing here
# targets `while` or any other gate-#8 coefficient -- doing so would be mimicry
# and would destroy the probe's evidentiary value.  The B-role prompt is
# UNTOUCHED (participial NP is the corpus's own B register).
#
# Pre-committed expectation: v3 does NOT need to reach 0.65 and will not.  It must
# pass the re-pinned bars (8a <= 0.73, 8b <= 0.60) plus all other 11 gates.
# This consumes A4's third and last permitted regeneration round.
# --------------------------------------------------------------------------
_A4_VERB_CLAUSE = (
    "then the subject's action using simple-present or present-progressive verbs"
)
_V3_VERB_CLAUSE = (
    "then the subject's action in present tense (plain 'is/are' constructions are fine)"
)


def build_system_prompt(role: str, n: int, variant: str = "a4") -> str:
    """A4 Q1 prompts.  `variant="a4"` is A4's verbatim text (the default).
    `variant="v2"` is the round-2 operator diagnostic; `variant="v3"` is v2 plus
    the round-3 be-verb fix (A-role only).  All three stay reproducible."""
    tmpl = _PROMPT_A_TEMPLATE if role == "A" else _PROMPT_B_TEMPLATE
    if variant in ("v2", "v3"):
        style = _V2_STYLE_A if role == "A" else _V2_STYLE_B
        n = calibrate_ask(n)
        # insert the structural guidance just before the length instruction
        marker = " Aim for about {N} words"
        tmpl = tmpl.replace(marker, style + marker)
    if variant == "v3" and role == "A":
        assert _A4_VERB_CLAUSE in tmpl, "A4 verb-form clause not found; prompt drifted"
        tmpl = tmpl.replace(_A4_VERB_CLAUSE, _V3_VERB_CLAUSE)
    return tmpl.format(N=n, NLO=n - 6, NHI=n + 8)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class HardStop(Exception):
    """HTTP 429 -- A4/operator directive: treat as a hard stop, do not grind."""


class AuditError(Exception):
    """The Layer-2 audit produced no usable verdict.  ALWAYS fatal, NEVER a pass.

    DOSSIER §21 recorded the defect this class exists to kill: when the auditor was
    down, `_post` returned `(None, err)`, `audit_one` left `verdict = None`, and the
    caller did `v = arec.get("verdict") or {}`.  Neither `v.get("leak") == "YES"` nor
    `v.get("inaccurate") == "YES"` then fired, so the description was **accepted as a
    clean pass** and stored with an `audit` field of `{}`.  An auditor outage thus
    minted records that LOOK audited and are not.

    That is the §13.12 defect class verbatim -- a checker whose failure is
    indistinguishable from a pass -- and it is the third time this campaign has been
    bitten by it (the smoke-gate parser reporting "unreadable" as "training broken";
    a gate whose bar bent to the logging interval).  A13 step 1 orders it fixed
    before anything audits: non-200, unparseable, empty, or field-incomplete raises.

    Proven to fire by `scripts/ctt_v2/tests/prove_audit_hard_error.py`.
    """


_VERDICT_FIELDS = ("leak", "inaccurate")
_VERDICT_VALUES = frozenset({"YES", "NO"})


def validate_audit_verdict(rec: dict) -> dict:
    """Return the audit verdict, or raise `AuditError`.  NEVER returns a default.

    A pure function of the ARCHIVED record, so the negative test drives every branch
    without touching the network, and so the raw response is on disk before this can
    abort the run.  There is deliberately no "lenient" mode and no default verdict:
    the only way to get a verdict out of this function is for the auditor to have
    actually answered, in the pinned schema, with both fields present and in-domain.
    """
    where = f"{rec.get('clip_id')}|{rec.get('role')}"
    if rec.get("error"):
        raise AuditError(f"{where}: audit call failed: {rec['error']}")
    if rec.get("raw_response") is None:
        raise AuditError(f"{where}: audit returned no response object")
    if rec.get("parse_error") is not None:
        raise AuditError(f"{where}: audit verdict is not JSON: {rec['parse_error']!r}")
    v = rec.get("verdict")
    if v is None:
        raise AuditError(f"{where}: audit returned an EMPTY verdict (no text in response)")
    if not isinstance(v, dict):
        raise AuditError(f"{where}: audit verdict is {type(v).__name__}, not an object: {v!r}")
    missing = [f for f in _VERDICT_FIELDS if f not in v]
    if missing:
        raise AuditError(f"{where}: audit verdict missing required field(s) {missing}: {v!r}")
    bad = {f: v[f] for f in _VERDICT_FIELDS if v[f] not in _VERDICT_VALUES}
    if bad:
        raise AuditError(f"{where}: audit verdict field(s) outside {{YES,NO}}: {bad!r}")
    return v


_stop = threading.Event()
_lock = threading.Lock()
_counters = {"calls": 0, "retries": 0, "http429": 0}


def _post(model: str, body: dict, timeout: int = 240, max_tries: int = 5):
    key = os.environ["GEMINI_API_KEY"]
    url = f"{API_ROOT}/{model}:generateContent"
    last = None
    for attempt in range(max_tries):
        if _stop.is_set():
            raise HardStop("stopped")
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
            time.sleep(min(2 ** attempt, 20) + random.random())
            continue
        with _lock:
            _counters["calls"] += 1
        if r.status_code == 429:
            with _lock:
                _counters["http429"] += 1
            _stop.set()
            raise HardStop(f"HTTP 429 from {model}: {r.text[:300]}")
        if r.status_code >= 500 or r.status_code in (408, 409):
            last = f"HTTP{r.status_code}:{r.text[:200]}"
            with _lock:
                _counters["retries"] += 1
            time.sleep(min(2 ** attempt, 20) + random.random())
            continue
        if r.status_code != 200:
            return None, f"HTTP{r.status_code}:{r.text[:300]}"
        return r.json(), None
    return None, f"exhausted_retries:{last}"


def _extract_text(resp: dict):
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        return None


_b64_cache: dict[str, str] = {}


def _b64(path: str) -> str:
    v = _b64_cache.get(path)
    if v is None:
        v = base64.b64encode(Path(path).read_bytes()).decode()
        _b64_cache[path] = v
    return v


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def sample_length(clip_id: str, role: str, attempt: int, empirical, seed: int) -> int:
    """A4 Q1: draw N uniformly from the 171 corpus per-description word counts,
    clamp to [15, 45].  Deterministic in (seed, clip, role, attempt)."""
    rng = random.Random(f"{seed}|{clip_id}|{role}|{attempt}")
    return max(15, min(45, rng.choice(empirical)))


def postprocess(raw: str):
    """A4 Q1 storage convention: strip the trailing period; keep everything else."""
    t = (raw or "").strip()
    t = t.strip("﻿").strip()
    if t.endswith("."):
        t = t[:-1].rstrip()
    return t


def generate_one(clip_id, role, video_path, attempt, empirical, seed, variant="a4"):
    n = sample_length(clip_id, role, attempt, empirical, seed)
    sysp = build_system_prompt(role, n, variant)
    body = {
        "systemInstruction": {"parts": [{"text": sysp}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "video/mp4", "data": _b64(video_path)}},
                    {"text": USER_TEXT},
                ],
            }
        ],
        "generationConfig": {
            "temperature": GEN_TEMPERATURE,
            "maxOutputTokens": GEN_MAX_TOKENS,
            "thinkingConfig": {"thinkingLevel": GEN_THINKING_LEVEL},
        },
    }
    resp, err = _post(GEN_MODEL, body)
    rec = {
        "clip_id": clip_id, "role": role, "attempt": attempt, "N_target": n,
        # v2 AND v3 both apply the length calibration in build_system_prompt(); the
        # archive must record what was actually asked, not the pre-calibration draw.
        "N_asked": calibrate_ask(n) if variant in ("v2", "v3") else n,
        "prompt_variant": variant,
        "model": GEN_MODEL, "temperature": GEN_TEMPERATURE,
        "max_output_tokens": GEN_MAX_TOKENS, "error": err,
        "raw_response": resp,
    }
    text = _extract_text(resp) if resp else None
    rec["raw_text"] = text
    rec["description"] = postprocess(text) if text else None
    return rec


def audit_one(clip_id, role, description, video_path):
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "video/mp4", "data": _b64(video_path)}},
                    {"text": f'Sentence: "{description}."\n\n{AUDIT_QUESTION}'},
                ],
            }
        ],
        "generationConfig": {
            "temperature": AUDIT_TEMPERATURE,
            "maxOutputTokens": AUDIT_MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "leak": {"type": "STRING", "enum": ["YES", "NO"]},
                    "inaccurate": {"type": "STRING", "enum": ["YES", "NO"]},
                    "errors": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["leak", "inaccurate", "errors"],
            },
            "thinkingConfig": {"thinkingLevel": AUDIT_THINKING_LEVEL},
        },
    }
    resp, err = _post(AUDIT_MODEL, body)
    rec = {
        "clip_id": clip_id, "role": role, "model": AUDIT_MODEL,
        "temperature": AUDIT_TEMPERATURE,
        "max_output_tokens": AUDIT_MAX_TOKENS,
        "thinking_level": AUDIT_THINKING_LEVEL,
        # Standing rule: archive the model version string for every measurement.
        "model_version_echo": (resp or {}).get("modelVersion"),
        "error": err, "raw_response": resp,
    }
    txt = _extract_text(resp) if resp else None
    verdict = None
    if txt:
        try:
            verdict = json.loads(txt)
        except Exception:
            rec["parse_error"] = txt[:400]
    rec["verdict"] = verdict
    return rec


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(pairs, outdir: Path, seed: int, workers: int, max_attempts: int,
        audit: bool = True, variant: str = "a4"):
    outdir.mkdir(parents=True, exist_ok=True)
    index = json.loads(STRIPS_INDEX.read_text())
    empirical = load_length_empirical()
    lf = LeakFilter()

    gen_archive = (outdir / "raw_generation_responses.jsonl").open("w")
    audit_archive = (outdir / "raw_audit_responses.jsonl").open("w")
    arch_lock = threading.Lock()

    results = {}
    t0 = time.time()

    def one_pair(pair):
        clip_id, role = pair
        entry = index[clip_id]
        video = entry[f"{role}_video"]
        history = []
        for attempt in range(1, max_attempts + 1):
            rec = generate_one(clip_id, role, video, attempt, empirical, seed, variant)
            with arch_lock:
                gen_archive.write(json.dumps(rec) + "\n")
            desc = rec["description"]
            step = {
                "attempt": attempt, "N_target": rec["N_target"], "error": rec["error"],
                "description": desc,
            }
            if not desc:
                step["fail"] = ["no_text"]
                history.append(step)
                continue
            fmt = format_violations(desc, role)
            t1 = lf.tier1(desc)
            step["format_violations"] = fmt
            step["tier1"] = t1
            step["tier2"] = lf.tier2(desc)
            if fmt or t1:
                step["fail"] = (["format"] if fmt else []) + (["tier1"] if t1 else [])
                history.append(step)
                continue
            if audit:
                arec = audit_one(clip_id, role, desc, video)
                # Archive BEFORE validating: a fatal verdict must still leave its raw
                # response on disk, and flush so an abort cannot lose the buffer.
                with arch_lock:
                    audit_archive.write(json.dumps(arec) + "\n")
                    audit_archive.flush()
                # Raises AuditError on non-200 / empty / unparseable / incomplete.
                # There is NO `or {}` here and there must never be one again: that
                # expression is what scored an auditor outage as a clean pass (§21).
                v = validate_audit_verdict(arec)
                step["audit"] = v
                if v.get("leak") == "YES" or v.get("inaccurate") == "YES":
                    step["fail"] = (["leak"] if v.get("leak") == "YES" else []) + \
                                   (["inaccurate"] if v.get("inaccurate") == "YES" else [])
                    history.append(step)
                    continue
            step["fail"] = []
            history.append(step)
            return {
                "clip_id": clip_id, "role": role, "bank": entry["bank"],
                "description": desc, "accepted_on_attempt": attempt,
                "N_target": rec["N_target"], "words": word_count(desc),
                "tier2": step["tier2"], "audit": step.get("audit"),
                "history": history,
            }
        return {
            "clip_id": clip_id, "role": role, "bank": entry["bank"],
            "description": None, "accepted_on_attempt": None,
            "status": "MANUAL_REWRITE_QUEUE", "history": history,
        }

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one_pair, p): p for p in pairs}
            done = 0
            for f, p in list(futs.items()):
                try:
                    r = f.result()
                except HardStop as e:
                    print(f"HARD STOP: {e}", file=sys.stderr)
                    raise
                except AuditError as e:
                    # A13 step 1: an unusable audit verdict aborts the run.  It is NOT
                    # downgraded to a skip, a retry or a manual-queue entry -- a store
                    # that is partly unaudited must never be mistaken for an audited one.
                    _stop.set()
                    print(f"AUDIT HARD ERROR (run aborted): {e}", file=sys.stderr)
                    raise
                results[f"{p[0]}|{p[1]}"] = r
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)
    finally:
        gen_archive.close()
        audit_archive.close()

    # description store
    store = {}
    for r in results.values():
        if r["description"]:
            store.setdefault(r["clip_id"], {})[r["role"]] = r["description"]
    (outdir / "descriptions.json").write_text(json.dumps(store, indent=1, sort_keys=True))
    (outdir / "records.json").write_text(json.dumps(results, indent=1, sort_keys=True))

    wall = time.time() - t0
    meta = {
        "generator_model": GEN_MODEL,
        # An unaudited run must NEVER archive an auditor model string: a reader who
        # trusts run_meta would conclude the Layer-2 audit ran when it did not.
        "audit_enabled": bool(audit),
        "auditor_model": AUDIT_MODEL if audit else None,
        "auditor_max_output_tokens": AUDIT_MAX_TOKENS if audit else None,
        "auditor_thinking_level": AUDIT_THINKING_LEVEL if audit else None,
        "auditor_substitution_note": (
            "A13 Q1 (2026-07-28): auditor pinned to gemini-3.1-pro-preview, temp 0, "
            "thinkingLevel 'low' (the pro tier rejects 'minimal'), max_output_tokens 512. "
            "A4 named a pro-tier auditor and made generator/auditor INDEPENDENCE binding; "
            "the earlier gemini-3.5-flash substitution was a forced deviation recorded "
            "under a factual error (DOSSIER §5.1 read the pro tier as rate-limited; "
            "gemini-3-pro-preview is in fact RETIRED/404). This promotion REMOVES a "
            "deviation. No-switch-back rule: the auditor does not revert if 3.5-flash "
            "returns. First-pass rates across this change are NOT a like-for-like series."
        ) if audit else (
            "NO LAYER-2 AUDIT RAN (--no-audit). Records carry no `audit` field; leak= and "
            "inaccurate= were NOT measured, so first-pass rate covers only format+Tier-1 "
            "and is an UPPER BOUND. Not eligible to gate the >=97% / <=8% bars."
        ),
        "empty_verdict_policy": (
            "HARD ERROR (A13 step 1): a non-200, empty, unparseable or field-incomplete "
            "audit response raises AuditError and aborts the run. It is NEVER scored as a "
            "clean pass. Proven to fire by scripts/ctt_v2/tests/prove_audit_hard_error.py."
        ),
        "gen_thinking_level": GEN_THINKING_LEVEL, "prompt_variant": variant,
        "gen_temperature": GEN_TEMPERATURE, "gen_max_output_tokens": GEN_MAX_TOKENS,
        "audit_temperature": AUDIT_TEMPERATURE,
        "seed": seed, "workers": workers, "max_attempts": max_attempts,
        "n_pairs": len(pairs), "wall_seconds": round(wall, 1),
        "api_calls": _counters["calls"], "retries": _counters["retries"],
        "http429": _counters["http429"],
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return results


def apply_role_scoped_exclusions(pairs):
    """A11 item 5 — drop the (clip, role) descriptions the M3 adjudication excluded.

    DERIVED from `POOL_DROPS_M3_ADJUDICATION.json`, never a hand-kept list: the pool file
    itself is deliberately byte-unchanged (nothing may desynchronise from the S2b render),
    so the adjudication is the only carrier of the instruction.  `openvid_T1MiFx98l3g_0_50to156`
    has a blank-white A-anchor (grayscale std 0.79) but a healthy B-anchor (std 74.05) and
    occupies the B field in 10/10 rendered S2b clips, so ONLY role A is skipped — the
    B-role description is legitimate and must still be generated.

    An ABSENT adjudication file is a hard stop, not an empty exclusion: silently generating
    the excluded description is exactly the failure mode the machine check exists to catch.
    `assert_root.py`'s caption-store check re-verifies the same derived list after the fact.
    """
    role, clip_level, prov = load_caption_store_exclusions()
    if prov.get("error"):
        raise SystemExit(f"[exclusions] {prov['file']}: {prov['error']}")
    kept, skipped = [], []
    for clip_id, r in pairs:
        if clip_id in clip_level or r in role.get(clip_id, ()):
            skipped.append((clip_id, r))
        else:
            kept.append((clip_id, r))
    print(f"[exclusions] {prov['file']} sha {prov['sha256'][:12]}: "
          f"{sum(len(v) for v in role.values())} role-scoped + {len(clip_level)} clip-level "
          f"exclusions on record; skipped {len(skipped)} of {len(pairs)} requested pairs"
          + (f" -> {sorted(skipped)}" if skipped else ""))
    return kept


def deterministic_sample(n_per_cell: int, seed: int):
    """N clips per bank (deterministic, sorted clip list, seeded), BOTH roles each.

    This mirrors the production store shape `{clip_id: {A, B}}`: every sampled
    clip contributes one A-role and one B-role description, giving
    n_per_cell descriptions in each of the (bank x role) cells.
    """
    index = json.loads(STRIPS_INDEX.read_text())
    banks = {}
    for cid, e in index.items():
        banks.setdefault(e["bank"], []).append(cid)
    pairs = []
    for bank in sorted(banks):
        clips = sorted(banks[bank])
        rng = random.Random(f"{seed}|{bank}")
        chosen = sorted(rng.sample(clips, min(n_per_cell, len(clips))))
        for c in chosen:
            pairs.append((c, "A"))
            pairs.append((c, "B"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None,
                    help='JSON file: [["clip_id","A"], ...]')
    ap.add_argument("--sample-per-cell", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="A4 Q3: one regeneration on failure, then manual queue")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--prompt-variant", choices=("a4", "v2", "v3"), default="a4",
                    help='"a4" = A4 Q1 verbatim (default); "v2" = round-2 operator '
                         'diagnostic; "v3" = v2 + the round-3 be-verb fix '
                         "(see module docstring)")
    a = ap.parse_args()

    if a.pairs:
        pairs = [tuple(x) for x in json.loads(Path(a.pairs).read_text())]
    elif a.sample_per_cell:
        pairs = deterministic_sample(a.sample_per_cell, a.seed)
    else:
        ap.error("need --pairs or --sample-per-cell")

    pairs = apply_role_scoped_exclusions(pairs)

    print(f"{len(pairs)} (clip, role) pairs; generator={GEN_MODEL} auditor={AUDIT_MODEL} "
          f"prompt_variant={a.prompt_variant}")
    run(pairs, Path(a.out), a.seed, a.workers, a.max_attempts, audit=not a.no_audit,
        variant=a.prompt_variant)


if __name__ == "__main__":
    main()
