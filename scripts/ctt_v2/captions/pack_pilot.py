#!/usr/bin/env python
"""K-packed description generation + audit -- advisor A12 (budget ruling, 2026-07-28).

WHY
---
The 393 TL budget does not fit the 4,532-description store at the round-2 unpacked rate
(682 tok/description measured).  A12 ruled: pack K=10 clips into one call, gated by a pilot
with pre-registered bars.  Packing amortises the *static* prompt over K descriptions.  It does
NOT amortise the per-clip video payload or (for audits) the description text itself, which is
why A12 corrected the packed-audit estimate from ~103 to ~143 tok/desc.

THE ONE THING THAT MUST NOT DRIFT
---------------------------------
The pinned ROUND-2 prompt text.  A12 §5: editing or compressing it is "the one proven way to
blow 8a" -- a mere prompt delta inside one model measurably opened 0.7233 separation, above
the 0.73 drift guard.  So the round-2 prompt is embedded **byte-identically, exactly once**,
produced by calling `generate_descriptions.build_system_prompt(role, n, "v2")` -- the same
code path round 2 ran, not a copy of its text -- and `assert_embedded_once()` proves both the
byte-identity and the single occurrence before any request is sent.

Round 3 is CANCELLED by A12 (DOSSIER §13.2: the 12/12-passing store is round 2).  There is no
`v3` path in this file on purpose.

DELIBERATE DEVIATIONS FROM ROUND 2, both forced by packing and both recorded
---------------------------------------------------------------------------
1. **The length target N is drawn per PACK, not per item.**  The round-2 prompt carries a
   single "Aim for about {N} words" clause and A12 restricts the wrapper to "item IDs,
   separators, and the independence instruction" -- so per-item word budgets are not
   available.  To keep the *marginal* per-item N distribution identical to round 2's (which is
   what gates #1/#2 measure), pack-level N is drawn by **stratified systematic sampling** of
   the same clamped 171-value empirical distribution rather than i.i.d.: the marginal is
   preserved exactly and the pack-level draw contributes no extra sampling noise to the
   p10/p50/p90 estimates.
2. **Schema-constrained JSON array output** keyed by echoed item IDs (A12 §3).  Round 2 took
   free text.  This is the mechanism that makes per-item attribution checkable at all.

ITEM IDS ARE OPAQUE, AND THAT IS A SAFETY DECISION
--------------------------------------------------
A12 says "keyed by echoed clip IDs".  Real clip IDs are NOT sent: ids like `davis_dogs-scale`
or `davis_blackswan` carry content words that are not in the nine frames, which would inject
information the unpacked round-2 calls never had -- a genuine contamination channel and a new
style cue.  Each item instead gets a deterministic 5-consonant opaque code (never sequential,
so echoing them in order is not a free pass, and digit-free so it cannot trip the Tier-1
leetspeak detector).  The code -> (clip, role) map lives locally.

Usage
-----
  source $LAB/secrets/gemini_transition.env
  PY=$LAB/envs/diffusion/bin/python
  $PY pack_pilot.py plan   --out <dir>                    # free: build + verify the packs
  $PY pack_pilot.py smoke  --out <dir> --auditor <model>  # ~2 TL: 1 pack each way
  $PY pack_pilot.py gen    --out <dir>                    # step 2
  $PY pack_pilot.py mismatch --out <dir> --auditor <model> # step 3 (391 wrong-clip pairs)
  $PY pack_pilot.py audit  --out <dir> --auditor <model>   # step 4 (matched + derangement)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from caption_common import STRIPS_INDEX, LeakFilter, format_violations, word_count  # noqa: E402
from generate_descriptions import (  # noqa: E402
    AUDIT_QUESTION,
    GEN_MAX_TOKENS,
    GEN_MODEL,
    GEN_TEMPERATURE,
    AUDIT_TEMPERATURE,
    build_system_prompt,
    postprocess,
)
from root_common import role_excluded  # noqa: E402

REPO = HERE.parents[2]
POOL = REPO / "data/processed/ctt_v2_strata/CONTENT_POOL_union.json"
ROUND2 = HERE / "pilot_m3/round2"
LENGTH_EMPIRICAL = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa"
                        "/misc/ctt_v2_final/M1_length_empirical.json")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
ROUND2_VARIANT = "v2"            # A12: round 2 IS the production prompt. Round 3 cancelled.
PACK_SEED = 4242
GEN_MAX_TOKENS_PACKED = 2000     # 10 x 120 visible + JSON envelope; a cap, not a charge
#: 1100, not 1400, and the number is a BUDGET SEATBELT as much as a format setting: with the
#: prompt bounded near 1,500 tokens (wrapper + 10 x (63-token video + ~45-token sentence)),
#: capping output at 1100 makes the worst-case cost of all 20 packs arithmetically provable
#: (20 x 2,600 = 52k tok = 19.3 TL) instead of merely expected, which is what keeps the
#: 90 TL step-1-4 cap from depending on a forecast. Measured need is ~600 (5 deranged items
#: carry ~90 tok of `errors` each, 5 matched carry ~28), so this is ~1.8x headroom.
AUDIT_MAX_TOKENS_PACKED = 1100

# --------------------------------------------------------------------------
# The minimal wrapper.  Adds ONLY: item IDs, separators, an independence
# instruction, and the output-shape line the JSON schema needs.  Nothing else.
# --------------------------------------------------------------------------
_WRAP_HEAD = (
    "You will receive {K} separate 9-frame video snippets. Each snippet is preceded by a "
    'line of the form "=== ITEM <n> | id: <CODE> ===". Apply the instruction block below '
    "independently to every snippet, exactly as if each snippet were the only one you had "
    "received. Describe each snippet on its own: never reference, compare with, or carry over "
    "wording, subjects, objects, colours, or setting from any other snippet in this request.\n\n"
    "----- BEGIN INSTRUCTION BLOCK (applies to each snippet independently) -----\n"
)
_WRAP_TAIL = (
    "\n----- END INSTRUCTION BLOCK -----\n\n"
    "Return a JSON array of exactly {K} objects, one per snippet, in the order the snippets "
    'appear. Each object is {{"id": "<that snippet\'s CODE, echoed exactly>", '
    '"description": "<that snippet\'s text, exactly as the instruction block specifies>"}}.'
)

_AUDIT_WRAP_HEAD = (
    "You will receive {K} separate items. Each item is a line of the form "
    '"=== ITEM <n> | id: <CODE> ===", then a 9-frame video snippet, then one sentence that '
    "is claimed to describe THAT snippet. Judge every item strictly on its own snippet and "
    "its own sentence: never let another item's snippet or sentence influence your judgement, "
    "and never assume the sentences are in the right order.\n\n"
    "----- BEGIN QUESTION BLOCK (answer for each item independently) -----\n"
)
_AUDIT_WRAP_TAIL = (
    "\n----- END QUESTION BLOCK -----\n\n"
    "Return a JSON array of exactly {K} objects, one per item, in the order the items appear. "
    'Each object is {{"id": "<that item\'s CODE, echoed exactly>", "leak": "YES"/"NO", '
    '"inaccurate": "YES"/"NO", "errors": [...]}}.'
)

_CODE_ALPHABET = "BCDFGHJKLMNPQRSTVWXZ"     # consonants only: digit-free, not word-forming


class HardStop(Exception):
    """HTTP 429 -- hard stop, never grind against a depleted balance."""


_stop = threading.Event()
_lock = threading.Lock()
_counters = {"calls": 0, "retries": 0, "http429": 0}


# ==========================================================================
# Prompt construction + the byte-identity assert
# ==========================================================================
def item_code(clip_id: str, role: str) -> str:
    h = hashlib.sha256(f"{PACK_SEED}|{clip_id}|{role}".encode()).digest()
    return "".join(_CODE_ALPHABET[b % len(_CODE_ALPHABET)] for b in h[:5])


def build_packed_system_prompt(role: str, n: int, k: int) -> tuple[str, str]:
    """(wrapper, embedded) with the pinned round-2 prompt inside, byte-identical, once."""
    embedded = build_system_prompt(role, n, ROUND2_VARIANT)
    wrapper = _WRAP_HEAD.format(K=k) + embedded + _WRAP_TAIL.format(K=k)
    assert_embedded_once(wrapper, embedded, role, n)
    return wrapper, embedded


def assert_embedded_once(wrapper: str, embedded: str, role: str, n: int) -> None:
    """A12 §3 / §5: the pinned prompt must appear byte-identically and exactly once.

    Re-derives the expected text from the pinned code path instead of trusting the caller,
    so a mutated `embedded` cannot slip through by being consistent with itself.
    """
    expect = build_system_prompt(role, n, ROUND2_VARIANT)
    if embedded != expect:
        raise AssertionError(
            f"embedded prompt is NOT the pinned round-2 text for role {role}, N={n}: "
            f"sha {hashlib.sha256(embedded.encode()).hexdigest()[:16]} != "
            f"{hashlib.sha256(expect.encode()).hexdigest()[:16]}")
    c = wrapper.count(expect)
    if c != 1:
        raise AssertionError(f"pinned round-2 prompt appears {c} times in the wrapper, not once")
    if "v3" in wrapper or "is/are' constructions" in wrapper:
        raise AssertionError("round-3 (v3) text detected in the wrapper; A12 CANCELLED round 3")


def build_packed_audit_prompt(k: int) -> tuple[str, str]:
    wrapper = _AUDIT_WRAP_HEAD.format(K=k) + AUDIT_QUESTION + _AUDIT_WRAP_TAIL.format(K=k)
    if wrapper.count(AUDIT_QUESTION) != 1:
        raise AssertionError("pinned A8 audit question must appear exactly once")
    return wrapper, AUDIT_QUESTION


# ==========================================================================
# HTTP -- with the thinkingLevel assert on EVERY call
# ==========================================================================
def _assert_thinking(body: dict) -> None:
    """DOSSIER §19.3: only the NESTED path works and it halves cost (91 tok vs 187).

    `generationConfig.thinkingLevel` (flat) is rejected 400 Unknown name and
    `thinkingConfig.thinkingBudget: 0` is rejected 400 INVALID_ARGUMENT, so a typo here does
    not fail loudly -- it silently doubles the bill.  Assert before every send.
    """
    gc = body.get("generationConfig") or {}
    tc = gc.get("thinkingConfig") or {}
    if tc.get("thinkingLevel") != "minimal":
        raise AssertionError("generationConfig.thinkingConfig.thinkingLevel must be 'minimal'")
    if "thinkingLevel" in gc:
        raise AssertionError("flat generationConfig.thinkingLevel is rejected 400 -- remove it")
    if "thinkingBudget" in tc:
        raise AssertionError("thinkingConfig.thinkingBudget is rejected 400 -- remove it")


def _post(model: str, body: dict, timeout: int = 600, max_tries: int = 5):
    _assert_thinking(body)
    key = os.environ["GEMINI_API_KEY"]
    url = f"{API_ROOT}/{model}:generateContent"
    last = None
    for attempt in range(max_tries):
        if _stop.is_set():
            raise HardStop("stopped")
        try:
            r = requests.post(url, headers={"x-goog-api-key": key,
                                            "Content-Type": "application/json"},
                              json=body, timeout=timeout)
        except Exception as e:
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
        j = r.json()
        mv = j.get("modelVersion")
        if mv and mv != model:
            # A12 §2: an alias whose modelVersion drifts mid-run cannot be a pinned config.
            return j, f"modelVersion_mismatch:requested={model} echoed={mv}"
        return j, None
    return None, f"exhausted_retries:{last}"


_b64_cache: dict[str, str] = {}


def _b64(path: str) -> str:
    v = _b64_cache.get(path)
    if v is None:
        v = base64.b64encode(Path(path).read_bytes()).decode()
        _b64_cache[path] = v
    return v


def _text(resp: dict):
    try:
        return "".join(p.get("text", "")
                       for p in resp["candidates"][0]["content"]["parts"]).strip()
    except Exception:
        return None


# ==========================================================================
# Pack planning
# ==========================================================================
def clamped_empirical() -> list[int]:
    """Round 2's marginal: `max(15, min(45, choice(empirical)))`."""
    return [max(15, min(45, v)) for v in json.loads(LENGTH_EMPIRICAL.read_text())]


def stratified_pack_lengths(n_packs: int, seed: int) -> list[int]:
    """Systematic sample of the clamped empirical distribution -- marginal preserved,
    no extra sampling noise from the pack-level draw (see module docstring, deviation 1)."""
    vals = sorted(clamped_empirical())
    rng = random.Random(f"{seed}|packlen")
    u = rng.random()
    out = [vals[min(len(vals) - 1, int((i + u) / n_packs * len(vals)))] for i in range(n_packs)]
    rng.shuffle(out)
    return out


def build_plan(k: int = 10, per_cell: int = 50, out: Path | None = None) -> dict:
    """20 role- AND bank-homogeneous packs: 4 (role x bank) cells x 5 packs x K=10.

    Each cell is half PAIRED (a round-2 accepted description exists for the same
    (clip, role), so the packed-vs-unpacked probe is paired) and half FRESH pool
    (clip, role)s, which are additive to the production store.  Paired and fresh are
    mixed *within* packs on purpose, so "paired" is never confounded with pack identity.

    Bank-homogeneous packs are the CONSERVATIVE choice for gate 8b: if packing induces
    phrase echo, echo inside a bank-pure pack pushes the two banks apart (8b up, worse),
    whereas bank-mixed packs would pull them together and flatter the gate.
    """
    idx = json.loads(STRIPS_INDEX.read_text())
    pool_ids = set(json.loads(POOL.read_text())["ids"])
    r2 = json.loads((ROUND2 / "records.json").read_text())

    r2_seen = {(v["clip_id"], v["role"]) for v in r2.values()}
    paired_pool = {}
    for v in r2.values():
        if not v.get("description"):
            continue
        cid, role = v["clip_id"], v["role"]
        if cid not in pool_ids or role_excluded(cid, role):
            continue
        paired_pool[(cid, role)] = v["description"]

    banks = ("synth_endpoints", "humanvid")
    cells, rng = {}, random.Random(f"{PACK_SEED}|cells")
    for role in ("A", "B"):
        for bank in banks:
            cand_p = sorted(k2 for k2 in paired_pool if k2[1] == role
                            and idx[k2[0]]["bank"] == bank)
            cand_f = sorted((cid, role) for cid in pool_ids
                            if idx[cid]["bank"] == bank
                            and (cid, role) not in r2_seen
                            and not role_excluded(cid, role))
            half = per_cell // 2
            if len(cand_p) < half or len(cand_f) < half:
                raise SystemExit(f"[plan] cell ({role},{bank}): paired {len(cand_p)} "
                                 f"fresh {len(cand_f)}, need {half} of each")
            sel_p = sorted(random.Random(f"{PACK_SEED}|p|{role}|{bank}").sample(cand_p, half))
            sel_f = sorted(random.Random(f"{PACK_SEED}|f|{role}|{bank}").sample(cand_f, half))
            items = ([{"clip_id": c, "role": r, "arm": "paired"} for c, r in sel_p]
                     + [{"clip_id": c, "role": r, "arm": "fresh"} for c, r in sel_f])
            random.Random(f"{PACK_SEED}|mix|{role}|{bank}").shuffle(items)
            cells[(role, bank)] = items

    packs, pid = [], 0
    for (role, bank), items in sorted(cells.items()):
        for s in range(0, len(items), k):
            chunk = items[s:s + k]
            assert len({i["role"] for i in chunk}) == 1, "packs must be role-homogeneous"
            assert len({idx[i['clip_id']]['bank'] for i in chunk}) == 1
            for i in chunk:
                if role_excluded(i["clip_id"], i["role"]):
                    raise SystemExit(f"[plan] role-excluded pair leaked in: {i}")
                i["code"] = item_code(i["clip_id"], i["role"])
                i["video"] = idx[i["clip_id"]][f"{i['role']}_video"]
                i["bank"] = idx[i["clip_id"]]["bank"]
            assert len({i["code"] for i in chunk}) == len(chunk), "code collision inside a pack"
            packs.append({"pack_id": f"P{pid:02d}", "role": role, "bank": bank,
                          "k": len(chunk), "items": chunk})
            pid += 1

    lens = stratified_pack_lengths(len(packs), PACK_SEED)
    for p, n in zip(packs, lens):
        p["N_target"] = n
        w, e = build_packed_system_prompt(p["role"], n, p["k"])
        p["system_prompt_sha256"] = hashlib.sha256(w.encode()).hexdigest()
        p["embedded_round2_prompt_sha256"] = hashlib.sha256(e.encode()).hexdigest()

    all_codes = [i["code"] for p in packs for i in p["items"]]
    assert len(set(all_codes)) == len(all_codes), "global item-code collision"
    plan = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "authority": "advisors/A12_budget_allocation_VERBATIM.md §3",
        "K": k, "n_packs": len(packs), "n_items": len(all_codes),
        "pack_seed": PACK_SEED, "prompt_variant": ROUND2_VARIANT,
        "round3_cancelled": True,
        "generator_model": GEN_MODEL,
        "deviations_from_round2": [
            "pack-level N by stratified systematic sampling of the clamped empirical "
            "distribution (marginal preserved; per-item N is impossible with one prompt)",
            "schema-constrained JSON array output keyed by echoed opaque item codes",
            "opaque 5-consonant item codes instead of real clip_ids (clip_ids carry content "
            "words that are not in the nine frames -- sending them would be a leak channel)",
        ],
        "packs": packs,
    }
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "pack_plan.json").write_text(json.dumps(plan, indent=1))
    return plan


# ==========================================================================
# Packed generation
# ==========================================================================
def gen_schema() -> dict:
    return {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "STRING"}, "description": {"type": "STRING"}},
        "required": ["id", "description"], "propertyOrdering": ["id", "description"]}}


def audit_schema() -> dict:
    return {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "STRING"},
                       "leak": {"type": "STRING", "enum": ["YES", "NO"]},
                       "inaccurate": {"type": "STRING", "enum": ["YES", "NO"]},
                       "errors": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["id", "leak", "inaccurate", "errors"],
        "propertyOrdering": ["id", "leak", "inaccurate", "errors"]}}


def generate_pack(pack: dict) -> dict:
    role, n, k = pack["role"], pack["N_target"], pack["k"]
    sysp, embedded = build_packed_system_prompt(role, n, k)
    parts = []
    for j, it in enumerate(pack["items"], 1):
        parts.append({"text": f"=== ITEM {j} | id: {it['code']} ==="})
        parts.append({"inline_data": {"mime_type": "video/mp4", "data": _b64(it["video"])}})
    parts.append({"text": f"Describe each of the {k} snippets above."})
    body = {
        "systemInstruction": {"parts": [{"text": sysp}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": GEN_TEMPERATURE,
            "maxOutputTokens": GEN_MAX_TOKENS_PACKED,
            "responseMimeType": "application/json",
            "responseSchema": gen_schema(),
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    resp, err = _post(GEN_MODEL, body)
    rec = {"kind": "packed_generation", "pack_id": pack["pack_id"], "role": role,
           "bank": pack["bank"], "k": k, "N_target": n, "prompt_variant": ROUND2_VARIANT,
           "model": GEN_MODEL, "temperature": GEN_TEMPERATURE,
           "max_output_tokens": GEN_MAX_TOKENS_PACKED,
           "system_prompt_sha256": hashlib.sha256(sysp.encode()).hexdigest(),
           "embedded_round2_prompt_sha256": hashlib.sha256(embedded.encode()).hexdigest(),
           "thinking_level": "minimal",
           "items": [{"clip_id": i["clip_id"], "role": i["role"], "code": i["code"],
                      "arm": i["arm"]} for i in pack["items"]],
           "error": err, "raw_response": resp}
    rec["parsed"], rec["parse_error"] = _parse_array(resp, [i["code"] for i in pack["items"]])
    return rec


def _parse_array(resp, codes: list[str]):
    """Parse the JSON array and check the ID echo -- exact set, exact order, no extras."""
    txt = _text(resp) if resp else None
    if not txt:
        return None, "no_text"
    try:
        arr = json.loads(txt)
    except Exception as e:
        return None, f"json:{type(e).__name__}:{str(e)[:120]}"
    if not isinstance(arr, list):
        return None, "not_a_list"
    got = [str(o.get("id", "")) for o in arr if isinstance(o, dict)]
    prob = []
    if len(arr) != len(codes):
        prob.append(f"length {len(arr)} != {len(codes)}")
    if got != codes[:len(got)] or set(got) != set(codes):
        prob.append(f"id echo mismatch: {got} vs {codes}")
    return arr, ("; ".join(prob) or None)


# ==========================================================================
# Audits
# ==========================================================================
def audit_unpacked(model: str, clip_id: str, role: str, description: str,
                   video: str, tag: str, truth: str) -> dict:
    """A8's validated shape, byte-for-byte: one video + one sentence + AUDIT_QUESTION."""
    body = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "video/mp4", "data": _b64(video)}},
            {"text": f'Sentence: "{description}."\n\n{AUDIT_QUESTION}'}]}],
        "generationConfig": {
            "temperature": AUDIT_TEMPERATURE, "maxOutputTokens": 400,
            "responseMimeType": "application/json",
            "responseSchema": {"type": "OBJECT", "properties": {
                "leak": {"type": "STRING", "enum": ["YES", "NO"]},
                "inaccurate": {"type": "STRING", "enum": ["YES", "NO"]},
                "errors": {"type": "ARRAY", "items": {"type": "STRING"}}},
                "required": ["leak", "inaccurate", "errors"]},
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    resp, err = _post(model, body)
    rec = {"kind": tag, "clip_id": clip_id, "role": role, "model": model,
           "temperature": AUDIT_TEMPERATURE, "thinking_level": "minimal",
           "packed": False, "truth": truth, "video": video,
           "description": description, "error": err, "raw_response": resp}
    txt = _text(resp) if resp else None
    try:
        rec["verdict"] = json.loads(txt) if txt else None
    except Exception:
        rec["verdict"], rec["parse_error"] = None, (txt or "")[:300]
    return rec


def audit_pack(model: str, pack_id: str, items: list[dict], tag: str) -> dict:
    """Packed audit.  `items` = [{code, clip_id, role, description, video, truth}]."""
    k = len(items)
    wrapper, embedded = build_packed_audit_prompt(k)
    parts = [{"text": wrapper}]
    for j, it in enumerate(items, 1):
        parts.append({"text": f"=== ITEM {j} | id: {it['code']} ==="})
        parts.append({"inline_data": {"mime_type": "video/mp4", "data": _b64(it["video"])}})
        parts.append({"text": f'Sentence: "{it["description"]}."'})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": AUDIT_TEMPERATURE, "maxOutputTokens": AUDIT_MAX_TOKENS_PACKED,
            "responseMimeType": "application/json", "responseSchema": audit_schema(),
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    resp, err = _post(model, body)
    rec = {"kind": tag, "pack_id": pack_id, "model": model, "packed": True, "k": k,
           "temperature": AUDIT_TEMPERATURE, "thinking_level": "minimal",
           "audit_question_sha256": hashlib.sha256(embedded.encode()).hexdigest(),
           "items": [{kk: it[kk] for kk in ("code", "clip_id", "role", "truth", "video")}
                     for it in items],
           "error": err, "raw_response": resp}
    rec["parsed"], rec["parse_error"] = _parse_array(resp, [i["code"] for i in items])
    return rec


# ==========================================================================
# Drivers
# ==========================================================================
def _pool(fn, jobs, workers, label):
    t0, out = time.time(), []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(fn, jobs), 1):
            out.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {label} {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    return out


def _archive(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"  archived {len(recs)} raw responses -> {path}")


def cmd_gen(a):
    """Generate (or, with --from-archive, re-derive rows from the archived responses).

    ROWS ARE KEYED BY THE ECHOED ID, NEVER BY ARRAY POSITION.  A12 §3 specifies "output
    keyed by echoed clip IDs", and the pilot immediately showed why: pack P06 returned all
    ten ids intact but with two ADJACENT items transposed, so position-keying would have
    silently attached each of those two descriptions to the other clip.  `id_echo_intact`
    (the code appears exactly once in the response) and `order_preserved` (it appears at its
    requested position) are therefore recorded as SEPARATE diagnostics: the first is
    condition 3(c), the second is the thing the derangement audit in 3(a) has to adjudicate.
    """
    out = Path(a.out)
    plan = json.loads((out / "pack_plan.json").read_text())
    arch = out / "raw_generation_responses.jsonl"
    if a.from_archive:
        recs = [json.loads(x) for x in arch.open()]
        print(f"[gen] re-deriving rows from {len(recs)} ARCHIVED responses (no API calls)")
    else:
        packs = plan["packs"][:a.limit] if a.limit else plan["packs"]
        print(f"[gen] {len(packs)} packs x K={plan['K']} = {sum(p['k'] for p in packs)} "
              f"descriptions | generator={GEN_MODEL} | prompt_variant={ROUND2_VARIANT}")
        recs = _pool(generate_pack, packs, a.workers, "pack")
        _archive(arch, recs)

    lf, store, rows = LeakFilter(), {}, []
    for rec in recs:
        parsed = rec.get("parsed") or []
        bycode: dict[str, list] = {}
        for o in parsed:
            if isinstance(o, dict) and o.get("id") is not None:
                bycode.setdefault(str(o["id"]), []).append(o)
        for j, it in enumerate(rec["items"]):
            cand = bycode.get(it["code"], [])
            o = cand[0] if len(cand) == 1 else None          # unique echo, or nothing
            at_pos = parsed[j] if j < len(parsed) and isinstance(parsed[j], dict) else {}
            raw = (o or {}).get("description")
            desc = postprocess(raw) if raw else None
            rows.append({
                "clip_id": it["clip_id"], "role": it["role"], "code": it["code"],
                "arm": it["arm"], "bank": rec["bank"], "pack_id": rec["pack_id"],
                "pack_pos": j, "N_target": rec["N_target"],
                "id_echo_intact": len(cand) == 1,
                "order_preserved": str(at_pos.get("id")) == it["code"],
                "raw_text": raw, "description": desc,
                "words": word_count(desc) if desc else None,
                "format_violations": format_violations(desc, it["role"]) if desc else ["empty"],
                "tier1": lf.tier1(desc) if desc else [],
                "tier2": lf.tier2(desc) if desc else []})
            if desc:
                store[f"{it['clip_id']}|{it['role']}"] = desc
    (out / "packed_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "descriptions.json").write_text(json.dumps(store, indent=1, sort_keys=True))
    n = len(rows)
    print(f"[gen] parsed {sum(1 for r in rows if r['description'])}/{n} | "
          f"ID ECHO INTACT {sum(r['id_echo_intact'] for r in rows)}/{n} | "
          f"order preserved {sum(r['order_preserved'] for r in rows)}/{n} | "
          f"format violations {sum(1 for r in rows if r['format_violations'])} | "
          f"tier1 {sum(1 for r in rows if r['tier1'])} | "
          f"tier2 {sum(1 for r in rows if r['tier2'])}")
    for r in recs:
        if r.get("parse_error") or r.get("error"):
            print(f"   ! {r['pack_id']}: err={r['error']} parse={r['parse_error']}")


def cmd_mismatch(a):
    """Step 3 -- A8's 391 wrong-clip pairings, unpacked, in the validated shape."""
    out = Path(a.out)
    r2 = json.loads((ROUND2 / "records.json").read_text())
    idx = json.loads(STRIPS_INDEX.read_text())
    acc = [v for v in r2.values() if v.get("description")]
    byrole: dict[str, list] = {}
    for v in acc:
        byrole.setdefault(v["role"], []).append(v)
    jobs = []
    for role, vs in byrole.items():
        shifted = vs[7:] + vs[:7]                      # A8's exact derangement
        for v, other in zip(vs, shifted):
            assert v["clip_id"] != other["clip_id"]
            jobs.append((a.auditor, v["clip_id"], role, v["description"],
                         idx[other["clip_id"]][f"{role}_video"],
                         "auditor_mismatch_control", "MISMATCHED"))
    print(f"[mismatch] {len(jobs)} wrong-clip pairings | auditor={a.auditor}")
    recs = _pool(lambda j: audit_unpacked(*j), jobs, a.workers, "audit")
    _archive(out / "raw_mismatch_responses.jsonl", recs)
    v = [r.get("verdict") or {} for r in recs]
    n = len(v)
    inacc = sum(1 for x in v if x.get("inaccurate") == "YES")
    leak = sum(1 for x in v if x.get("leak") == "YES")
    flag = sum(1 for x in v if x.get("inaccurate") == "YES" or x.get("leak") == "YES")
    res = {"auditor": a.auditor, "n": n, "inaccurate_yes": inacc, "leak_yes": leak,
           "flagged_any": flag, "errors": sum(1 for x in v if not x),
           "inaccurate_pct": round(100 * inacc / n, 2),
           "flagged_pct": round(100 * flag / n, 2),
           "bar": "mismatched flag rate >= 99%",
           "verdict": "PASS" if 100 * flag / n >= 99.0 else "FAIL"}
    (out / "auditor_mismatch_control.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_audit(a):
    """Step 4 -- 200 matched unpacked audits + 20 mixed packs (5 deranged / 5 matched)."""
    out = Path(a.out)
    rows = json.loads((out / "packed_rows.json").read_text())
    plan = json.loads((out / "pack_plan.json").read_text())
    idx = json.loads(STRIPS_INDEX.read_text())
    ok = [r for r in rows if r["description"] and not r["format_violations"] and not r["tier1"]]
    print(f"[audit] {len(ok)}/{len(rows)} rows eligible for audit | auditor={a.auditor} "
          f"| phase={a.phase}")

    # ---- (i) matched, UNPACKED: A8's validated instrument shape ------------
    if a.phase in ("matched", "both"):
        jobs = [(a.auditor, r["clip_id"], r["role"], r["description"],
                 idx[r["clip_id"]][f"{r['role']}_video"], "pilot_matched_audit", "MATCHED")
                for r in ok]
        m = _pool(lambda j: audit_unpacked(*j), jobs, a.workers, "matched")
        _archive(out / "raw_matched_audit_responses.jsonl", m)
    if a.phase == "matched":
        return

    # ---- (ii) packed audits: 5 deranged + 5 matched per pack ---------------
    bycode = {r["code"]: r for r in ok}
    packjobs = []
    for p in plan["packs"][:a.limit] if a.limit else plan["packs"]:
        items = [bycode[i["code"]] for i in p["items"] if i["code"] in bycode]
        if len(items) < 10:
            continue
        items = items[:10]
        # positions 0-4 keep their own video; 5-9 get another in-pack clip's video
        shift = 5
        built = []
        for j, r in enumerate(items):
            if j < shift:
                built.append({"code": r["code"], "clip_id": r["clip_id"], "role": r["role"],
                              "description": r["description"], "truth": "MATCHED",
                              "video": idx[r["clip_id"]][f"{r['role']}_video"]})
            else:
                other = items[(j + 3) % len(items)]
                assert other["clip_id"] != r["clip_id"]
                built.append({"code": r["code"], "clip_id": r["clip_id"], "role": r["role"],
                              "description": r["description"], "truth": "DERANGED",
                              "video": idx[other["clip_id"]][f"{r['role']}_video"],
                              "video_of": other["clip_id"]})
        packjobs.append((a.auditor, p["pack_id"], built, "pilot_packed_audit"))
    print(f"[audit] {len(packjobs)} packed audits "
          f"({sum(1 for j in packjobs for i in j[2] if i['truth']=='DERANGED')} deranged items)")
    pa = _pool(lambda j: audit_pack(*j), packjobs, a.workers, "packed")
    _archive(out / "raw_packed_audit_responses.jsonl", pa)
    print("[audit] done -- run pack_analysis.py for the six conditions")


def cmd_smoke(a):
    out = Path(a.out)
    plan = json.loads((out / "pack_plan.json").read_text())
    p = plan["packs"][0]
    print(f"[smoke] generating pack {p['pack_id']} (K={p['k']}, role {p['role']})")
    g = generate_pack(p)
    u = (g.get("raw_response") or {}).get("usageMetadata") or {}
    print(f"  gen: err={g['error']} parse={g['parse_error']} usage={u} "
          f"modelVersion={(g.get('raw_response') or {}).get('modelVersion')}")
    descs = []
    for j, it in enumerate(g["items"]):
        o = (g["parsed"] or [{}] * p["k"])[j] or {}
        d = postprocess(o.get("description") or "")
        descs.append(d)
        print(f"   [{o.get('id')}=={it['code']}] {word_count(d):>2}w  {d[:96]}")
    idx = json.loads(STRIPS_INDEX.read_text())
    it0 = p["items"][0]
    au = audit_unpacked(a.auditor, it0["clip_id"], it0["role"], descs[0],
                        idx[it0["clip_id"]][f"{it0['role']}_video"], "smoke_audit", "MATCHED")
    print(f"  unpacked audit: err={au['error']} verdict={au.get('verdict')} "
           f"usage={(au.get('raw_response') or {}).get('usageMetadata')} "
           f"modelVersion={(au.get('raw_response') or {}).get('modelVersion')}")
    items = [{"code": i["code"], "clip_id": i["clip_id"], "role": i["role"],
              "description": d, "truth": "MATCHED",
              "video": idx[i["clip_id"]][f"{i['role']}_video"]}
             for i, d in zip(p["items"], descs)]
    ap = audit_pack(a.auditor, p["pack_id"], items, "smoke_packed_audit")
    print(f"  packed audit: err={ap['error']} parse={ap['parse_error']} "
          f"usage={(ap.get('raw_response') or {}).get('usageMetadata')}")
    print(f"   verdicts: {json.dumps(ap.get('parsed'))[:400]}")
    _archive(out / "raw_smoke_responses.jsonl", [g, au, ap])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "smoke", "gen", "mismatch", "audit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--auditor", default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--from-archive", action="store_true",
                    help="re-derive rows from archived responses; makes NO API calls")
    ap.add_argument("--phase", choices=("matched", "packed", "both"), default="both",
                    help="split step 4 so spend can be measured between batches")
    a = ap.parse_args()
    if a.cmd in ("smoke", "mismatch", "audit") and not a.auditor:
        ap.error("--auditor is required (and BANNED: gemini-3.6-flash, gemini-flash-latest)")
    if a.auditor in (GEN_MODEL, "gemini-flash-latest"):
        ap.error(f"auditor {a.auditor} is BANNED by A12 §2")
    if a.cmd == "plan":
        p = build_plan(k=a.k, out=Path(a.out))
        print(f"[plan] {p['n_packs']} packs, {p['n_items']} items -> {a.out}/pack_plan.json")
        for q in p["packs"]:
            arms = {x: sum(1 for i in q["items"] if i["arm"] == x) for x in ("paired", "fresh")}
            print(f"  {q['pack_id']} role={q['role']} bank={q['bank']:<16} N={q['N_target']:>2} "
                  f"paired={arms['paired']} fresh={arms['fresh']} "
                  f"embedded_sha={q['embedded_round2_prompt_sha256'][:12]}")
    elif a.cmd == "smoke":
        cmd_smoke(a)
    elif a.cmd == "gen":
        cmd_gen(a)
    elif a.cmd == "mismatch":
        cmd_mismatch(a)
    elif a.cmd == "audit":
        cmd_audit(a)


if __name__ == "__main__":
    main()
