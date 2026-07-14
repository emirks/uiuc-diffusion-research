"""M4 — the rubric itself, backend-agnostic (pure python/numpy, no torch).

The checklist is the contract; backends (local Gemma in judge.py, Gemini API
in judge_gemini.py) only change how the videos reach the model. Question KEYS
are stable identifiers — result files across backends stay comparable.

q2/q5 carry explicit calibration clauses: under the 8-frame-strip backend both
answered false for every item of every arm (exp_052), i.e. the harsh reading
of "match" / "significant" dominated. The clauses pin the intended severity;
the native-video backend additionally removes the sparse-sampling excuse.
"""

from __future__ import annotations

import json
import re

import numpy as np

RUBRIC_QUESTIONS = {
    "q1_same_type": "Does the GENERATED video contain a transition of the same TYPE as the "
                    "REFERENCE (same kind of effect medium and mechanism — e.g. engulfing black "
                    "smoke, flock of birds, melting)? Judge the mechanism, not the scene content.",
    "q2_dynamics": "Do the dynamics and timing of the generated transition match the reference "
                   "(how the effect enters, travels, and clears; relative duration)? Judge the "
                   "overall character and pace of the motion, not frame-exact correspondence — "
                   "answer true if the effect enters, travels, and clears in the same manner at "
                   "a broadly similar pace.",
    "q3_endpoints": "Are the generated video's OWN start and end scenes preserved and entered/"
                    "exited seamlessly (no warping, popping, or identity drift at either end)?",
    "q4_leakage": "Does the generated video contain objects, people, or backgrounds that appear "
                  "in the REFERENCE video but NOT in the generated video's own start/end scenes? "
                  "(Reference content leaking through is a failure.)",
    "q5_artifacts": "Are there significant visual artifacts (flicker, smearing, duplicated limbs, "
                    "text corruption, frozen frames)? Answer true only for defects a casual "
                    "viewer would notice on one viewing; ignore mild softness, codec blur, or "
                    "brief blending inherent to the transition effect itself.",
}

FAIL_IF_TRUE = ("q4_leakage", "q5_artifacts")


def parse_judge_json(text: str) -> dict:
    """Extract the checklist JSON from a model response; regex fallback for
    backends that wrap it in prose. Always preserves the raw text."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            parsed["_raw"] = text
            return parsed
        except json.JSONDecodeError:
            pass
    return {"parse_error": True, "_raw": text}


def item_pass(result: dict) -> bool | None:
    """One item passes iff every question lands on its good side; None if any
    answer is missing/unparsed (do not silently count as fail)."""
    votes = []
    for q in RUBRIC_QUESTIONS:
        ans = result.get(q, {})
        if not (isinstance(ans, dict) and isinstance(ans.get("answer"), bool)):
            return None
        votes.append((not ans["answer"]) if q in FAIL_IF_TRUE else ans["answer"])
    return all(votes)


def judge_pass_rate(results: list[dict]) -> dict:
    """Aggregate checklist answers; q4/q5 are pass-if-false. Adds `all_pass`
    (fraction of items passing every question — the headline judge number)."""
    rates = {}
    for q in RUBRIC_QUESTIONS:
        vals = []
        for r in results:
            ans = r.get(q, {})
            if isinstance(ans, dict) and isinstance(ans.get("answer"), bool):
                good = (not ans["answer"]) if q in FAIL_IF_TRUE else ans["answer"]
                vals.append(good)
        rates[q] = float(np.mean(vals)) if vals else float("nan")
    passes = [p for p in (item_pass(r) for r in results) if p is not None]
    rates["all_pass"] = float(np.mean(passes)) if passes else float("nan")
    rates["n_parsed"] = sum(1 for r in results if not r.get("parse_error"))
    return rates
