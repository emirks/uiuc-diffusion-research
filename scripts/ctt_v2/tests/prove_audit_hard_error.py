"""PROVE that an unusable Layer-2 audit verdict RAISES and is never scored as a pass.

Advisor A13 step 1, and the campaign's third encounter with the same defect class.

THE DEFECT (DOSSIER §21).  When the auditor was down, `_post` returned `(None, err)`,
`audit_one` left `verdict = None`, and the caller wrote:

    v = arec.get("verdict") or {}                      # <-- None becomes {}
    if v.get("leak") == "YES" or v.get("inaccurate") == "YES":   # neither fires
        ...
    # falls through to ACCEPT

so an auditor outage minted descriptions that LOOK audited and are not.  A checker whose
failure is indistinguishable from a pass is worse than no checker, because it also
manufactures the evidence that it worked.

This harness is evidence rather than assertion.  It does three things:

  1. drives the pure validator `validate_audit_verdict` through EVERY failure mode plus
     in-domain positive controls, requiring a raise on each failure and a clean verdict on
     each pass;
  2. re-evaluates the OLD caller expression on the very same records, and requires that it
     would have returned a clean pass -- so the regression this guards against is
     demonstrated, not merely described;
  3. runs the REAL `run()` driver end-to-end over a scripted network boundary, because the
     defect lived in the CALLER, not in the validator: a correct validator that the driver
     forgets to consult would still ship an unaudited store.  Both the fatal path (must
     abort) and the happy path (must accept, and must still flag a genuine leak) are run.

    python scripts/ctt_v2/tests/prove_audit_hard_error.py

Exit 0 = every case proved.  No network, no API key, no GPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "captions"))
sys.path.insert(0, str(HERE.parent))

import generate_descriptions as gd  # noqa: E402

RESULTS: list[dict] = []


def _record(case: str, kind: str, ok: bool, detail: str) -> None:
    RESULTS.append({"case": case, "kind": kind, "ok": ok, "detail": detail})
    print(f"  [{'ok ' if ok else 'FAIL'}] {kind:<14} {case}: {detail}")


# ======================================================================================
# 1. the pure validator -- every failure mode must raise
# ======================================================================================
def _rec(**kw) -> dict:
    """An audit record shaped exactly as `audit_one` builds it."""
    base = {
        "clip_id": "clip_x", "role": "A", "model": gd.AUDIT_MODEL,
        "temperature": 0.0, "error": None, "raw_response": {"candidates": []},
        "verdict": {"leak": "NO", "inaccurate": "NO", "errors": []},
    }
    base.update(kw)
    return base


#: (name, record, why it must be fatal)
FATAL_CASES = [
    ("http_500_exhausted",
     _rec(error="exhausted_retries:HTTP503:model overloaded", raw_response=None, verdict=None),
     "the actual §21 outage: 5xx retried out"),
    ("http_non_200",
     _rec(error="HTTP400:invalid thinkingLevel", raw_response=None, verdict=None),
     "non-200 rejected by the API"),
    ("no_response_object",
     _rec(raw_response=None, verdict=None),
     "no response object at all"),
    ("empty_verdict_none",
     _rec(verdict=None),
     "200 OK but the model emitted no text"),
    ("empty_verdict_dict",
     _rec(verdict={}),
     "parsed to an EMPTY object -- the literal 'empty verdict'"),
    ("unparseable",
     _rec(verdict=None, parse_error="I'm sorry, I cannot analyse this video."),
     "prose instead of JSON"),
    ("truncated_json",
     _rec(verdict=None, parse_error='{"leak": "NO", "inacc'),
     "verdict truncated by the token cap"),
    ("missing_inaccurate",
     _rec(verdict={"leak": "NO", "errors": []}),
     "only one of the two required fields"),
    ("missing_leak",
     _rec(verdict={"inaccurate": "NO", "errors": []}),
     "only one of the two required fields"),
    ("both_fields_missing",
     _rec(verdict={"errors": []}),
     "schema honoured in shape only"),
    ("out_of_domain_value",
     _rec(verdict={"leak": "MAYBE", "inaccurate": "NO", "errors": []}),
     "value outside {YES,NO}"),
    ("null_field",
     _rec(verdict={"leak": None, "inaccurate": "NO", "errors": []}),
     "field present but null"),
    ("verdict_not_an_object",
     _rec(verdict=["NO", "NO"]),
     "JSON parsed to a list"),
    ("verdict_is_string",
     _rec(verdict="NO"),
     "JSON parsed to a bare string"),
]

#: verdicts that MUST be honoured -- the fix must not turn the auditor into a rubber stamp
PASS_CASES = [
    ("clean", _rec(verdict={"leak": "NO", "inaccurate": "NO", "errors": []}), "NO", "NO"),
    ("leak_yes", _rec(verdict={"leak": "YES", "inaccurate": "NO", "errors": ["dissolve"]}),
     "YES", "NO"),
    ("inaccurate_yes", _rec(verdict={"leak": "NO", "inaccurate": "YES", "errors": ["red/blue"]}),
     "NO", "YES"),
    ("both_yes", _rec(verdict={"leak": "YES", "inaccurate": "YES", "errors": ["x"]}),
     "YES", "YES"),
    ("no_errors_key", _rec(verdict={"leak": "NO", "inaccurate": "NO"}), "NO", "NO"),
]


def prove_validator() -> None:
    print("\n[1] the pure validator: every unusable verdict must RAISE")
    for name, rec, why in FATAL_CASES:
        try:
            got = gd.validate_audit_verdict(rec)
        except gd.AuditError as e:
            _record(name, "raises", True, f"{why} -> AuditError({str(e)[:70]})")
        except Exception as e:  # noqa: BLE001
            _record(name, "raises", False,
                    f"raised {type(e).__name__}, not AuditError: {e}")
        else:
            _record(name, "raises", False,
                    f"DID NOT RAISE -- returned {got!r} (this is the §21 defect)")

    print("\n[1b] in-domain verdicts must survive unchanged (no rubber-stamping)")
    for name, rec, leak, inacc in PASS_CASES:
        try:
            v = gd.validate_audit_verdict(rec)
        except gd.AuditError as e:
            _record(name, "accepts", False, f"wrongly raised: {e}")
        else:
            ok = v.get("leak") == leak and v.get("inaccurate") == inacc
            _record(name, "accepts", ok, f"leak={v.get('leak')} inaccurate={v.get('inaccurate')}")


# ======================================================================================
# 2. the OLD expression, on the same records -- the regression, demonstrated
# ======================================================================================
def prove_old_expression_passed() -> None:
    """Show what the pre-fix caller did with these very records.

    Not decoration: it is the difference between "we changed some code" and "we can show
    the old code called an outage clean".  Every fatal case whose verdict is falsy or
    field-incomplete was previously ACCEPTED.
    """
    print("\n[2] the pre-fix caller expression `v = arec.get('verdict') or {}`")
    silently_passed = []
    for name, rec, _why in FATAL_CASES:
        v = rec.get("verdict") or {}
        if not isinstance(v, dict):
            continue  # the old code would have crashed on .get, not passed
        old_accepts = not (v.get("leak") == "YES" or v.get("inaccurate") == "YES")
        if old_accepts:
            silently_passed.append(name)
    ok = len(silently_passed) >= 12
    _record("old_expression", "regression", ok,
            f"{len(silently_passed)}/{len(FATAL_CASES)} unusable verdicts were scored as "
            f"CLEAN PASSES before the fix: {', '.join(silently_passed[:4])}...")


# ======================================================================================
# 3. end-to-end through the REAL driver -- the caller must not swallow it
# ======================================================================================
GOOD_DESC = ("A woman with long brown hair in a red jacket walks along a quiet street "
             "beside parked cars under gray afternoon light.")


def _gen_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}],
            "modelVersion": gd.GEN_MODEL}


def _audit_response(payload: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": payload}]}}],
            "modelVersion": gd.AUDIT_MODEL}


class _Harness:
    """Replaces the network and video-file boundaries; everything else is the real code."""

    def __init__(self, audit_behaviour):
        self.audit_behaviour = audit_behaviour
        self.audit_calls = 0

    def post(self, model, body, timeout=240, max_tries=5):
        if model == gd.GEN_MODEL:
            return _gen_response(GOOD_DESC), None
        self.audit_calls += 1
        return self.audit_behaviour(self.audit_calls)


def _run_driver(tmp: Path, audit_behaviour, tag: str):
    """Drive the real `gd.run()` over one (clip, role) pair."""
    index = {"clip_x": {"bank": "humanvid",
                        "A_video": str(tmp / "a.mp4"), "B_video": str(tmp / "b.mp4")}}
    idx_path = tmp / f"strips_{tag}.json"
    idx_path.write_text(json.dumps(index))

    h = _Harness(audit_behaviour)
    orig_index, orig_post, orig_b64 = gd.STRIPS_INDEX, gd._post, gd._b64
    gd.STRIPS_INDEX = idx_path
    gd._post = h.post
    gd._b64 = lambda path: "QUJD"
    gd._stop.clear()
    try:
        out = tmp / tag
        res = gd.run([("clip_x", "A")], out, seed=42, workers=1, max_attempts=2,
                     audit=True, variant="v3")
        return res, out, h, None
    except gd.AuditError as e:
        return None, tmp / tag, h, e
    finally:
        gd.STRIPS_INDEX, gd._post, gd._b64 = orig_index, orig_post, orig_b64
        gd._stop.clear()


def prove_end_to_end(tmp: Path) -> None:
    print("\n[3] end-to-end through the real run() driver")

    # -- 3a. the §21 outage: every audit call fails ------------------------------------
    res, out, h, exc = _run_driver(
        tmp, lambda n: (None, "exhausted_retries:HTTP503:overloaded"), "outage")
    _record("driver_aborts_on_outage", "end2end", exc is not None and res is None,
            f"AuditError propagated out of run(): {str(exc)[:70]}"
            if exc else "run() COMPLETED on a total auditor outage -- the §21 defect is live")
    # the raw response must still be archived, even though the run died
    arch = out / "raw_audit_responses.jsonl"
    lines = arch.read_text().splitlines() if arch.exists() else []
    _record("archives_before_raising", "end2end", len(lines) >= 1,
            f"{len(lines)} raw audit record(s) on disk despite the abort")
    # and no store may be written by an aborted run
    _record("no_store_from_aborted_run", "end2end", not (out / "descriptions.json").exists(),
            "descriptions.json was NOT written by the aborted run")

    # -- 3b. empty text, 200 OK (the subtler outage) -----------------------------------
    _res, _out, _h, exc = _run_driver(tmp, lambda n: (_audit_response(""), None), "empty")
    _record("driver_aborts_on_empty_text", "end2end", exc is not None,
            f"AuditError on a 200-OK empty verdict: {str(exc)[:60]}"
            if exc else "an EMPTY verdict was accepted as a pass")

    # -- 3c. verdict missing a required field ------------------------------------------
    _res, _out, _h, exc = _run_driver(
        tmp, lambda n: (_audit_response('{"leak": "NO", "errors": []}'), None), "partial")
    _record("driver_aborts_on_missing_field", "end2end", exc is not None,
            f"AuditError on a half-filled verdict: {str(exc)[:60]}"
            if exc else "a verdict missing `inaccurate` was accepted as a pass")

    # -- 3d. HAPPY PATH: a clean verdict must still produce a stored description --------
    res, out, h, exc = _run_driver(
        tmp, lambda n: (_audit_response('{"leak":"NO","inaccurate":"NO","errors":[]}'), None),
        "clean")
    stored = json.loads((out / "descriptions.json").read_text()) if res else {}
    ok = exc is None and stored.get("clip_x", {}).get("A")
    _record("driver_accepts_clean_audit", "end2end", bool(ok),
            f"description stored, {h.audit_calls} audit call(s)"
            if ok else f"happy path broken: exc={exc} stored={stored}")
    if res:
        rec = res["clip_x|A"]
        _record("clean_verdict_recorded", "end2end", rec.get("audit") == {
            "leak": "NO", "inaccurate": "NO", "errors": []},
            f"audit field carries the real verdict: {rec.get('audit')}")

    # -- 3e. a genuine leak must still route to the manual queue, not abort -------------
    def leak_then_clean(n):
        if n == 1:
            return _audit_response('{"leak":"YES","inaccurate":"NO","errors":["dissolve"]}'), None
        return _audit_response('{"leak":"NO","inaccurate":"NO","errors":[]}'), None

    res, out, h, exc = _run_driver(tmp, leak_then_clean, "leak")
    rec = res["clip_x|A"] if res else {}
    ok = (exc is None and rec.get("accepted_on_attempt") == 2
          and rec["history"][0].get("fail") == ["leak"])
    _record("real_leak_still_regenerates", "end2end", bool(ok),
            f"attempt 1 failed on leak, accepted on attempt {rec.get('accepted_on_attempt')}"
            if ok else f"leak handling changed: exc={exc} rec={rec.get('accepted_on_attempt')}")


# ======================================================================================
def main() -> int:
    print("PROVE: an unusable Layer-2 audit verdict is a HARD ERROR (A13 step 1)")
    print(f"       auditor pin = {gd.AUDIT_MODEL} | temp {gd.AUDIT_TEMPERATURE} | "
          f"thinkingLevel {gd.AUDIT_THINKING_LEVEL!r} | max_output_tokens {gd.AUDIT_MAX_TOKENS}")

    # Config invariants that hold under ANY auditor.  The MODEL is deliberately not
    # asserted: it is contested between two concurrent advisor rulings (DOSSIER §23.2) and
    # is env-overridable, so hard-coding one choice here would make this file fail
    # spuriously the moment the other lane sets `CTT_AUDIT_MODEL` -- turning the proof of a
    # broken-checker fix into a broken checker.  The pin is REPORTED, and the properties
    # that must hold whatever it is are PROVEN.
    _record("auditor_model", "config", bool(gd.AUDIT_MODEL), f"pinned = {gd.AUDIT_MODEL}")
    _record("token_cap", "config", gd.AUDIT_MAX_TOKENS >= 512, str(gd.AUDIT_MAX_TOKENS))
    _record("temperature", "config", gd.AUDIT_TEMPERATURE == 0.0, str(gd.AUDIT_TEMPERATURE))
    # "minimal" is rejected HTTP 400 by the pro tier and is the cheap default on flash
    _record("thinking_level", "config",
            gd.AUDIT_THINKING_LEVEL == ("low" if "pro" in gd.AUDIT_MODEL else "minimal"),
            f"{gd.AUDIT_THINKING_LEVEL!r} for {gd.AUDIT_MODEL}")

    prove_validator()
    prove_old_expression_passed()
    with tempfile.TemporaryDirectory() as td:
        prove_end_to_end(Path(td))

    bad = [r for r in RESULTS if not r["ok"]]
    report = HERE / "PROVE_AUDIT_HARD_ERROR.json"
    report.write_text(json.dumps({
        "auditor_pin": {"model": gd.AUDIT_MODEL, "temperature": gd.AUDIT_TEMPERATURE,
                        "thinking_level": gd.AUDIT_THINKING_LEVEL,
                        "max_output_tokens": gd.AUDIT_MAX_TOKENS},
        "n_cases": len(RESULTS), "n_failed": len(bad), "results": RESULTS,
    }, indent=1))

    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} cases proved -> {report}")
    if bad:
        print("PROBLEMS:")
        for r in bad:
            print(f"   - {r['case']}: {r['detail']}")
        return 1
    print("EVERY UNUSABLE AUDIT VERDICT IS PROVEN TO RAISE; NONE IS SCORED AS A PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
