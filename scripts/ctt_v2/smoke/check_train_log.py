"""CTT v2 — gate the REAL trainer's log for the mixed-format run (seatbelt 7 + A9 §3).

`REF_mixed_length.md` names the campaign's biggest silent risk: `datasets.py:202-228` joins the
five source trees by identical relative path and DROPS any sample whose match is missing —
`logger.debug` on the happy path, one `logger.info` line among thousands otherwise, and
nothing at all for the reverse direction.  A new arm can vanish entirely while the run looks
healthy.  The mandated mitigation is a hard gate on the trainer's own index line:

    Fast index: N valid samples from N total          with N == expected

This script is that gate, plus the loss-finiteness and step-completion checks A9 §3 item 3
requires of the smoke run.  It parses ONLY the trainer's own stdout — no instrumentation.

    python check_train_log.py --log <log> --root <smoke root> --steps 30 --out <json>
    python check_train_log.py --self-test          # parser-only; no GPU, no root needed

🔴 THE TWO DEFECTS THIS FILE EXISTS IN ORDER NOT TO HAVE AGAIN (job 9688250_1, DOSSIER §13.12)
----------------------------------------------------------------------------------------------
1. The trainer logs through `RichHandler`, so every number arrives wrapped in SGR colour codes
   and OSC-8 hyperlinks — literally `Step \x1b[1;36m20\x1b[0m/\x1b[1;36m30\x1b[0m`, and
   `trainer.py:453` rendered as a clickable link.  Regexing the raw capture matches NOTHING.
2. Worse: the gate then *reported* that empty match set as a DATA failure — "T1 absent",
   "T3 highest step 0 of 30", "T4 a logged loss is NaN/Inf" — about a run that had in fact
   completed 30/30 steps with a finite loss of 0.6540.  `T4 FAIL` sitting next to `T6 PASS`
   ("no NaN/Inf anywhere in the log") is a self-contradiction, and that contradiction is the
   only reason anyone noticed.

So: strip the escapes BEFORE matching (fix 1), and make "I extracted nothing" a distinct,
loud, parser-attributed error that can never be dressed up as a verdict about the training
(fix 2).  `T0_parser_sane` is that error.  When it fires, every check that depends on the
extraction is reported UNEVALUABLE — never FAIL — and the verdict is `PARSER_FAIL`, not
`FAIL`.  A check whose result is decoupled from the thing it checks is the failure mode this
campaign keeps meeting; here it produced a false NEGATIVE rather than the usual vacuous pass.

Fix 3 is the standing proof that fixes 1 and 2 still work: `--self-test` replays the parser
over the real, permanently archived job-9688250_1 capture (Rich escapes and all) and asserts
the four numbers it must recover, plus three negative cases that must trip `T0_parser_sane`.
The self-test runs on EVERY invocation, before the log under test is even opened — a parser
that cannot read a known-good log does not get to judge a new one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

#: The permanent regression fixture: the verbatim stdout of job 9688250_1 — the run whose
#: health this parser originally denied.  Copied into the artefacts dir so it cannot vanish
#: with the Slurm log directory.  Do NOT "clean" it: the Rich escapes ARE the fixture.
FIXTURE = LAB / "misc/ctt_v2_final/artefacts/smoke_gate/fixtures/ctt2_smoke-9688250_1.out"
FIXTURE_SHA256 = "9d899c2ede06243b86ca4296cbb19d177ca2ff8da37c5a6a90c717c5f64ac096"
#: what the parser MUST recover from it (A9 §3 item 3's four numbers)
FIXTURE_EXPECT = {"expected_samples": 10, "steps": 30, "index": (10, 10, 0),
                  "highest_step": 30, "losses": [0.6540]}

RE_INDEX = re.compile(r"Fast index:\s*(\d+)\s*valid samples from\s*(\d+)\s*total"
                      r"(?:\s*\((\d+)\s*skipped\))?")
RE_STEP = re.compile(r"Step\s+(\d+)/(\d+)\s*-\s*Loss:\s*([0-9.eE+\-]+|nan|inf|-inf)", re.I)
#: The trainer logs a loss line only every `logging_steps` (20), so a 30-step run logs exactly
#: ONE.  Step 30 is evidenced instead by the checkpoint the trainer writes at the end —
#: "💾 Lora weights for step 30 saved in checkpoints/lora_weights_step_00030.safetensors".
#: Both are the trainer's own output; neither is instrumentation.
RE_CKPT = re.compile(r"weights for step\s+(\d+)\s+saved", re.I)
RE_CKPT_FILE = re.compile(r"lora_weights_step_0*(\d+)\.safetensors", re.I)
#: only NaN/Inf actually attached to a loss or gradient — a bare `\binf\b` sweep over the whole
#: log matches ordinary words and file paths and produces false FAILs.
RE_NAN = re.compile(r"Loss:\s*-?(?:nan|inf)\b|loss is (?:nan|inf)\b|"
                    r"(?:NaN|Inf) (?:detected|encountered)", re.I)
RE_TRACE = re.compile(r"Traceback \(most recent call last\)")

RE_SGR = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
RE_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(s: str) -> str:
    """Remove Rich's SGR colour runs and OSC-8 hyperlinks. Applied ONCE, to the whole log."""
    return RE_SGR.sub("", RE_OSC.sub("", s))


#: banners the sbatch prints around the trainer's own stdout inside the combined Slurm capture
RE_REGION_START = re.compile(r"^\[smoke\] =====\s*TASK 1\b.*$", re.M)
RE_REGION_END = re.compile(r"^\[smoke\] train exit=", re.M)


def trainer_region(text: str) -> str:
    """Slice the TRAINER's own stdout out of a combined Slurm capture.

    🔴 In production the gate reads `$ART/train_mixed.log`, the tee'd trainer stdout, and so
    never sees anything but the trainer.  The Slurm `%x-%A_%a.out` capture is different: the
    sbatch appends the GATE'S OWN REPORT to it.  Feeding that whole file back into the gate
    self-contaminates — the archived fixture literally ends with the old gate's false
    "[FAIL] T4_losses_finite: a logged loss is NaN/Inf", and `RE_NAN` matches that sentence,
    so T6 would FAIL on a run that had no NaN anywhere.  A checker must never be handed its
    own output.  The self-test therefore parses only the region the trainer wrote, and then
    asserts the un-sliced capture really does trip T6 — so this hazard stays pinned.
    """
    s = RE_REGION_START.search(text)
    e = RE_REGION_END.search(text, s.end() if s else 0)
    return text[(s.end() if s else 0):(e.start() if e else len(text))]


def _chk(ok, detail, **extra):
    """A check result.  `ok=None` means UNEVALUABLE — the parser could not see the evidence.

    UNEVALUABLE is deliberately NOT False: it must be structurally impossible for "I cannot
    read the log" to come out as "your training is broken".
    """
    return {"ok": ok, "state": "UNEVALUABLE" if ok is None else ("PASS" if ok else "FAIL"),
            "detail": detail, **extra}


# --------------------------------------------------------------------------------------
def evaluate(raw: str, expected: int, target_steps: int) -> tuple[dict, str]:
    """Parse one trainer capture.  Returns (checks, verdict).

    verdict ∈ {PASS, FAIL, PARSER_FAIL}.  PARSER_FAIL means the checks say nothing at all
    about the training — only about this file's ability to read the log.
    """
    text = strip_ansi(raw)

    idx = RE_INDEX.findall(text)
    steps = [(int(s), int(t), f) for s, t, f in RE_STEP.findall(text)]
    ckpt = [int(x) for x in RE_CKPT.findall(text)] + [int(x) for x in RE_CKPT_FILE.findall(text)]
    losses = []
    for _, _, f in steps:
        try:
            losses.append(float(f))
        except ValueError:
            losses.append(float("nan"))

    checks: dict[str, dict] = {}

    # ---- T0: the parser must prove it can read THIS log before any check speaks ---------
    stripped_bytes = len(raw) - len(text)
    empty_log = not raw.strip()
    missing = []
    if not idx:
        missing.append("0 'Fast index' lines")
    if not steps:
        missing.append("0 'Step N/M - Loss:' lines")
    if not steps and not ckpt:
        missing.append("0 checkpoint-save lines")
    parser_sane = not empty_log and not missing
    if empty_log:
        t0_detail = ("the log is EMPTY (0 non-whitespace bytes) — nothing to parse; this is a "
                     "job/plumbing failure, and NOTHING below is a statement about training")
    elif missing:
        t0_detail = (f"PARSER EXTRACTED NOTHING: {', '.join(missing)} from a NON-EMPTY log "
                     f"({len(raw)} raw bytes, {len(text)} after stripping {stripped_bytes} "
                     f"bytes of Rich SGR/OSC-8 escapes). This is a failure of THIS PARSER or "
                     f"of the log format — NOT of the training run. Every dependent check "
                     f"below is UNEVALUABLE and the verdict is PARSER_FAIL: an unreadable log "
                     f"must never be reportable as a data failure (job 9688250_1, §13.12)")
    else:
        t0_detail = (f"parser recovered {len(idx)} index line(s), {len(steps)} step/loss "
                     f"line(s), {len(ckpt)} checkpoint line(s) after stripping "
                     f"{stripped_bytes} bytes of Rich SGR/OSC-8 escapes "
                     f"({len(raw)} raw -> {len(text)} clean)")
    checks["T0_parser_sane"] = _chk(parser_sane, t0_detail, raw_bytes=len(raw),
                                    clean_bytes=len(text), ansi_bytes_stripped=stripped_bytes,
                                    n_index_lines=len(idx), n_step_lines=len(steps),
                                    n_checkpoint_lines=len(ckpt))

    # ---- T1..T4 read the extraction, so they go UNEVALUABLE when it is empty ------------
    checks["T1_fast_index_present"] = _chk(
        True if idx else None,
        f"{len(idx)} 'Fast index' line(s) found" if idx else
        "no 'Fast index' line was recovered — see T0. Cannot prove no sample was silently "
        "dropped, and equally cannot claim one was",
        lines=[{"valid": int(v), "total": int(t), "skipped": int(s or 0)} for v, t, s in idx])

    if idx:
        ok_n = all(int(v) == expected and int(t) == expected and int(s or 0) == 0
                   for v, t, s in idx)
        d = (f"every index line reads {expected} of {expected}, 0 skipped" if ok_n else
             f"index line(s) do NOT read {expected} of {expected}: "
             f"{[(int(v), int(t), int(s or 0)) for v, t, s in idx]}")
    else:
        ok_n, d = None, "no index line to check — see T0"
    checks["T2_fast_index_N_of_N"] = _chk(ok_n, d, expected_n=expected)

    from_steps = max((s for s, _, _ in steps), default=0)
    from_ckpt = max(ckpt, default=0)
    reached = max(from_steps, from_ckpt)
    if steps or ckpt:
        t3_ok = reached >= target_steps
        d = (f"highest evidenced step {reached} of {target_steps} (loss lines reach "
             f"{from_steps} — the trainer logs one every 20 steps; checkpoint lines reach "
             f"{from_ckpt})")
    else:
        t3_ok, d = None, "no step or checkpoint evidence recovered — see T0"
    checks["T3_steps_completed"] = _chk(t3_ok, d, steps_logged=len(steps),
                                        highest_step_from_loss_lines=from_steps,
                                        highest_step_from_checkpoints=from_ckpt,
                                        highest_step=reached, target=target_steps)

    if losses:
        finite = all(math.isfinite(x) for x in losses)
        d = ("every logged loss is finite" if finite else
             f"a logged loss is NaN/Inf: {[x for x in losses if not math.isfinite(x)]}")
    else:
        finite, d = None, ("no loss value was recovered — see T0. This is NOT a claim that a "
                           "loss was NaN/Inf; that claim is exactly the false negative of "
                           "job 9688250_1")
    checks["T4_losses_finite"] = _chk(finite, d, n_losses=len(losses), losses=losses)

    # ---- T5/T6 are evidence-of-absence over the cleaned text; they were correct before --
    tb = RE_TRACE.search(text)
    checks["T5_no_traceback"] = _chk(
        not tb, "no traceback in the trainer's output" if not tb else "the trainer raised")
    nan_hits = RE_NAN.findall(text)
    checks["T6_no_nan_attached_to_loss"] = _chk(
        not nan_hits,
        "no NaN/Inf attached to a loss or gradient anywhere in the log" if not nan_hits
        else f"NaN/Inf mentions: {nan_hits[:5]}")

    if not parser_sane:
        verdict = "PARSER_FAIL"
    elif all(v["ok"] for v in checks.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return checks, verdict


# --------------------------------------------------------------------------------------
def self_test() -> tuple[bool, list[str]]:
    """Prove the parser still reads the real job-9688250_1 capture, and still refuses to turn
    an unreadable log into a data verdict.  No GPU, no smoke root, no network."""
    rep: list[str] = []
    ok = True

    def want(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        rep.append(f"    [{'ok ' if cond else 'BAD'}] {msg}")

    # --- positive case: the archived Rich-escaped capture of a HEALTHY run --------------
    if not FIXTURE.exists():
        return False, [f"    [BAD] regression fixture missing: {FIXTURE}"]
    raw_b = FIXTURE.read_bytes()
    sha = hashlib.sha256(raw_b).hexdigest()
    rep.append(f"  fixture {FIXTURE} ({len(raw_b)} bytes, sha256 {sha[:16]}...)")
    want(sha == FIXTURE_SHA256, f"fixture sha256 unchanged ({FIXTURE_SHA256[:16]}...)")
    whole = raw_b.decode(errors="replace")
    want("\x1b[" in whole, "fixture really does contain raw ANSI SGR escapes (the whole point)")
    text = trainer_region(whole)   # the trainer's own stdout, i.e. what train_mixed.log holds
    want(0 < len(text) < len(whole),
         f"sliced the trainer's own region out of the Slurm capture "
         f"({len(whole)} -> {len(text)} bytes)")

    c, v = evaluate(text, FIXTURE_EXPECT["expected_samples"], FIXTURE_EXPECT["steps"])
    want(c["T0_parser_sane"]["ok"] is True, "T0_parser_sane PASS")
    want(c["T0_parser_sane"]["ansi_bytes_stripped"] > 0,
         f"stripped {c['T0_parser_sane']['ansi_bytes_stripped']} bytes of Rich escapes")
    want(c["T1_fast_index_present"]["ok"] is True, "T1_fast_index_present PASS")
    got = c["T1_fast_index_present"]["lines"]
    want(bool(got) and (got[0]["valid"], got[0]["total"], got[0]["skipped"])
         == FIXTURE_EXPECT["index"],
         f"T2 index line reads 10 of 10, 0 skipped (got {got})")
    want(c["T2_fast_index_N_of_N"]["ok"] is True, "T2_fast_index_N_of_N PASS")
    want(c["T3_steps_completed"]["highest_step"] == FIXTURE_EXPECT["highest_step"],
         f"T3 highest evidenced step 30 of 30 (got "
         f"{c['T3_steps_completed']['highest_step']}; loss lines reach "
         f"{c['T3_steps_completed']['highest_step_from_loss_lines']}, checkpoints reach "
         f"{c['T3_steps_completed']['highest_step_from_checkpoints']})")
    want(c["T3_steps_completed"]["ok"] is True, "T3_steps_completed PASS")
    want(c["T4_losses_finite"]["ok"] is True, "T4_losses_finite PASS")
    want(c["T4_losses_finite"]["losses"] == FIXTURE_EXPECT["losses"],
         f"T4 recovered finite loss {FIXTURE_EXPECT['losses']} "
         f"(got {c['T4_losses_finite']['losses']})")
    want(v == "PASS", f"fixture verdict PASS (got {v})")

    # --- the self-contamination hazard, pinned ------------------------------------------
    # The un-sliced Slurm capture ends with the OLD gate's own false report, including the
    # sentence "a logged loss is NaN/Inf". RE_NAN matches it. Assert that, so nobody ever
    # "simplifies" trainer_region() away and re-introduces a checker reading its own output.
    cw, _vw = evaluate(whole, FIXTURE_EXPECT["expected_samples"], FIXTURE_EXPECT["steps"])
    want(cw["T6_no_nan_attached_to_loss"]["ok"] is False,
         "the UN-SLICED Slurm capture trips T6 — because it contains the old gate's own "
         "'a logged loss is NaN/Inf' text; this is why trainer_region() exists")

    # --- negative 1: a NON-EMPTY log the parser cannot read ------------------------------
    # THE load-bearing assertion. The old parser turned this state into "T4 FAIL: a logged
    # loss is NaN/Inf". It must now be PARSER_FAIL with T1..T4 UNEVALUABLE, never False.
    junk = "loading transformer\nsome unrelated output\n" * 40
    cj, vj = evaluate(junk, 10, 30)
    want(vj == "PARSER_FAIL", f"unreadable non-empty log -> PARSER_FAIL (got {vj})")
    want(cj["T0_parser_sane"]["ok"] is False, "T0_parser_sane FAIL on an unreadable log")
    for k in ("T1_fast_index_present", "T2_fast_index_N_of_N", "T3_steps_completed",
              "T4_losses_finite"):
        want(cj[k]["ok"] is None and cj[k]["state"] == "UNEVALUABLE",
             f"{k} is UNEVALUABLE, not FAIL, when the parser saw nothing")

    # --- negative 2: the ORIGINAL bug — escapes present and unstrippable ----------------
    frozen = text.replace("\x1b", "␛")   # neutralise ESC so strip_ansi cannot help
    cf, vf = evaluate(frozen, 10, 30)
    want(vf == "PARSER_FAIL",
         f"the original bug (Rich escapes unstripped) -> PARSER_FAIL, not FAIL (got {vf})")
    want(cf["T4_losses_finite"]["ok"] is not False,
         "and T4 is NOT reported as a NaN/Inf data failure — the exact false negative that "
         "job 9688250_1 produced")

    # --- negative 3: an empty log --------------------------------------------------------
    ce, ve = evaluate("", 10, 30)
    want(ve == "PARSER_FAIL", f"empty log -> PARSER_FAIL (got {ve})")
    want(ce["T4_losses_finite"]["ok"] is None, "empty log -> T4 UNEVALUABLE")
    return ok, rep


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--root")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true",
                    help="run ONLY the parser regression self-test and exit (no GPU/root)")
    a = ap.parse_args()

    # The self-test runs on EVERY invocation, before the log under test is opened.
    st_ok, st_rep = self_test()
    print("[self-test] parser regression fixture + negative cases")
    for line in st_rep:
        print(line)
    print(f"[self-test] {'PASS' if st_ok else 'FAIL'}")
    if a.self_test:
        return 0 if st_ok else 1
    if not st_ok:
        print("VERDICT: PARSER_FAIL — the parser failed its own regression self-test, so it "
              "is NOT reporting on the training run.")
        return 2
    for req in ("log", "root", "steps", "out"):
        if getattr(a, req) in (None, ""):
            ap.error(f"--{req} is required unless --self-test")

    log = Path(a.log)
    raw = log.read_text(errors="replace") if log.exists() else ""
    man = json.loads((Path(a.root) / "SMOKE_ROOT_MANIFEST.json").read_text())
    expected = man["n_samples"]

    checks, verdict = evaluate(raw, expected, a.steps)
    rec = {"schema": "ctt_v2_smoke_train_gate/2",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "log": str(log), "log_exists": log.exists(), "root": str(a.root),
           "expected_samples": expected, "target_steps": a.steps,
           "parser_self_test": {"ok": st_ok, "fixture": str(FIXTURE),
                                "fixture_sha256": FIXTURE_SHA256, "report": st_rep},
           "checks": checks, "VERDICT": verdict}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")

    for k, v in checks.items():
        print(f"[{v['state']:11s}] {k}: {v['detail']}")
    print(f"VERDICT: {verdict} -> {a.out}")
    return 0 if verdict == "PASS" else (2 if verdict == "PARSER_FAIL" else 1)


if __name__ == "__main__":
    sys.exit(main())
