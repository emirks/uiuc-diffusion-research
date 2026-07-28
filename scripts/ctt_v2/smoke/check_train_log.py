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
the four numbers it must recover, plus four negative cases that must trip `T0_parser_sane`.
The self-test runs on EVERY invocation, before the log under test is even opened — a parser
that cannot read a known-good log does not get to judge a new one.

THREE MORE, found by MUTATING the archived log of the passing run (`prove_smoke_gate.py`)
------------------------------------------------------------------------------------------
4. A log that EXISTS but cannot be READ (permissions, a directory, a dead handle) raised an
   uncaught `OSError`, and Python exits **1** for that — the same code this gate uses for a
   DATA failure, which A9 §4's fallback ladder consumes.  `read_log()` makes the read a
   checked step; absent and unreadable are both PARSER_FAIL (exit 2), with the cause named.
5. Under PARTIAL extraction — escapes present but unstrippable, so `RE_STEP` matches nothing
   while `RE_CKPT_FILE` still matches a checkpoint filename — `T3_steps_completed` had just
   enough evidence to evaluate and reported "highest evidenced step N of 30" about a run that
   finished 30/30.  The docstring's contract was right and one check quietly escaped it, so
   T1..T4 are now forced UNEVALUABLE whenever `T0_parser_sane` fails; the suppressed reading is
   kept under `reading_before_suppression` so nothing is hidden.
6. The gate read its own output.  Production points it at the tee'd `train_mixed.log`, but the
   Slurm capture has THIS GATE'S REPORT appended, and un-sliced the phrase "loss is NaN" in the
   old report trips T6 — verified: the capture of the GREEN job 9688835_1 came out FAIL, exit 1.
   The self-test pinned that hazard for `evaluate()`; nothing guarded the CLI path, which now
   slices `trainer_region()` first (a no-op on a trainer-only log).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def read_log(path: Path) -> tuple[str, str | None]:
    """(text, read_error).  NEVER raises: an I/O failure is an INSTRUMENT fact, not a verdict.

    🔴 The gate used to do `log.read_text() if log.exists() else ""`, which is two bugs in one
    line.  A file that exists but cannot be READ (permissions, a directory, a dead NFS handle)
    raised an uncaught `OSError`, and Python's exit code for an uncaught exception is **1** —
    the very code this gate uses for "the TRAINING failed".  A9 §4's fallback ladder consumes
    that exit code, so a `chmod` accident would have been indistinguishable from a data
    failure and could have executed a punishment branch.  And a MISSING file was reported as
    "the log is EMPTY", which names the wrong cause.  Both now surface as their own
    instrument-attributed error, and the verdict is PARSER_FAIL (exit 2), never FAIL.
    """
    if not path.exists():
        return "", f"the log does not exist: {path}"
    try:
        return path.read_text(errors="replace"), None
    except OSError as exc:
        return "", (f"the log EXISTS but could not be read: {type(exc).__name__}: {exc}. "
                    f"This is an instrument/plumbing failure and says NOTHING about the "
                    f"training run")


def _chk(ok, detail, **extra):
    """A check result.  `ok=None` means UNEVALUABLE — the parser could not see the evidence.

    UNEVALUABLE is deliberately NOT False: it must be structurally impossible for "I cannot
    read the log" to come out as "your training is broken".
    """
    return {"ok": ok, "state": "UNEVALUABLE" if ok is None else ("PASS" if ok else "FAIL"),
            "detail": detail, **extra}


# --------------------------------------------------------------------------------------
def evaluate(raw: str, expected: int, target_steps: int,
             read_error: str | None = None) -> tuple[dict, str]:
    """Parse one trainer capture.  Returns (checks, verdict).

    verdict ∈ {PASS, FAIL, PARSER_FAIL}.  PARSER_FAIL means the checks say nothing at all
    about the training — only about this file's ability to read the log.  `read_error`, if
    given, is an I/O failure from `read_log()`: it forces PARSER_FAIL and is quoted verbatim
    in T0, so the cause named is the cause that happened.
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
    parser_sane = not read_error and not empty_log and not missing
    if read_error:
        t0_detail = (f"THE LOG COULD NOT BE READ — {read_error}. The verdict is PARSER_FAIL and "
                     f"every check below is UNEVALUABLE: an unreadable or absent log is an "
                     f"INSTRUMENT failure and must never be reportable as a data failure, "
                     f"because A9 §4's fallback ladder consumes this exit code")
    elif empty_log:
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
                                    n_checkpoint_lines=len(ckpt), read_error=read_error)

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

    # ---- T0 is the gate on T1..T4: PARTIAL evidence may not produce a verdict either -----
    # 🔴 Found by mutation (`prove_smoke_gate.py` L5): with the Rich escapes present but
    # unstrippable, `RE_STEP` matches nothing while `RE_CKPT_FILE` still matches a checkpoint
    # filename — so T3 had just enough to evaluate and reported
    # "[FAIL] T3_steps_completed: highest evidenced step N of 30" about a run that completed
    # 30/30.  That is the SAME false negative as job 9688250_1, surviving in one check after
    # the headline verdict was fixed.  The docstring's contract is "every check that depends on
    # the extraction is reported UNEVALUABLE — never FAIL"; enforce it here rather than trusting
    # each check to notice, because partial extraction is exactly when they cannot.
    if not parser_sane:
        for k in ("T1_fast_index_present", "T2_fast_index_N_of_N", "T3_steps_completed",
                  "T4_losses_finite"):
            c = checks[k]
            if c["ok"] is not None:
                c["suppressed_by_T0"] = True
                c["reading_before_suppression"] = {"ok": c["ok"], "detail": c["detail"]}
                c["ok"], c["state"] = None, "UNEVALUABLE"
                c["detail"] = ("UNEVALUABLE — T0_parser_sane FAILED, so the extraction is "
                               "PARTIAL and cannot support any verdict about the training "
                               "(what it would have said is kept in "
                               "`reading_before_suppression`)")

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
    # partial extraction is the subtle case: RE_STEP matches nothing, RE_CKPT_FILE still
    # matches a checkpoint filename, and T3 used to report "highest evidenced step N of 30"
    # about a 30/30 run.  Every extraction-dependent check must go UNEVALUABLE, not just the
    # ones that happened to extract nothing.
    want(all(cf[k]["ok"] is None for k in ("T1_fast_index_present", "T2_fast_index_N_of_N",
                                           "T3_steps_completed", "T4_losses_finite")),
         "and T1..T4 are ALL UNEVALUABLE under partial extraction, not just the empty ones "
         f"(T3 was {cf['T3_steps_completed'].get('reading_before_suppression')})")

    # --- negative 3: an empty log --------------------------------------------------------
    ce, ve = evaluate("", 10, 30)
    want(ve == "PARSER_FAIL", f"empty log -> PARSER_FAIL (got {ve})")
    want(ce["T4_losses_finite"]["ok"] is None, "empty log -> T4 UNEVALUABLE")

    # --- negative 4: the log cannot be READ (absent, or present-but-unreadable) ----------
    # An uncaught OSError exits 1 — the DATA-failure code A9 §4's ladder consumes. So the
    # read itself must be a checked, instrument-attributed step, not an exception.
    import tempfile  # noqa: PLC0415

    absent_txt, absent_err = read_log(Path(tempfile.gettempdir()) / "ctt2_no_such_log_XYZ.out")
    want(absent_err is not None and absent_txt == "",
         f"an ABSENT log returns a read_error instead of raising (got {absent_err!r})")
    ca, va = evaluate(absent_txt, 10, 30, read_error=absent_err)
    want(va == "PARSER_FAIL", f"absent log -> PARSER_FAIL, not FAIL (got {va})")
    want(ca["T4_losses_finite"]["ok"] is None, "absent log -> T4 UNEVALUABLE")
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fh:
        fh.write("Fast index: 10 valid samples from 10 total\n")
        locked = Path(fh.name)
    try:
        os.chmod(locked, 0o000)
        txt, err = read_log(locked)
        if err is None:                      # running as root, or an exotic filesystem
            rep.append("    [ok ] (skipped: this process can read a chmod-000 file)")
        else:
            want(txt == "", "an UNREADABLE log yields no text and no exception")
            cu, vu = evaluate(txt, 10, 30, read_error=err)
            want(vu == "PARSER_FAIL",
                 f"unreadable (chmod 000) log -> PARSER_FAIL, not FAIL (got {vu})")
            want(cu["T0_parser_sane"]["read_error"] is not None,
                 "T0 names the I/O error as the cause")
            for k in ("T1_fast_index_present", "T2_fast_index_N_of_N", "T3_steps_completed",
                      "T4_losses_finite"):
                want(cu[k]["ok"] is None, f"{k} is UNEVALUABLE on an unreadable log")
    finally:
        os.chmod(locked, 0o600)
        locked.unlink(missing_ok=True)
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
    whole, read_error = read_log(log)
    #: Slice the TRAINER's own region out, exactly as the self-test does.  In production the
    #: gate is pointed at the tee'd `train_mixed.log` and this is a no-op (no banners, so
    #: `trainer_region` returns the whole text).  Pointed at the Slurm `%x-%A_%a.out` capture
    #: — which the sbatch appends THIS GATE'S OWN REPORT to — it is what stops the gate
    #: reading its own output: verified, the un-sliced capture of the GREEN job 9688835_1
    #: reports `[FAIL] T6 ... NaN/Inf mentions: ['loss is NaN']` and exits 1, the data-failure
    #: code, purely from the previous run's report text.  The self-test pinned that hazard for
    #: `evaluate()` but nothing guarded the CLI path.
    raw = trainer_region(whole)
    man = json.loads((Path(a.root) / "SMOKE_ROOT_MANIFEST.json").read_text())
    expected = man["n_samples"]

    checks, verdict = evaluate(raw, expected, a.steps, read_error=read_error)
    rec = {"schema": "ctt_v2_smoke_train_gate/2",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "log": str(log), "log_exists": log.exists(), "root": str(a.root),
           "log_read_error": read_error,
           "log_bytes_read": len(whole), "trainer_region_bytes": len(raw),
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
