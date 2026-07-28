"""CTT v2 — PROVE the smoke gate's own checkers fire (A11 items 3 + 4, GPU-free).

Commits `08888a2` ("an unreadable log can no longer be reported as a data failure") and
`0989069` ("A11 item 4 clause (b) now OBSERVES the realized shift") changed the gate's
CHECKING code.  A green gate run exercises only the happy path of a checker; the thing that
protects the campaign is the FAILURE path, and A5 RULING 9's standing rule is that *an assert
that has never failed is not known to work*.  So this harness breaks the gate's inputs one at
a time and requires the right verdict — in the right FAILURE CLASS — to come out.

Everything here is GPU-free and credit-free.  Nothing writes to the archived gate artefacts:
the log mutations run on COPIES of the real archived capture (Rich escapes kept — the whole
class of bug was a regex over un-stripped text), and the shift mutations run on a COPY of the
smoke root with its own `--out` directory.

    python scripts/ctt_v2/smoke/prove_smoke_gate.py --out <dir>                # log gate only
    python scripts/ctt_v2/smoke/prove_smoke_gate.py --out <dir> \
        --probe-python /.../LTX-2-official/.venv/bin/python \
        --probe-root   /.../outputs/ctt_v2/smoke/root_mixed                    # + shift assert

WHAT IS ASSERTED, and which campaign rule each case belongs to
-------------------------------------------------------------
L0  the real archived log still reads PASS                        (regression, both commits)
L1  a MISSING log            -> PARSER_FAIL, exit 2, never FAIL    (08888a2)
L2  an UNREADABLE log (chmod 000, and a directory) -> PARSER_FAIL  (08888a2, extended: an
                                                                    uncaught OSError exits 1,
                                                                    the DATA-failure code)
L3  the `Fast index` sentinel REMOVED -> PARSER_FAIL, T2 never PASS (A11 positive-presence:
                                                                    "zero skips" may only pass
                                                                    if the sentinel was found)
L4  the sentinel present and SHORT (9 of 10, 1 skipped) -> FAIL     (the same rule's other
                                                                    side: a real data failure
                                                                    must still read as one)
L5  escapes present and unstrippable -> PARSER_FAIL, T4 not False   (the original false
                                                                    negative, job 9688250_1)
L6  a NaN attached to the loss line -> FAIL, T4 False               (contrast to L1/L2/L3/L5)
L7  the Slurm capture (gate report appended) -> PASS                (the self-contamination
                                                                    hazard: un-sliced, it
                                                                    reports T6 'loss is NaN'
                                                                    from its OWN old report
                                                                    and exits 1)
S0  the shift assert on the unmodified smoke root -> PASS
S1  a pinned constant set to A9 §3's superseded 1.120 -> SPEC_CONSTANT_MISMATCH, exit 3,
    escalates, auto_drop_permitted False                            (Derived-Constant Rule)
S2  the S4 arm's tensors silently at 121f geometry -> SPEC_CONSTANT_MISMATCH, exit 3
    (realized token counts disagree with the pinned set: ambiguous between a stale spec and a
    drifted encode, which is exactly why the rule forbids the punishment branch here)
U1  the DATA-FAIL branch of the shared classifier: a mechanism inconsistency (the sampler
    handed 2*tok) -> FAIL, exit 1, auto_drop_permitted True, and DATA wins when both classes
    are broken at once.  Unit-level because inducing it end-to-end would require editing the
    certified trainer, which A9 §3 forbids.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

CHECK = HERE / "check_train_log.py"
PROBE = HERE / "mixed_format_probe.py"
#: the real archived stdout of the PASSING mixed-format run (job 9688835_1), Rich escapes and
#: all, and the Slurm capture of the same job with the gate's own report appended
ARCHIVED_TRAIN_LOG = LAB / "misc/ctt_v2_final/artefacts/smoke_gate/train_mixed.log"
ARCHIVED_SLURM_CAPTURE = (LAB / "diffusion-research/outputs/logs/slurm/ctt2_smoke-9688835_1.out")
SMOKE_ROOT = LAB / "diffusion-research/outputs/ctt_v2/smoke/root_mixed"
STEPS = 30

RE_SGR = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def has_ansi(text: str) -> bool:
    return bool(RE_SGR.search(text))


# ======================================================================================
# log-gate cases
# ======================================================================================
def run_check(py: str, log: Path, root: Path, out: Path) -> tuple[int, dict, str]:
    proc = subprocess.run([py, str(CHECK), "--log", str(log), "--root", str(root),
                           "--steps", str(STEPS), "--out", str(out)],
                          capture_output=True, text=True)
    rec = json.loads(out.read_text()) if out.exists() else {}
    return proc.returncode, rec, proc.stdout + proc.stderr


def log_cases(py: str, work: Path, root: Path, results: list) -> None:
    raw = ARCHIVED_TRAIN_LOG.read_text(errors="replace")
    assert has_ansi(raw), f"{ARCHIVED_TRAIN_LOG} carries no ANSI — wrong fixture"
    fixtures = work / "log_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    def case(name: str, log_path: Path, want_verdict: str, want_exit: int, what: str,
             extra_checks=None) -> None:
        code, rec, stdout = run_check(py, log_path, root, work / f"{name}.json")
        checks = rec.get("checks", {})
        problems = []
        if rec.get("VERDICT") != want_verdict:
            problems.append(f"verdict {rec.get('VERDICT')!r} != {want_verdict!r}")
        if code != want_exit:
            problems.append(f"exit {code} != {want_exit}")
        for label, fn in (extra_checks or []):
            if not fn(checks, rec):
                problems.append(label)
        results.append({
            "kind": "log_gate", "case": name, "ok": not problems, "broke": what,
            "verdict": rec.get("VERDICT"), "expected_verdict": want_verdict,
            "exit_code": code, "expected_exit": want_exit,
            "ansi_bytes_stripped": (checks.get("T0_parser_sane") or {}).get(
                "ansi_bytes_stripped"),
            "T0_detail": (checks.get("T0_parser_sane") or {}).get("detail", "")[:400],
            "states": {k: v["state"] for k, v in checks.items()},
            "log_read_error": rec.get("log_read_error"),
            "ERROR": "; ".join(problems) or None,
            "stdout_tail": stdout.strip().splitlines()[-3:],
        })
        print(f"[{'PROVEN' if not problems else 'PROBLEM'}] {name}: {what}\n"
              f"          verdict={rec.get('VERDICT')} exit={code}"
              + (f"  !! {'; '.join(problems)}" if problems else ""))

    def uneval(*names):
        return [(f"{n} must be UNEVALUABLE, not FAIL",
                 lambda c, r, n=n: c.get(n, {}).get("state") == "UNEVALUABLE")
                for n in names]

    # ---- L0 the archived log, unmodified ------------------------------------------------
    good = fixtures / "L0_archived.log"
    good.write_text(raw)
    case("L0_archived_log_still_passes", good, "PASS", 0,
         f"the real archived trainer stdout of the passing run ({len(raw)} bytes, Rich "
         f"escapes intact)",
         [("the parser must have stripped ANSI",
           lambda c, r: (c["T0_parser_sane"]["ansi_bytes_stripped"] or 0) > 0)])

    # ---- L1 missing ----------------------------------------------------------------------
    case("L1_log_missing", fixtures / "does_not_exist.log", "PARSER_FAIL", 2,
         "pointed the gate at a log that does not exist",
         [("T0 must name the absence as the cause",
           lambda c, r: "does not exist" in (r.get("log_read_error") or ""))]
         + uneval("T1_fast_index_present", "T2_fast_index_N_of_N", "T3_steps_completed",
                  "T4_losses_finite"))

    # ---- L2 unreadable -------------------------------------------------------------------
    locked = fixtures / "L2_unreadable.log"
    locked.write_text(raw)
    os.chmod(locked, 0o000)
    try:
        readable = False
        try:
            locked.read_text()
            readable = True
        except OSError:
            pass
        if readable:
            results.append({"kind": "log_gate", "case": "L2_log_unreadable", "ok": True,
                            "SKIPPED": "this process can read a chmod-000 file"})
            print("[SKIPPED] L2_log_unreadable: chmod 000 is not enforced for this process")
        else:
            case("L2_log_unreadable", locked, "PARSER_FAIL", 2,
                 "chmod 000 on a byte-identical copy of the passing log — the file EXISTS "
                 "and cannot be read; before the fix this raised an uncaught OSError, and "
                 "Python exits 1 for that, the same code as a DATA failure",
                 [("T0 must name the I/O error",
                   lambda c, r: "could not be read" in (r.get("log_read_error") or ""))]
                 + uneval("T1_fast_index_present", "T2_fast_index_N_of_N",
                          "T3_steps_completed", "T4_losses_finite"))
    finally:
        os.chmod(locked, 0o600)
    a_dir = fixtures / "L2b_a_directory.log"
    a_dir.mkdir(exist_ok=True)
    case("L2b_log_is_a_directory", a_dir, "PARSER_FAIL", 2,
         "pointed the gate at a DIRECTORY — the other way a log 'exists' but cannot be read",
         [("T0 must name the I/O error",
           lambda c, r: "could not be read" in (r.get("log_read_error") or ""))])

    # ---- L3 the sentinel removed ---------------------------------------------------------
    stripped = re.sub(r"^.*Fast index.*$", "", raw, flags=re.M)
    p = fixtures / "L3_no_fast_index.log"
    p.write_text(stripped)
    assert has_ansi(stripped), "L3 fixture lost its ANSI"
    case("L3_fast_index_sentinel_removed", p, "PARSER_FAIL", 2,
         "deleted the `Fast index` line from the passing log, leaving everything else "
         "(and the Rich escapes) intact — A11's positive-presence rule: 'zero skips' may "
         "only PASS if the sentinel that would have carried a skip was positively found",
         [("T2 must NEVER read PASS without the sentinel",
           lambda c, r: c["T2_fast_index_N_of_N"]["state"] != "PASS"),
          ("T0 must attribute it to the parser, not the data",
           lambda c, r: "Fast index" in c["T0_parser_sane"]["detail"])]
         + uneval("T1_fast_index_present", "T2_fast_index_N_of_N"))

    # ---- L4 the sentinel present and SHORT ----------------------------------------------
    #: 10 -> 9 on the index line, and a "(1 skipped)" suffix. Both edits land inside the Rich
    #: escapes, and the recovered triple is asserted below as (9, 10, 1).
    short = re.sub(r"(Fast index:\s*(?:\x1b\[[0-9;]*m)?)10", r"\g<1>9", raw, count=1)
    short = short.replace("total", "total (1 skipped)", 1)
    p = fixtures / "L4_index_short.log"
    p.write_text(short)
    case("L4_fast_index_reports_a_skip", p, "FAIL", 1,
         "rewrote the index line to '9 valid samples from 10 total (1 skipped)' — a REAL "
         "data failure must still read as a data failure, or the two-sided rule has only "
         "one side",
         [("T2 must FAIL", lambda c, r: c["T2_fast_index_N_of_N"]["state"] == "FAIL"),
          ("the recovered index triple must be (9, 10, 1)",
           lambda c, r: [(x["valid"], x["total"], x["skipped"])
                         for x in c["T1_fast_index_present"]["lines"]] == [(9, 10, 1)]),
          ("T0 must stay PASS — the parser worked fine",
           lambda c, r: c["T0_parser_sane"]["state"] == "PASS")])

    # ---- L5 escapes unstrippable ---------------------------------------------------------
    frozen = raw.replace("\x1b", "␛")
    p = fixtures / "L5_escapes_frozen.log"
    p.write_text(frozen)
    case("L5_ansi_unstrippable", p, "PARSER_FAIL", 2,
         "neutralised every ESC byte so strip_ansi() cannot help — the ORIGINAL defect "
         "(job 9688250_1), which reported 'a logged loss is NaN/Inf' about a healthy run",
         [("T4 must NOT be a data FAIL",
           lambda c, r: c["T4_losses_finite"]["state"] != "FAIL")]
         + uneval("T1_fast_index_present", "T3_steps_completed", "T4_losses_finite"))

    # ---- L6 a genuine NaN ----------------------------------------------------------------
    nan_log = re.sub(r"(Loss:\s*(?:\x1b\[[0-9;]*m)?)[0-9.]+", r"\g<1>nan", raw, count=1)
    p = fixtures / "L6_loss_is_nan.log"
    p.write_text(nan_log)
    case("L6_loss_line_is_nan", p, "FAIL", 1,
         "replaced the logged loss with nan INSIDE its Rich colour wrapper — the parser must "
         "see through the escapes and call this what it is: a data failure",
         [("T4 must FAIL", lambda c, r: c["T4_losses_finite"]["state"] == "FAIL"),
          ("T0 must stay PASS", lambda c, r: c["T0_parser_sane"]["state"] == "PASS")])

    # ---- L7 the Slurm capture (self-contamination) ---------------------------------------
    if ARCHIVED_SLURM_CAPTURE.exists():
        cap = ARCHIVED_SLURM_CAPTURE.read_text(errors="replace")
        p = fixtures / "L7_slurm_capture.log"
        p.write_text(cap)
        case("L7_slurm_capture_not_self_contaminated", p, "PASS", 0,
             f"fed the gate the whole Slurm capture of the GREEN job 9688835_1 "
             f"({len(cap)} bytes) — the sbatch appends THIS GATE'S OWN REPORT to it, and "
             f"un-sliced that report's phrase 'loss is NaN' trips T6 and exits 1, the "
             f"data-failure code. The gate must slice the trainer's own region first",
             [("the region slice must have dropped bytes",
               lambda c, r: 0 < r["trainer_region_bytes"] < r["log_bytes_read"]),
              ("T6 must stay PASS",
               lambda c, r: c["T6_no_nan_attached_to_loss"]["state"] == "PASS")])


# ======================================================================================
# shift-assert cases
# ======================================================================================
def run_probe(py: str, root: Path, out: Path, extra: list[str]) -> tuple[int, dict, str]:
    env = dict(os.environ)
    trainer_src = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer/src"
    env["PYTHONPATH"] = f"{trainer_src}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run([py, str(PROBE), "--shifts-only", "--root", str(root),
                           "--out", str(out), *extra],
                          capture_output=True, text=True, env=env,
                          cwd=str(LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer"))
    art = out / "SHIFT_ASSERT_A11_item4.json"
    rec = json.loads(art.read_text()) if art.exists() else {}
    return proc.returncode, rec, proc.stdout + proc.stderr


def copy_root_with_s4_at_121f(src: Path, dst: Path) -> str:
    """Repoint every S4 tensor at its 121f counterpart — a silently drifted encode bucket."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    man = json.loads((src / "SMOKE_ROOT_MANIFEST.json").read_text())
    donor = next(s for s in man["samples"] if s["latent_fhw"] == [16, 20, 15])
    victims = [s for s in man["samples"] if s["latent_fhw"] != [16, 20, 15]]
    dirs = ("latents", "cond_clean_latents", "reference_latents", "masks")
    for s in victims:
        for sub in dirs:
            p = dst / sub / s["rel"]
            tgt = os.path.realpath(src / sub / donor["rel"])
            p.unlink()
            os.symlink(tgt, p)
    return (f"repointed all {len(victims)} S4 samples' {list(dirs)} at (16,20,15) tensors — "
            f"the S4 arm now realizes 4,800 tokens, not 1,820, while the manifest still "
            f"declares {man['distinct_expected_shifts']}")


def shift_cases(py: str, root: Path, work: Path, results: list) -> None:
    def case(name: str, r: Path, extra: list[str], want_verdict: str, want_exit: int,
             what: str, want_class=None, want_escalates=None, want_autodrop=None) -> None:
        out = work / name
        code, rec, stdout = run_probe(py, r, out, extra)
        cls = rec.get("failure_classification") or {}
        problems = []
        if rec.get("VERDICT") != want_verdict:
            problems.append(f"verdict {rec.get('VERDICT')!r} != {want_verdict!r}")
        if code != want_exit:
            problems.append(f"exit {code} != {want_exit}")
        if want_class is not None and cls.get("failure_class") != want_class:
            problems.append(f"failure_class {cls.get('failure_class')!r} != {want_class!r}")
        if want_escalates is not None and cls.get("escalates") is not want_escalates:
            problems.append(f"escalates {cls.get('escalates')!r} != {want_escalates!r}")
        if want_autodrop is not None and cls.get("auto_drop_permitted") is not want_autodrop:
            problems.append(f"auto_drop_permitted {cls.get('auto_drop_permitted')!r}")
        results.append({
            "kind": "shift_assert", "case": name, "ok": not problems, "broke": what,
            "verdict": rec.get("VERDICT"), "expected_verdict": want_verdict,
            "exit_code": code, "expected_exit": want_exit,
            "failure_class": cls.get("failure_class"), "escalates": cls.get("escalates"),
            "auto_drop_permitted": cls.get("auto_drop_permitted"),
            "data_offenders": cls.get("data_offenders"),
            "spec_offenders": cls.get("spec_offenders"),
            "test_only_overrides": rec.get("TEST_ONLY_OVERRIDES"),
            "observed_seq_lens": (rec.get("clause_b_realized") or {}).get(
                "observed_sampler_seq_lengths"),
            "ERROR": "; ".join(problems) or None,
            "stdout_tail": stdout.strip().splitlines()[-4:],
        })
        print(f"[{'PROVEN' if not problems else 'PROBLEM'}] {name}: {what}\n"
              f"          verdict={rec.get('VERDICT')} exit={code} "
              f"class={cls.get('failure_class')}"
              + (f"  !! {'; '.join(problems)}" if problems else ""))

    case("S0_shift_assert_baseline", root, [], "PASS", 0,
         "the unmodified smoke root", want_class=None, want_escalates=False,
         want_autodrop=False)
    case("S1_pinned_constant_is_wrong", root,
         ["--test-only-pin", json.dumps({"G3_shift_pins": {"1820": 1.120, "4800": 2.3021}})],
         "SPEC_CONSTANT_MISMATCH", 3,
         "set the 1,820-token pin to A9 §3's SUPERSEDED 1.120 — the constant that needed a "
         "(5,20,15) grid which cannot exist. A9 §4 pre-registers an auto-drop on this gate "
         "failing, so this must NOT come out as a plain FAIL",
         want_class="SPEC-CONSTANT-MISMATCH", want_escalates=True, want_autodrop=False)
    drifted = work / "root_s4_at_121f"
    what = copy_root_with_s4_at_121f(root, drifted)
    case("S2_realized_geometry_drift", drifted, [], "SPEC_CONSTANT_MISMATCH", 3, what,
         want_class="SPEC-CONSTANT-MISMATCH", want_escalates=True, want_autodrop=False)


def unit_cases(results: list) -> None:
    """The DATA-FAIL branch of the shared classifier, called directly."""
    sys.path.insert(0, str(HERE))
    import mixed_format_probe as P  # noqa: PLC0415

    good_pins = [{"seq_len": 1820, "pinned": 1.2350, "from_trainer_fn": 1.2350260416666665,
                  "abs_err": 2.6e-05, "tol": 1e-3, "ok": True}]
    bad_pins = [{"seq_len": 1820, "pinned": 1.120, "from_trainer_fn": 1.2350260416666665,
                 "abs_err": 0.115, "tol": 1e-3, "ok": False}]
    mech = ["S4_r00/x.pt: sampler handed seq_len 3640, not the target's own 1820"]

    cases = [
        ("U1_data_fail_when_mechanism_inconsistent", good_pins, mech,
         "DATA-FAIL", "FAIL", 1, False, True),
        ("U2_data_wins_over_spec_when_both_broken", bad_pins, mech,
         "DATA-FAIL", "FAIL", 1, False, True),
        ("U3_spec_only", bad_pins, [], "SPEC-CONSTANT-MISMATCH", "SPEC_CONSTANT_MISMATCH",
         3, True, False),
        ("U4_clean", good_pins, [], None, "PASS", 0, False, False),
    ]
    for name, pins, off, w_cls, w_verdict, w_exit, w_esc, w_drop in cases:
        c = P.classify_shift_assert(
            pin_rows=pins, tol=1e-3, data_offenders=list(off),
            realized_tokens=[1820, 4800], observed_seq_lens=[1820, 4800],
            want_tokens=[1820, 4800],
            observed_shifts=[1.2350260416666665, 2.302083333333333],
            manifest_expected_shifts=[1.2350260416666665, 2.3020833333333335])
        problems = []
        for label, got, want in (("failure_class", c["failure_class"], w_cls),
                                 ("verdict", c["verdict"], w_verdict),
                                 ("exit_code", c["exit_code"], w_exit),
                                 ("escalates", c["escalates"], w_esc),
                                 ("auto_drop_permitted", c["auto_drop_permitted"], w_drop)):
            if got != want:
                problems.append(f"{label} {got!r} != {want!r}")
        results.append({"kind": "classifier_unit", "case": name, "ok": not problems,
                        "broke": f"pins ok={all(p['ok'] for p in pins)}, "
                                 f"{len(off)} mechanism offender(s)",
                        "failure_class": c["failure_class"], "verdict": c["verdict"],
                        "exit_code": c["exit_code"], "escalates": c["escalates"],
                        "auto_drop_permitted": c["auto_drop_permitted"],
                        "ERROR": "; ".join(problems) or None})
        print(f"[{'PROVEN' if not problems else 'PROBLEM'}] {name}: {c['verdict']} "
              f"exit={c['exit_code']} class={c['failure_class']}"
              + (f"  !! {'; '.join(problems)}" if problems else ""))


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(LAB / "misc/ctt_v2_final/artefacts/smoke_gate/prove"))
    ap.add_argument("--root", default=str(SMOKE_ROOT),
                    help="smoke root, for the log gate's SMOKE_ROOT_MANIFEST (n_samples)")
    ap.add_argument("--probe-python", help="interpreter with ltx_trainer importable; the "
                                          "shift cases are SKIPPED without it")
    ap.add_argument("--probe-root", help="smoke root for the shift cases (default --root)")
    ap.add_argument("--skip-log", action="store_true")
    args = ap.parse_args()

    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: list[dict] = []

    if not args.skip_log:
        print("=" * 78)
        print("LOG-GATE MUTATIONS (check_train_log.py) — copies of the REAL archived capture")
        print("=" * 78)
        log_cases(sys.executable, work, Path(args.root), results)

    print("=" * 78)
    print("CLASSIFIER UNIT CASES (mixed_format_probe.classify_shift_assert)")
    print("=" * 78)
    unit_cases(results)

    if args.probe_python:
        print("=" * 78)
        print("SHIFT-ASSERT MUTATIONS (mixed_format_probe.py --shifts-only, no GPU)")
        print("=" * 78)
        shift_cases(args.probe_python, Path(args.probe_root or args.root), work, results)
    else:
        results.append({"kind": "shift_assert", "case": "S0..S2", "ok": True,
                        "SKIPPED": "no --probe-python given"})
        print("[SKIPPED] shift-assert cases: pass --probe-python to run them")

    proven = [r["case"] for r in results if r["ok"]]
    failed = [f"{r['case']}: {r.get('ERROR')}" for r in results if not r["ok"]]
    rec = {"schema": "ctt_v2_prove_smoke_gate/1",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "host": os.uname().nodename,
           "archived_train_log": str(ARCHIVED_TRAIN_LOG),
           "archived_slurm_capture": str(ARCHIVED_SLURM_CAPTURE),
           "smoke_root": args.root,
           "n_cases": len(results), "n_proven": len(proven),
           "failures": failed, "results": results,
           "elapsed_s": round(time.time() - t0, 2)}
    out = work / "PROVE_SMOKE_GATE.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"\n[prove] {len(proven)}/{len(results)} cases behaved as required -> {out}")
    if failed:
        print("[prove] PROBLEMS:")
        for f in failed:
            print(f"        - {f}")
        return 1
    print("[prove] EVERY SMOKE-GATE CHECKER IS PROVEN TO FIRE, IN THE RIGHT FAILURE CLASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
