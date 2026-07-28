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
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

RE_INDEX = re.compile(r"Fast index:\s*(\d+)\s*valid samples from\s*(\d+)\s*total"
                      r"(?:\s*\((\d+)\s*skipped\))?")
RE_STEP = re.compile(r"Step\s+(\d+)/(\d+)\s*-\s*Loss:\s*([0-9.eE+\-]+|nan|inf|-inf)", re.I)
#: only NaN/Inf that is actually attached to a loss or gradient — a bare `\binf\b` sweep over
#: the whole log matches ordinary words and file paths and produces false FAILs.
RE_NAN = re.compile(r"Loss:\s*-?(?:nan|inf)\b|loss is (?:nan|inf)\b|"
                    r"(?:NaN|Inf) (?:detected|encountered)", re.I)
RE_TRACE = re.compile(r"Traceback \(most recent call last\)")

#: 🔴 The trainer logs through `RichHandler`, so every number arrives wrapped in SGR colour
#: codes and OSC-8 hyperlinks — literally `Step \x1b[1;36m20\x1b[0m/\x1b[1;36m30\x1b[0m`.
#: Regexing the raw capture silently matches NOTHING and the gate reports a spurious FAIL on
#: a healthy run. Strip both escape families before parsing.
RE_SGR = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
RE_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(s: str) -> str:
    return RE_SGR.sub("", RE_OSC.sub("", s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log = Path(a.log)
    raw = log.read_text(errors="replace") if log.exists() else ""
    text = strip_ansi(raw)
    man = json.loads((Path(a.root) / "SMOKE_ROOT_MANIFEST.json").read_text())
    expected = man["n_samples"]

    idx = RE_INDEX.findall(text)
    steps = [(int(s), int(t), f) for s, t, f in RE_STEP.findall(text)]
    losses = []
    for _, _, f in steps:
        try:
            losses.append(float(f))
        except ValueError:
            losses.append(float("nan"))

    checks = {}
    checks["T1_fast_index_present"] = {
        "ok": bool(idx),
        "detail": f"{len(idx)} 'Fast index' line(s) found" if idx else
                  "the trainer never printed a 'Fast index' line — cannot prove no sample "
                  "was silently dropped",
        "lines": [{"valid": int(v), "total": int(t), "skipped": int(s or 0)} for v, t, s in idx],
    }
    ok_n = bool(idx) and all(int(v) == expected and int(t) == expected and int(s or 0) == 0
                             for v, t, s in idx)
    checks["T2_fast_index_N_of_N"] = {
        "ok": ok_n, "expected_n": expected,
        "detail": f"every index line reads {expected} of {expected}, 0 skipped" if ok_n
                  else f"index line(s) do NOT read {expected} of {expected}: "
                       f"{[(v, t, s) for v, t, s in idx]}",
    }
    reached = max((s for s, _, _ in steps), default=0)
    checks["T3_steps_completed"] = {
        "ok": reached >= a.steps - (a.steps % 20 or 0) and reached > 0,
        "detail": f"highest logged step {reached} of {a.steps} "
                  f"(the trainer only logs every 20th step with progress bars off)",
        "steps_logged": len(steps), "highest_step": reached, "target": a.steps,
    }
    finite = all(math.isfinite(x) for x in losses) and bool(losses)
    checks["T4_losses_finite"] = {
        "ok": finite, "n_losses": len(losses), "losses": losses,
        "detail": "every logged loss is finite" if finite else "a logged loss is NaN/Inf",
    }
    checks["T5_no_traceback"] = {
        "ok": not RE_TRACE.search(text),
        "detail": "no traceback in the trainer's output" if not RE_TRACE.search(text)
                  else "the trainer raised",
    }
    nan_hits = RE_NAN.findall(text)
    checks["T6_no_nan_attached_to_loss"] = {
        "ok": not nan_hits,
        "detail": "no NaN/Inf attached to a loss or gradient anywhere in the log"
                  if not nan_hits else f"NaN/Inf mentions: {nan_hits[:5]}",
    }
    checks["T0_ansi_stripped"] = {
        "ok": len(text) <= len(raw),
        "detail": f"stripped {len(raw) - len(text)} bytes of Rich SGR/OSC-8 escapes before "
                  f"parsing ({len(raw)} raw -> {len(text)} clean); without this every regex "
                  f"below silently matches nothing and the gate FAILs a healthy run",
    }

    verdict = "PASS" if all(v["ok"] for v in checks.values()) else "FAIL"
    rec = {"schema": "ctt_v2_smoke_train_gate/1",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "log": str(log), "root": str(a.root), "expected_samples": expected,
           "checks": checks, "VERDICT": verdict}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")

    for k, v in checks.items():
        print(f"[{'PASS' if v['ok'] else 'FAIL'}] {k}: {v['detail']}")
    print(f"VERDICT: {verdict} -> {a.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
