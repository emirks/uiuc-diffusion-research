#!/usr/bin/env python
"""Aggregate Gemini token spend from the archived raw responses.

WHY THIS EXISTS
---------------
The M3 pilot consumed the project's entire Gemini prepayment balance (895 TRY) and produced
**zero** production descriptions -- it was a calibration run, by design. When the owner asked
"which proportion of captioning is done and what did it cost", the answer had to be
reconstructed by hand from agent reports, because nothing aggregated the spend.

The raw responses *do* carry `usageMetadata` (`raw_generation_responses.jsonl`,
`raw_audit_responses.jsonl`, and the S4 verification archives) -- it is only the processed
`records.json` that drops it. So the data was always there; what was missing was a reader.
This is that reader. Run it after every batch so the next top-up is *measured*, never estimated.

Under the round-9 config-archiving rule an unarchived measurement cannot gate a decision.
Spend is not a gate, but the same logic applies: a number nobody can reproduce is a number
nobody should act on.

USAGE
-----
  PY=$LAB/envs/diffusion/bin/python
  # everything archived so far, with a per-file breakdown
  $PY scripts/ctt_v2/captions/cost_ledger.py --verbose

  # forecast what the remaining production store will cost, calibrated on observed rates
  $PY scripts/ctt_v2/captions/cost_ledger.py --forecast 4532 --spent-try 895

`--spent-try` anchors TRY-per-token on a known invoice total. The resulting rate is an UPPER
bound: some early calls (the throughput benchmarks) were never archived, so the true token
count behind a given invoice is higher than the archived count, which makes TRY/token look
larger than it is. Forecasts are therefore conservative.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
#: $LAB is the parent of the *main* repo, not of the worktree. In a worktree, REPO_ROOT is
#: <LAB>/diffusion-research/.claude/worktrees/<name>, so LAB is 4 levels up, not 2 -- getting
#: this wrong silently drops every EXTERNAL_GLOBS archive, which undercounted the first run of
#: this script by 2,761 calls (1,712 reported against 4,473 actual) and inflated TRY/token 4.5x.
#: Anchor on the marker directory instead of counting, so the depth can't drift again.
def _find_lab(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "misc" / "ctt_v2_final").is_dir() and (p / "secrets").is_dir():
            return p
    return start.parent


LAB = _find_lab(REPO_ROOT)

#: Every archive that carries `usageMetadata`. Add new lanes here as they appear -- a lane
#: missing from this list is silently uncounted, which is the failure mode this file exists
#: to prevent, so `--strict` fails when a `raw_*.jsonl` on disk matches none of these.
ARCHIVE_GLOBS = (
    "scripts/ctt_v2/captions/pilot_m3/*/raw_*.jsonl",
    "scripts/ctt_v2/captions/pilot_m3/raw_*.jsonl",
    "scripts/ctt_v2/captions/store/raw_*.jsonl",          # the production store, when it runs
    "scripts/ctt_v2/captions/store/*/raw_*.jsonl",
    "scripts/ctt_v2/s1/gate/raw_*.jsonl",                 # S1 blind class-ID gate
    "scripts/ctt_v2/s4/gate/raw_*.jsonl",                 # S4 blind-guess gate
)
#: archives that live outside the repo (campaign verification lanes)
EXTERNAL_GLOBS = ("misc/ctt_v2_final/_verify_*/raw/*.jsonl",
                  "misc/ctt_v2_final/_verify_*/*.jsonl")


def _usage(rec: dict) -> dict:
    """usageMetadata, whether the line wraps the response or *is* the response."""
    return (rec.get("raw_response") or rec).get("usageMetadata") or {}


def scan(paths: list[Path]) -> tuple[dict, dict]:
    per_file: dict[str, dict] = {}
    per_model: dict[str, dict] = defaultdict(lambda: dict(calls=0, prompt=0, out=0, think=0, total=0))
    for p in paths:
        f = dict(calls=0, prompt=0, out=0, think=0, total=0, no_usage=0)
        for line in p.open(errors="replace"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            u = _usage(rec)
            if not u:
                f["no_usage"] += 1
                continue
            raw = rec.get("raw_response") or rec
            model = rec.get("model") or raw.get("modelVersion") or "?"
            for tgt in (f, per_model[model]):
                tgt["calls"] += 1
                tgt["prompt"] += u.get("promptTokenCount", 0)
                tgt["out"] += u.get("candidatesTokenCount", 0)
                tgt["think"] += u.get("thoughtsTokenCount", 0)
                tgt["total"] += u.get("totalTokenCount", 0)
        if f["calls"] or f["no_usage"]:
            per_file[str(p)] = f
    return per_file, dict(per_model)


def _looks_like_api_archive(p: Path, probe_lines: int = 5) -> bool:
    """Does this .jsonl actually hold Gemini responses? Content test, not a filename test."""
    try:
        with p.open(errors="replace") as fh:
            for _ in range(probe_lines):
                line = fh.readline()
                if not line:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if _usage(rec) or "raw_response" in rec or "candidates" in rec:
                    return True
    except OSError:
        return False
    return False


def collect(strict: bool = False) -> list[Path]:
    found: set[Path] = set()
    for g in ARCHIVE_GLOBS:
        found |= {Path(x) for x in glob.glob(str(REPO_ROOT / g), recursive=True)}
    for g in EXTERNAL_GLOBS:
        found |= {Path(x) for x in glob.glob(str(LAB / g), recursive=True)}
    if strict:
        # (a) any raw_*.jsonl in either tree that we did NOT match -- would be silently uncounted
        #: bounded on purpose. A `**` recursive glob over misc/ctt_v2_final took >90 s (it walks
        #: the verification lanes' media), which made --strict unusable and therefore useless.
        #: Depth-limited patterns cover every layout the campaign actually uses and run instantly.
        everywhere: set[Path] = set()
        for pat in ("scripts/ctt_v2/*/raw_*.jsonl", "scripts/ctt_v2/*/*/raw_*.jsonl",
                    "scripts/ctt_v2/*/*/*/raw_*.jsonl"):
            everywhere |= {Path(x) for x in glob.glob(str(REPO_ROOT / pat))}
        for pat in ("misc/ctt_v2_final/*/*.jsonl", "misc/ctt_v2_final/*/*/*.jsonl"):
            everywhere |= {Path(x) for x in glob.glob(str(LAB / pat))}
        #: Decide by CONTENT, not by filename. A name-based rule flagged the root-machinery
        #: harness's SAMPLES.jsonl -- sample manifests, not API archives -- and a --strict that
        #: cries wolf gets disabled, which defeats the point. A file is an API archive iff one
        #: of its first lines actually carries usageMetadata / raw_response.
        missed = {p for p in (everywhere - found) if _looks_like_api_archive(p)}
        if missed:
            raise SystemExit("[ledger] STRICT: archives on disk matched by no glob "
                             "(they would be silently uncounted):\n  "
                             + "\n  ".join(sorted(str(m) for m in missed)))
        # (b) AT LEAST ONE external pattern must resolve. This exists to prove LAB is right --
        #     a mis-derived LAB makes every external glob silently empty, the exact bug that
        #     undercounted the first run, and one that (a) cannot see because it globs from the
        #     same wrong root. Requiring *every* pattern to match would be wrong: some cover
        #     layouts that legitimately do not exist yet, and a seatbelt that fails on a healthy
        #     tree gets switched off.
        if not any(glob.glob(str(LAB / g), recursive=True) for g in EXTERNAL_GLOBS):
            raise SystemExit(
                f"[ledger] STRICT: no external glob resolved to anything under\n"
                f"    {LAB}\n"
                f"  patterns tried: {list(EXTERNAL_GLOBS)}\n"
                f"  LAB was derived as {LAB} -- if that looks wrong, the marker-directory\n"
                f"  lookup in _find_lab() failed and external archives are being dropped.")
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="per-file breakdown")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any raw_*.jsonl on disk is matched by no glob")
    ap.add_argument("--forecast", type=int, metavar="N_DESCRIPTIONS",
                    help="forecast cost for N further descriptions (gen + audit each)")
    ap.add_argument("--spent-try", type=float, metavar="TRY",
                    help="known invoice total, to anchor TRY per token")
    ap.add_argument("--out", type=Path, help="write the ledger as JSON")
    args = ap.parse_args()

    paths = collect(strict=args.strict)
    per_file, per_model = scan(paths)
    if not per_model:
        raise SystemExit("[ledger] no archived usageMetadata found — nothing to report")

    if args.verbose:
        print("=== per-archive ===")
        for k in sorted(per_file):
            f = per_file[k]
            warn = f"  ({f['no_usage']} lines WITHOUT usage)" if f["no_usage"] else ""
            print(f"  {os.path.relpath(k, str(LAB)):<74s} {f['calls']:>6,d} calls "
                  f"{f['total']:>10,d} tok{warn}")
        print()

    print("=== spend by model ===")
    print(f"  {'model':24s} {'calls':>7s} {'prompt':>12s} {'out':>9s} {'think':>9s} {'total':>12s}")
    T = defaultdict(int)
    for m, d in sorted(per_model.items(), key=lambda x: -x[1]["calls"]):
        print(f"  {m:24s} {d['calls']:>7,d} {d['prompt']:>12,d} {d['out']:>9,d} "
              f"{d['think']:>9,d} {d['total']:>12,d}")
        for k in d:
            T[k] += d[k]
    print(f"  {'':24s} {'-'*7} {'-'*12} {'-'*9} {'-'*9} {'-'*12}")
    print(f"  {'TOTAL':24s} {T['calls']:>7,d} {T['prompt']:>12,d} {T['out']:>9,d} "
          f"{T['think']:>9,d} {T['total']:>12,d}")
    print(f"  {'per call':24s} {'':>7s} {T['prompt']/T['calls']:>12,.0f} "
          f"{T['out']/T['calls']:>9,.0f} {T['think']/T['calls']:>9,.0f} "
          f"{T['total']/T['calls']:>12,.0f}")

    #: the production-shaped rate: a finished description costs one generation call plus one
    #: 100%-coverage Layer-2 audit call, both on a 9-frame video payload. Measured from the
    #: round-2 pilot, which used exactly that shape -- NOT from the pooled average, which is
    #: inflated by the S4 verification lane's single-image payloads (~1,197 prompt tokens
    #: against ~434 for a video call).
    r2 = {k: v for k, v in per_file.items() if "round2" in k}
    per_desc = None
    if len(r2) >= 2:
        calls = sum(v["calls"] for v in r2.values())
        toks = sum(v["total"] for v in r2.values())
        per_desc = toks / (calls / 2)
        print(f"\n=== production-shaped rate (round-2 pilot: 9-frame video payloads) ===")
        print(f"  {calls:,d} calls, {toks:,d} tokens  =>  {per_desc:,.0f} tokens per finished "
              f"description (generation + audit)")

    rate = None
    if args.spent_try:
        rate = args.spent_try / T["total"]
        print(f"\n=== TRY anchoring ===")
        print(f"  invoice {args.spent_try:,.0f} TRY over {T['total']:,d} archived tokens")
        print(f"  => {rate*1e6:,.2f} TRY per million tokens  (UPPER bound: unarchived calls")
        print(f"     also contributed to the invoice, so true TRY/token is lower)")

    if args.forecast:
        if per_desc is None:
            raise SystemExit("[ledger] --forecast needs the round-2 archives to calibrate")
        gen = args.forecast * per_desc
        gates = 150 * 3 * 700 + 700 * 400          # blind-guess arms + per-stratum batteries
        est = gen + gates
        print(f"\n=== forecast: {args.forecast:,d} further descriptions ===")
        print(f"  descriptions (gen + audit) : {gen:>12,.0f} tok")
        print(f"  gates (blind-guess + battery): {gates:>10,.0f} tok")
        print(f"  {'TOTAL':27s}: {est:>12,.0f} tok   ({est/T['total']:.2f}x the archived spend)")
        if rate:
            print(f"  {'estimated cost':27s}: {est*rate:>12,.0f} TRY  (conservative — see above)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"per_model": per_model, "per_file": per_file, "totals": dict(T),
             "tokens_per_description": per_desc,
             "try_per_token_upper_bound": rate}, indent=1))
        print(f"\n[ledger] -> {args.out}")


if __name__ == "__main__":
    main()
