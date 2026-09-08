#!/usr/bin/env python3
"""build_r1.py — prompts.jsonl -> reg/r1.jsonl (40 R1 rows).

R1 = start anchor only (sided one, prefix 9f) + full prompt (start caption + change clause +
end caption), rendered VERBATIM by the generator. No reference, no twin.

    python build_r1.py [PROMPTS_JSONL] [--out reg/r1.jsonl]

Defaults to the real prompts file (misc/2026-09-08_collapse_probe/prompts/prompts.jsonl); falls
back to prompts/_example.jsonl only if the real one is absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _probe_common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="?", default=None, help="prompts.jsonl (default: real path)")
    ap.add_argument("--out", default=str(C.REG / "r1.jsonl"))
    args = ap.parse_args()

    ppath = Path(args.prompts) if args.prompts else C.default_prompts()
    prompts = C.load_prompts(ppath)
    rows = [C.build_r1_row(p) for p in prompts]

    # sanity: every start window exists (generation would FileNotFoundError otherwise)
    missing = [r["endpoint"] for r in rows if not (C.CONDS / f"{r['endpoint']}_start9.mp4").exists()]
    if missing:
        raise SystemExit(f"[build_r1] missing start windows for: {missing}")

    C.write_jsonl(Path(args.out), rows)
    n_hi = sum(1 for r in rows if r["tier"] == "high")
    n_lo = sum(1 for r in rows if r["tier"] == "low")
    print(f"[build_r1] {ppath.name}: wrote {len(rows)} R1 rows -> {args.out} "
          f"({n_hi} high, {n_lo} low)")
    print(f"[build_r1] item_id example: {rows[0]['item_id']}")
    print(f"[build_r1] out mp4 example: "
          f"{C.out_mp4('r1', rows[0]['item_id'], C.SEEDS[0]).relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
