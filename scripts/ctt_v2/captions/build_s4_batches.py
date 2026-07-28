#!/usr/bin/env python3
"""Split the 2,000 S4 first frames into effect-stratified batches for fan-out captioning.

S4 is one-sided (frame-0 conditioning only, owner decision 2026-07-28), so every clip needs
exactly ONE role-A description and no role-B.  The source dataset ships no per-clip caption --
only a per-effect trigger phrase shared by ~48 clips -- so the descriptions are generated, not
edited.

Batches are round-robin over an effect-sorted roster so each batch carries ~2 clips of each of
the 42 effects.  That is the point of the stratification: a captioner's own style drift then
cannot correlate with an effect, which is the confound that would let the transition be read
off the caption's *register* rather than its content.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SELECTION = REPO / "data/processed/s4_refvfx/selection.json"
ROSTER = REPO / "outputs/ctt_v2/encodes/S4/ROSTER.json"
FRAMES = REPO / "outputs/ctt_v2/captions/s4_frame0"
OUTDIR = REPO / "outputs/ctt_v2/captions/s4_batches"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=25)
    args = ap.parse_args()

    stems = json.loads(ROSTER.read_text())["stems"]
    effect = {s["k"]: s["effect"] for s in json.loads(SELECTION.read_text())["samples"]}

    rows = []
    for st in stems:
        jpg = FRAMES / f"{st}.jpg"
        if not jpg.exists():
            raise SystemExit(f"missing frame: {jpg}")
        rows.append({"stem": st, "effect": effect[st], "frame": str(jpg.relative_to(REPO))})

    # round-robin over (effect, stem) so every batch sees the whole effect spread
    rows.sort(key=lambda r: (r["effect"], r["stem"]))
    batches: list[list[dict]] = [[] for _ in range(args.n_batches)]
    for i, r in enumerate(rows):
        batches[i % args.n_batches].append(r)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(batches, start=1):
        # the captioner never sees `effect` -- it would be exactly the leak we forbid
        payload = [{"stem": r["stem"], "frame": r["frame"]} for r in b]
        (OUTDIR / f"batch_{i:02d}.json").write_text(json.dumps(payload, indent=1))

    manifest = {
        "n_clips": len(rows),
        "n_batches": args.n_batches,
        "sizes": [len(b) for b in batches],
        "effects": sorted({r["effect"] for r in rows}),
        "role": "A only (S4 is one-sided: frame-0 conditioning)",
        "effect_withheld_from_captioner": True,
        "per_batch_effect_spread": {
            f"batch_{i:02d}": len({r["effect"] for r in b})
            for i, b in enumerate(batches, start=1)
        },
    }
    (OUTDIR / "BATCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] {len(rows)} clips -> {args.n_batches} batches, sizes {manifest['sizes'][:5]}...")
    print(f"[ok] effect spread per batch: "
          f"{min(manifest['per_batch_effect_spread'].values())}-"
          f"{max(manifest['per_batch_effect_spread'].values())} of {len(manifest['effects'])}")


if __name__ == "__main__":
    main()
