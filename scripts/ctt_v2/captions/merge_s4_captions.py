#!/usr/bin/env python3
"""Merge the fanned-out S4 first-frame batches into one store, and measure it.

S4 is one-sided (owner decision 2026-07-28: condition on video frame 0 alone, i.e. latent
frame 0, i.e. `mask[:1]`), so every clip carries exactly ONE role-A description and no role-B.

The checks here are the ones that can actually reject the batch, and nothing else:

  coverage    every roster stem present exactly once
  collision   no key already owned by the locked S0/S1/S2 store (that store is LOCKED; a
              collision would silently reassign one of its descriptions)
  format      the pipeline's own `format_violations(text, "A")` -- same function the locked
              store was held to, so the two stores are held to one standard
  leak        the pipeline's own Tier-1 filter, which is the ONLY place the refVFX trigger
              lexicon lives.  A caption naming its own effect hands the model the transition.
  register    word/comma/colour distribution against the 750 role-A descriptions already in
              the locked store -- gate #8 measures exactly this kind of style distance, so a
              drift here is a measurable defect, not a taste question.

Usage:
    python scripts/ctt_v2/captions/merge_s4_captions.py            # report only
    python scripts/ctt_v2/captions/merge_s4_captions.py --write    # write the store
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2/captions"))
import caption_common as cc  # noqa: E402

ROSTER = REPO / "outputs/ctt_v2/encodes/S4/ROSTER.json"
SELECTION = REPO / "data/processed/s4_refvfx/selection.json"
BATCHES = REPO / "outputs/ctt_v2/captions/s4_batches"
LOCKED_STORE = REPO / "outputs/ctt_v2/captions/CAPTION_STORE.json"
OUT_STORE = REPO / "outputs/ctt_v2/captions/S4_CAPTION_STORE.json"
SPEC = REPO / "outputs/ctt_v2/captions/S4_CAPTION_SPEC.md"

#: A4 Q1 storage convention: strip the trailing period, keep everything else.
def store_form(text: str) -> str:
    return re.sub(r"\.\s*$", "", text.strip())


COLOUR = re.compile(
    r"\b(red|orange|yellow|green|blue|purple|violet|pink|brown|black|white|grey|gray|"
    r"beige|tan|cream|gold|golden|silver|bronze|teal|turquoise|navy|maroon|crimson|"
    r"amber|olive|lavender|magenta|cyan|khaki|charcoal|ivory)\b", re.I)


def describe(vals: list[str], label: str) -> dict:
    w = [len(v.split()) for v in vals]
    c = [v.count(",") for v in vals]
    col = [len(COLOUR.findall(v)) for v in vals]
    q = statistics.quantiles(w, n=10)
    return {
        "label": label, "n": len(vals),
        "words_p10": round(q[0], 1), "words_p50": statistics.median(w),
        "words_p90": round(q[-1], 1), "words_min": min(w), "words_max": max(w),
        "words_mean": round(statistics.mean(w), 2),
        "commas_mean": round(statistics.mean(c), 3),
        "colour_terms_mean": round(statistics.mean(col), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--from-gemini", action="store_true",
                    help="source descriptions from outputs/ctt_v2/captions/s4_gemini/ "
                         "(gemini-3.6-flash, prompt v2, per-item length draw) instead of the "
                         "Sonnet fan-out batches")
    args = ap.parse_args()

    stems = json.loads(ROSTER.read_text())["stems"]
    effect = {s["k"]: s["effect"] for s in json.loads(SELECTION.read_text())["samples"]}

    raw: dict[str, str] = {}
    origin: dict[str, str] = {}
    dupes: list[str] = []
    if args.from_gemini:
        src = REPO / "outputs/ctt_v2/captions/s4_gemini/descriptions.json"
        raw = json.loads(src.read_text())
        origin = {k: src.name for k in raw}
        files = [src]
    else:
        files = sorted(BATCHES.glob("out_*.json"))
        for f in files:
            for k, v in json.loads(f.read_text()).items():
                if k in raw:
                    dupes.append(f"{k} ({origin[k]} vs {f.name})")
                raw[k] = v
                origin[k] = f.name

    desc = {k: store_form(v) for k, v in raw.items()}

    missing = [s for s in stems if s not in desc]
    extra = sorted(set(desc) - set(stems))

    locked = json.loads(LOCKED_STORE.read_text())["descriptions"]
    collisions = sorted(k for k in desc if f"{k}|A" in locked or f"{k}|B" in locked)

    filt = cc.LeakFilter()
    fmt_bad, t1_bad, t2_flag = {}, {}, {}
    for k, v in desc.items():
        if (bad := cc.format_violations(v, "A")):
            fmt_bad[k] = bad
        if (hits := filt.tier1(v)):
            t1_bad[k] = hits
        if (flags := filt.tier2(v)):
            t2_flag[k] = flags

    corpus_a = [v for kk, v in locked.items() if kk.endswith("|A")]
    report = {
        "schema": "ctt_v2_s4_caption_merge/v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "batch_files": [f.name for f in files],
        "coverage": {
            "roster": len(stems), "merged": len(desc),
            "missing": missing[:40], "n_missing": len(missing),
            "extra": extra[:40], "n_extra": len(extra),
            "duplicate_keys": dupes[:20], "n_duplicate_keys": len(dupes),
            "complete": not missing and not extra and not dupes,
        },
        "collision_with_locked_store": {"n": len(collisions), "keys": collisions[:20]},
        "format_violations": {"n": len(fmt_bad), "detail": dict(list(fmt_bad.items())[:20])},
        "tier1_leaks": {"n": len(t1_bad), "detail": dict(list(t1_bad.items())[:20])},
        "tier2_flags": {
            "n": len(t2_flag),
            "pct": round(100 * len(t2_flag) / max(1, len(desc)), 2),
            "detail": dict(list(t2_flag.items())[:20]),
        },
        "register": {
            "s4_new": describe(list(desc.values()), "S4 role-A (this merge)"),
            "corpus_role_a": describe(corpus_a, "locked store role-A"),
        },
        "per_effect_word_p50": {
            e: statistics.median([len(desc[s].split()) for s in stems
                                  if s in desc and effect[s] == e])
            for e in sorted(set(effect.values()))
        },
    }

    hard = []
    if not report["coverage"]["complete"]:
        hard.append("coverage")
    if collisions:
        hard.append("collision_with_locked_store")
    if fmt_bad:
        hard.append("format_violations")
    if t1_bad:
        hard.append("tier1_leaks")
    report["hard_fail"] = hard

    print(json.dumps({k: v for k, v in report.items() if k != "per_effect_word_p50"}, indent=2))
    pe = report["per_effect_word_p50"].values()
    print(f"\n[per-effect word p50] min {min(pe)} max {max(pe)} "
          f"(a wide spread here means style tracks effect -- the confound)")
    print(f"[hard_fail] {hard or 'NONE'}")

    if args.write:
        if hard:
            raise SystemExit(f"refusing to write: hard_fail {hard}")
        payload = {
            "schema": "ctt_v2_s4_caption_store/v1",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "stratum": "S4",
            "keying": "'clip_id|A'.  S4 is ONE-SIDED: video frame 0 alone is conditioned "
                      "(latent frame 0, mask[:1]), so there is no role-B description and the "
                      "A-role description covers exactly the conditioned pixels.",
            "sided_authority": "owner decision 2026-07-28 (frame 0, not frames 0-8); "
                               "S4_spec.json sided='one'",
            "generator": ("gemini-3.6-flash, prompt v2 role-A, per-item length draw over the "
                          "171 corpus word counts (the SAME instrument and rule as the locked "
                          "store) -- adopted after a controlled comparison showed captioner "
                          "identity is NOT what gate 8a measures: Sonnet 0.8849 vs Gemini "
                          "0.8913, a 0.006 difference at SE 0.006. See CAPTIONS.md 12.4."
                          if args.from_gemini else
                          "claude-sonnet vision fan-out, 25 batches x 80 clips, "
                          "effect-stratified so captioner style cannot track effect"),
            "prompt_variant": "v2-s4f0",
            "prompt_variant_delta": "prompt v2 role-A verbatim EXCEPT '9-frame snippet' -> "
                                    "'single still frame', because only frame 0 is conditioned. "
                                    "Spec text: outputs/ctt_v2/captions/S4_CAPTION_SPEC.md",
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "source_captions": "NONE editable -- refVFX ships one trigger phrase per EFFECT "
                               "(42 phrases over 2,000 clips), not a per-clip description. The "
                               "trigger is a Tier-1 leak string, so it could not be adapted; "
                               "these descriptions are generated from pixels.",
            "counts": {"clips": len(desc), "required": len(stems),
                       "coverage": f"{len(desc)}/{len(stems)}"},
            "content_hash": "sha256:" + hashlib.sha256(
                json.dumps({f"{k}|A": desc[k] for k in sorted(desc)},
                           sort_keys=True).encode()).hexdigest(),
            #: NOT the locked store's recipe (that one uses compact separators and
            #: ensure_ascii=False).  Recorded rather than silently differing, because a hash
            #: whose recipe is not written down cannot be checked by anyone later.
            "content_hash_covers": "the `descriptions` map only, keyed 'clip|A': "
                                   "json.dumps(sorted-key dict, sort_keys=True) with "
                                   "DEFAULT separators and ensure_ascii=True, sha256 of the "
                                   "utf-8 bytes -- stable against provenance churn",
            "merge_report": {k: v for k, v in report.items() if k != "per_effect_word_p50"},
            "per_effect_word_p50": report["per_effect_word_p50"],
            "tier2_queue": sorted(t2_flag),
            "descriptions": {f"{k}|A": desc[k] for k in sorted(desc)},
            "effect_of_clip_NOT_FOR_CAPTIONING": {k: effect[k] for k in sorted(desc)},
        }
        OUT_STORE.write_text(json.dumps(payload, indent=1))
        print(f"\n[ok] wrote {OUT_STORE.relative_to(REPO)}  "
              f"{len(desc)} descriptions  content_hash {payload['content_hash'][7:23]}")


if __name__ == "__main__":
    main()
