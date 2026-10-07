"""S6 (EffectData) Lane-A caption store — validate the fan-out outputs and assemble shelf 004.

Captions are PER-SUBJECT (the first frame is identical across a subject's clips on Axis A;
2000 subjects, keyed '<subject>|A'), unlike S4 which is per-clip. Everything else mirrors the
S4 store: a `descriptions` map + metadata, with `content_hash` = sha256 over the descriptions
map serialised as json.dumps(sorted-key, sort_keys=True, default separators, ensure_ascii=True).

    validate  — merge pilot + out_*.json, check coverage/dupes/format/leak, report gaps. No writes.
    assemble  — validate, then write store/captions/004_effectdata/{...}. Refuses if not clean.

Authority: store/TEXT_LIFECYCLE.md §2 (Lane A).  Spec the agents followed:
data/processed/effectdata/captions/CAPTION_TASK.md (variant v2-s4f0).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAPDIR = REPO / "data/processed/effectdata/captions"
SELECTION = REPO / "data/processed/effectdata/selection_top2000.json"
ROSTER = REPO / "outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json"
SHELF = REPO / "store/captions/004_effectdata"
SPEC_SRC = CAPDIR / "CAPTION_TASK.md"
REGEN_JSON = CAPDIR / "regen/gemini_captions.json"   # per-image gemini-3.6-flash regen (2026-08-28)

#: which caption source to build from. 'regen' = the alignment-safe per-image gemini regen that
#: replaced the batch-hallucinated opus captions; 'batches' = the original opus fan-out.
SOURCE = "regen"
GENERATOR = {
    "regen": "gemini-3.6-flash, prompt v2-s4f0 ('single still frame'), ONE IMAGE PER CALL "
             "(structural alignment fix after the opus 80-image batches misaligned "
             "caption<->subject; see misc/2026-08-28_effectdata_s6/BUILD.md). Per-item length "
             "draw over the 171-value corpus. Same instrument/rule as the S4 store.",
    "batches": "claude-opus-4-8 vision, prompt v2-s4f0, per-item length draw — SUPERSEDED "
               "(batch caption<->subject misalignment; do not use).",
}


def _strip_terminal_period(s: str) -> str:
    """S4 convention: descriptions carry NO trailing period; build_encode_inputs adds `. sksz.`
    A stored period yields the double-period `...front.. sksz.` bug this fixes."""
    s = s.strip()
    return s[:-1].rstrip() if s.endswith(".") else s

# HARD leaks: a caption naming/foreshadowing the transition is worse than none. These are
# process/outcome/effect/frame/sound words. State words ("a low sun", "glowing embers") are NOT
# here on purpose — they describe the still and are allowed; only clear leaks fail.
HARD_LEAK = re.compile(r"""\b(
    transform(?:s|ing|ed|ation)? | morph(?:s|ing|ed)? | becom(?:e|es|ing) |
    turn(?:s|ing)?\s+into | turning\s+into | dissolv(?:e|es|ing|ed) |
    erupt(?:s|ing|ed)? | explod(?:e|es|ing|ed) | shatter(?:s|ing|ed)? |
    disintegrat(?:e|es|ing) | material?is(?:e|es|ing) | vanish(?:es|ing|ed)? |
    VFX | CGI | shader | overlay | glitch | \btransition\b |
    the\s+(?:image|frame|photo|video|footage|shot)\s+shows |
    in\s+this\s+(?:frame|image|shot) |
    \bmusic\b | \bspeech\b | \bsinging\b | \bvoiceover\b
)\b""", re.IGNORECASE | re.VERBOSE)
# soft: worth a human glance but not an auto-fail (state-vs-process ambiguous)
SOFT_WATCH = re.compile(r"\b(glow(?:s|ing)?|beam|aura|energy|magical?)\b", re.IGNORECASE)

WORD_LO, WORD_HI = 10, 56          # generous band around the per-item targets (13..47 ± spread)


def _selection_subjects() -> set[str]:
    sel = json.loads(SELECTION.read_text())
    # selection schema: {policy, n_subjects, n_clips, ..., clips:[{subject, effect, ...}]}
    if isinstance(sel, dict) and "clips" in sel:
        return {c["subject"] for c in sel["clips"]}
    if isinstance(sel, dict) and "subjects" in sel:
        subs = sel["subjects"]
    elif isinstance(sel, list):
        subs = sel
    else:
        subs = sel.get("selected") or sel.get("subject_ids") or []
    return {s if isinstance(s, str) else (s.get("subject") or s.get("id")) for s in subs}


def _merged() -> dict[str, str]:
    if SOURCE == "regen":
        d = json.loads(REGEN_JSON.read_text())        # {subject: caption} (2000)
        return {k: v.strip() for k, v in d.items() if isinstance(v, str)}
    out: dict[str, str] = {}
    dup: list[str] = []
    files = [CAPDIR / "pilot_captions.json"] + sorted(glob.glob(str(CAPDIR / "out_*.json")))
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for k, v in d.items():
            if k in out and out[k] != v:
                dup.append(k)
            out[k] = v.strip() if isinstance(v, str) else v
    if dup:
        print(f"[warn] {len(dup)} subjects captioned in >1 file with DIFFERING text: {dup[:5]}")
    return out


def _word_count(s: str) -> int:
    return len(s.split())


def validate() -> tuple[dict[str, str], list[str]]:
    caps = _merged()
    want = _selection_subjects()
    problems: list[str] = []

    missing = sorted(want - set(caps))
    extra = sorted(set(caps) - want)
    print(f"coverage: {len(caps)} captions | selection {len(want)} subjects | "
          f"missing {len(missing)} | extra {len(extra)}")
    if missing:
        # which out_*.json still owe these? map subject -> batch
        problems.append(f"MISSING {len(missing)}: {missing[:8]}")
    if extra:
        problems.append(f"EXTRA {len(extra)} not in selection: {extra[:8]}")

    bad_fmt, bad_len, leaks, soft = [], [], [], []
    for sub, cap in caps.items():
        if not isinstance(cap, str) or not cap.strip():
            bad_fmt.append(sub); continue
        c = cap.strip()
        if not c.endswith("."):
            bad_fmt.append(sub)
        if c.count(".") > 3 or "\n" in c or c.startswith(("-", "*", '"', "`")):
            bad_fmt.append(sub)
        wc = _word_count(c)
        if not (WORD_LO <= wc <= WORD_HI):
            bad_len.append((sub, wc))
        if HARD_LEAK.search(c):
            leaks.append((sub, HARD_LEAK.search(c).group(0)))
        elif SOFT_WATCH.search(c):
            soft.append((sub, SOFT_WATCH.search(c).group(0)))

    print(f"format bad: {len(bad_fmt)} | length out-of-band: {len(bad_len)} | "
          f"HARD leaks: {len(leaks)} | soft-watch: {len(soft)}")
    if bad_fmt:
        problems.append(f"FORMAT {len(bad_fmt)}: {bad_fmt[:8]}")
    if bad_len:
        problems.append(f"LENGTH {len(bad_len)}: {bad_len[:8]}")
    if leaks:
        problems.append(f"HARD-LEAK {len(leaks)}: {leaks[:12]}")
    if soft:
        print(f"  [soft-watch, not failing] {soft[:12]}")

    print("RESULT:", "CLEAN ✓" if not problems else "PROBLEMS ✗")
    for p in problems:
        print("  -", p)
    return caps, problems


def _content_hash(descriptions: dict[str, str]) -> str:
    blob = json.dumps(dict(sorted(descriptions.items())), sort_keys=True, ensure_ascii=True)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assemble() -> int:
    caps, problems = validate()
    if problems:
        print("\n[assemble] REFUSING to write — validation not clean. Fix the above first.")
        return 1
    # descriptions keyed '<subject>|A'
    # descriptions carry NO trailing period (S4 convention); build_encode_inputs adds `. sksz.`
    descriptions = {f"{sub}|A": _strip_terminal_period(cap) for sub, cap in sorted(caps.items())}
    ch = _content_hash(descriptions)

    roster = json.loads(ROSTER.read_text())
    n_clips = roster["n_clips"]
    # subject -> #clips, to record the 1:N expansion Lane A undergoes at assembly
    from collections import Counter
    per_sub = Counter(c["subject"] for c in roster["clips"])

    SHELF.mkdir(parents=True, exist_ok=True)
    store = {
        "schema": "ctt_v2_s6_caption_store/v1",
        "stratum": "S6",
        "keying": "'<subject>|A'. EffectData is ONE-SIDED and Axis-A counterfactual: a subject's "
                  "first frame is identical across all its clips, so ONE A-description per subject "
                  "(2000) is reused by that subject's clips at assembly (28,644 clips total). There "
                  "is no role-B description (frame 0 alone is conditioned: latent frame 0, prefix_latents=1).",
        "sided_authority": "roster sided='one'; prefix_latents=1 (RULED_SHAPES, commit 117daa0)",
        "generator": GENERATOR[SOURCE],
        "prompt_variant": "v2-s4f0",
        "prompt_variant_delta": "S4 prompt v2 role-A verbatim; spec text CAPTION_TASK.md carried into this shelf.",
        "spec_sha256": hashlib.sha256(SPEC_SRC.read_bytes()).hexdigest() if SPEC_SRC.exists() else None,
        "source_captions": "NONE editable — EffectData ships an effect instruction per clip (the operator), "
                           "never a first-frame description. A-descriptions are authored fresh from the "
                           "first frame; the effect instruction is deliberately NOT used (it is the leak).",
        "counts": {"subjects": len(caps), "clips": n_clips,
                   "clips_per_subject_p50": sorted(per_sub.values())[len(per_sub) // 2]},
        "content_hash": ch,
        "content_hash_covers": "the `descriptions` map only, keyed '<subject>|A': "
                               "json.dumps(sorted dict, sort_keys=True, default separators, ensure_ascii=True), "
                               "sha256 of the utf-8 bytes.",
        "descriptions": descriptions,
    }
    out = SHELF / "EFFECTDATA_CAPTION_STORE.json"
    out.write_text(json.dumps(store, indent=1, ensure_ascii=True) + "\n")
    # carry the spec + a lock (hash) into the shelf
    if SPEC_SRC.exists():
        (SHELF / "CAPTION_TASK.md").write_text(SPEC_SRC.read_text())
    (SHELF / "CAPTION_LOCK.json").write_text(json.dumps(
        {"content_hash": ch, "n_subjects": len(caps), "n_clips": n_clips,
         "schema": store["schema"]}, indent=1) + "\n")
    # shelf descriptor (mirrors captions/002_ctt_v2_s4/meta.yaml)
    meta = f"""id: captions/004_effectdata
seq: 4
shelf: captions
created: 2026-08-28
role: S6 (EffectData) first-frame A descriptions (one-sided) — Lane A training text
keyed: "<subject>|A"          # per-subject; reused across the subject's clips (Axis-A shared first frame)
coverage: {len(caps)}/2000 subjects -> {n_clips} clips
content_hash: {ch}   # covers the `descriptions` map (see store JSON `content_hash_covers`)
prompt_variant: v2-s4f0       # S4 role-A prompt; spec CAPTION_TASK.md carried into this shelf
generator: {GENERATOR[SOURCE]}  Descriptions stored WITHOUT trailing period (S4 convention; build_encode_inputs adds '. sksz.'). Assembled + hashed by scripts/ctt_v2/s6/build_caption_store.py.
spec: CAPTION_TASK.md
leak_gate: generic HARD-leak scan (process/effect/frame/sound families) — 0 hard leaks; 20 state-word soft-watch (glow/beam of sunlight) accepted as literal still-state per spec.
assembled_by: scripts/ctt_v2/s6/build_caption_store.py -> descriptions '<subject>|A'; training grammar "{{A}}. sksz." (one-sided) at root assembly
source: data/processed/effectdata/captions/{{pilot_captions,out_00..out_24}}.json
authority: ../../TEXT_LIFECYCLE.md   # §2 Lane A
"""
    (SHELF / "meta.yaml").write_text(meta)
    print(f"\n[assemble] wrote {out}")
    print(f"[assemble] content_hash = {ch}")
    print(f"[assemble] {len(caps)} subjects -> {n_clips} clips (p50 {store['counts']['clips_per_subject_p50']}/subject)")
    return 0


def main() -> int:
    global SOURCE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["validate", "assemble"])
    ap.add_argument("--source", choices=["regen", "batches"], default=SOURCE,
                    help="regen = per-image gemini regen (default, alignment-safe); "
                         "batches = superseded opus fan-out")
    args = ap.parse_args()
    SOURCE = args.source
    if args.cmd == "validate":
        _, problems = validate()
        return 1 if problems else 0
    return assemble()


if __name__ == "__main__":
    raise SystemExit(main())
