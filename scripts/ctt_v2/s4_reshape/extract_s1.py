"""ctt_v2 S4 — turn refVFX I2V-LoRA prompts into campaign S1 captions.

OWNER RULING (2026-07-25): every S4 sample is ONE-SIDED, so the rendered prompt is
``"{S1}. sksz."`` and no S2 is needed. That removes the outcome half entirely -- which is
the half that leaked the transition in every prior ladder -- so the only thing to extract is
a static snapshot of the START scene.

WHY REWRITING BEATS RE-CAPTIONING
---------------------------------
The refVFX prompts are formulaic to a degree that makes vision unnecessary for 96% of rows:

    [0] The video starts with A middle-aged East Asian woman.        <- S1
    [1] The c1455y classy transformation occurs and ... is now ...   <- transition (has trigger)
    [2] ... is smiling and looking at the camera.                    <- outcome
    [3] The background is golden and luxurious.                      <- outcome

Measured on all 6,994 usable rows: the obfuscated trigger token appears in 100% of prompts,
which makes the transition sentence trivially locatable, and a clean S1 falls out of the
first trigger-free sentence for **96.3%** of rows. The 256-row residue goes to the locally
cached Gemma-3-12B (no API key needed, no network) rather than an API.

WHAT IS BEING GUARDED AGAINST
-----------------------------
Two distinct leaks, and they need different treatment:

1.  **The obfuscated trigger token** (``c1455y``, ``s31lf13``) is an unambiguous effect label
    from another model. Banned outright, checked on every output.
2.  **Change vocabulary** ("transforms", "then", "becomes", "during the") describes the
    transition rather than the start scene. Rejected.

Deliberately NOT banned: the trigger's english gloss words as a set. Triggers like
"4ct3ion Action Run" and "l4a6ing laughing" gloss to common English, and banning those words
wholesale rejected 526 perfectly good start-scene sentences. Likewise "looking at the camera"
and "begins with a neutral expression" are static descriptions, not camera moves or
transitions -- an earlier, cruder filter rejected 1,115 rows on those two phrases alone.

Every emitted S1 is re-verified against both bans regardless of which path produced it.

Usage:
    python scripts/ctt_v2/s4_reshape/extract_s1.py            # rules only, reports residue
    python scripts/ctt_v2/s4_reshape/extract_s1.py --llm --device cuda   # + Gemma for residue
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

INDEX = LAB / "diffusion-research/data/raw/refvfx/_viewer_index/lora.jsonl.gz"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
OUT = REPO_ROOT / "data/processed/s4_refvfx"

#: lead-ins that name the medium; stripped, not rejected
LEAD = re.compile(
    r"^\s*(?:(?:in|at)\s+the\s+(?:video|clip)\s*,?\s*|"
    r"the\s+(?:video|clip|image|scene)\s+(?:starts?|begins?|opens?)\s+(?:with|on)\s+|"
    r"the\s+(?:video|clip)\s+shows\s+|the\s+first\s+frame\s+shows\s+|"
    r"(?:a\s+)?close[-\s]up\s+of\s+)+",
    re.I,
)
#: genuine change vocabulary -- describes the transition, not the start scene
TRANS = re.compile(
    r"\b(then|transform\w*|transition\w*|turns?\s+into|morph\w*|suddenly|after\s+that|"
    r"begins?\s+to|starts?\s+to|during\s+the|is\s+now|are\s+now|becomes?)\b",
    re.I,
)
MIN_WORDS = 4


def trigger_token(effect: str) -> str:
    """The obfuscated id, e.g. 'c1455y' from 'c1455y classy transformation'."""
    return effect.split()[0].lower()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def tidy(text: str) -> str:
    t = LEAD.sub("", text).strip().rstrip(".,;: ").strip()
    t = " ".join(t.split())
    # refVFX splices a capitalised subject mid-sentence ("on A young African man") and emits
    # "A elderly" regardless of the following vowel. Fix the article, then lowercase it --
    # capitalisation is restored for the sentence head only, below, so we never emit
    # "a portrait of An elderly woman".
    t = re.sub(r"\bA\s+(?=[aeiouAEIOU])", "an ", t)
    t = re.sub(r"\bA\s+(?=[^aeiouAEIOU\s])", "a ", t)
    if t:
        t = t[0].upper() + t[1:]
    return t


def verify(s1: str, token: str) -> str | None:
    """Return a rejection reason, or None if the caption is clean."""
    if not s1 or len(s1.split()) < MIN_WORDS:
        return "too_short"
    if token in s1.lower():
        return "trigger_token"
    if TRANS.search(s1):
        return "change_vocabulary"
    return None


def extract_by_rule(prompt: str, effect: str) -> str | None:
    token = trigger_token(effect)
    for sent in sentences(prompt)[:3]:
        if token in sent.lower():
            continue
        cand = tidy(sent)
        if verify(cand, token) is None:
            return cand
    return None


def load_rows() -> list[dict]:
    rows = [json.loads(x) for x in gzip.open(INDEX, "rt")]
    return [r for r in rows if r["pr"].strip() and r["et"].strip()]


LLM_INSTRUCTION = (
    "Below is a caption describing a video in which an effect is applied to a person.\n"
    "Rewrite ONLY the opening scene -- how things look BEFORE the effect -- as one static "
    "sentence of 8-30 words describing the person's appearance, clothing, expression and "
    "setting.\n"
    "Do NOT mention the effect, the transformation, what changes, what happens next, or any "
    "code-like word such as '{token}'. Do NOT mention a video, clip or frame.\n"
    "Reply with the sentence only.\n\nCaption: {prompt}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use local Gemma for the rule residue")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict] = {}
    residue: list[dict] = []
    for r in rows:
        s1 = extract_by_rule(r["pr"], r["et"])
        if s1 is None:
            residue.append(r)
        else:
            out[r["k"]] = {"s1": s1, "effect": r["et"], "via": "rule"}

    print(f"[s4] rows={len(rows)}  rule={len(out)} ({len(out) / len(rows):.1%})  "
          f"residue={len(residue)}")

    if args.llm and residue:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        print(f"[s4] loading {GEMMA.name} for {len(residue)} residue rows", flush=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            GEMMA, torch_dtype=torch.bfloat16, device_map=args.device
        ).eval()
        proc = AutoProcessor.from_pretrained(GEMMA)

        rescued = 0
        for i, r in enumerate(residue, 1):
            token = trigger_token(r["et"])
            msg = [{"role": "user", "content": [{"type": "text", "text":
                    LLM_INSTRUCTION.format(token=token, prompt=r["pr"])}]}]
            inputs = proc.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            in_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=90, do_sample=False)
            cand = tidy(proc.decode(gen[0][in_len:], skip_special_tokens=True).strip('"'))
            cand = sentences(cand)[0] if sentences(cand) else cand
            cand = tidy(cand)
            if verify(cand, token) is None:
                out[r["k"]] = {"s1": cand, "effect": r["et"], "via": "gemma"}
                rescued += 1
            if i % 25 == 0 or i == len(residue):
                print(f"[s4] residue {i}/{len(residue)} rescued={rescued}", flush=True)
        print(f"[s4] gemma rescued {rescued}/{len(residue)}")

    # Final gate: nothing leaves this script unverified, whatever produced it.
    bad = [(k, v["s1"], verify(v["s1"], trigger_token(v["effect"]))) for k, v in out.items()
           if verify(v["s1"], trigger_token(v["effect"])) is not None]
    if bad:
        raise SystemExit(f"[s4] FATAL: {len(bad)} captions failed final verification, "
                         f"e.g. {bad[:3]}")

    path = OUT / "s1_captions.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    by = {}
    for v in out.values():
        by[v["via"]] = by.get(v["via"], 0) + 1
    print(f"[s4] wrote {len(out)}/{len(rows)} verified S1 captions {by} -> {path}")


if __name__ == "__main__":
    main()
