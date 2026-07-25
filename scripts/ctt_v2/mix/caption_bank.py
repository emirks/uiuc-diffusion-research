"""ctt_v2 — caption the shared endpoint bank for S2/S3 training prompts.

WHY THIS EXISTS
---------------
S2 (2D shader) and S3 (3D camera) procedural samples are built from the 227-clip neutral
endpoint bank. The bank ships `clip_id / mp4 / source / label / score / bbox_area` and NO
captions, and the S2/S3 hand-off spec does not produce any either. But every training sample
in this campaign needs a prompt, and prompts are RENDERED, never authored
(`eval_ladder/prompts.py` is the only renderer):

    one-sided  ->  "{S1}. {token}."
    two-sided  ->  "{S1}. {token}. {S2}."

For a procedural sample the transition runs from content A to content B, so
S1 = a static snapshot of A's FIRST frame and S2 = a static snapshot of B's LAST frame.
Both endpoints of every bank clip therefore need a caption: 227 clips x 2 = 454 captions.

WHAT IS BEING GUARDED AGAINST
-----------------------------
The campaign's worst caption defect was TRANSITION LEAKAGE — captions that describe the
change ("... The scene transforms into ...") teach the model to read the effect off the text
instead of the reference. So the instruction below demands a single static sentence and the
output is filtered: any caption containing motion/transition/camera vocabulary is rejected and
retried, then hard-failed. A leaked caption is worse than a missing one.

The bank directory is READ-ONLY by owner ruling. Output goes to our own tree.

Run (needs a GPU; ~12B model, bf16):
    python scripts/ctt_v2/mix/caption_bank.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

BANK = LAB / "diffusion-research/data/processed/synth_endpoints"  # READ-ONLY
BANK_JSON = BANK / "bank_tightened.json"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
OUT = REPO_ROOT / "data/processed/synth_endpoint_captions"

#: The caption style is copied from the audited corpus captions, e.g.
#: "A young man with slicked-back dark hair and a serious expression crouches in front of a
#:  chain-link fence, wearing a bright red sweatshirt and dark pants under soft natural light."
INSTRUCTION = (
    "Describe this single video frame as one static snapshot, in ONE sentence of 20-40 words.\n"
    "Name the main subject, their appearance and clothing, what they are doing, the setting "
    "behind them, and the lighting.\n"
    "Write it as a still photograph caption. Do NOT mention the camera, camera movement, "
    "zooming, panning, a video, a clip, frames, time passing, anything changing, transforming, "
    "or transitioning. Do NOT begin with 'The image shows' or 'This frame'.\n"
    "Reply with the sentence only, no preamble and no quotation marks."
)

#: vocabulary that means the caption drifted from a static snapshot into describing change
_BANNED = re.compile(
    r"\b(transition|transform|transforms|transforming|morph\w*|dissolv\w*|fade[sd]?|fading|"
    r"camera|zoom\w*|pan(?:s|ning)?|tilt\w*|dolly|tracking shot|"
    r"video|clip|footage|frame|scene changes|begins to|starts to|then\b|"
    r"gradually|slowly (?:becomes|turns|shifts))\b",
    re.IGNORECASE,
)
_PREAMBLE = re.compile(r"^\s*(?:here(?:'s| is)[^:]*:|the image shows|this (?:image|frame|photo)[^,]*,)\s*",
                       re.IGNORECASE)


def clean(text: str) -> str:
    """Strip preamble/quotes/whitespace and collapse to a single sentence-ish string."""
    t = text.strip().strip('"').strip()
    t = _PREAMBLE.sub("", t).strip()
    t = " ".join(t.split())
    # keep only the first sentence if the model produced several
    parts = re.split(r"(?<=[.!?])\s+", t)
    t = parts[0] if parts else t
    return t.rstrip(".,;: ").strip()


def load_bank() -> list[dict]:
    return json.loads(BANK_JSON.read_text())["clips"]


def read_endpoint_frames(mp4: Path) -> tuple["Image.Image", "Image.Image"]:  # noqa: F821
    """Return (first frame, last frame) as PIL images."""
    import decord  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    vr = decord.VideoReader(str(mp4))
    n = len(vr)
    first = Image.fromarray(vr[0].asnumpy())
    last = Image.fromarray(vr[n - 1].asnumpy())
    return first, last


def read_endpoint_frames_cv2(mp4: Path):
    """decord-free fallback: OpenCV, which is already a dependency of the eval harness."""
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read first frame of {mp4}")
    first = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - 1, 0))
    ok, frame = cap.read()
    if not ok:  # some containers refuse the exact last index; walk back
        for i in range(2, 6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - i, 0))
            ok, frame = cap.read()
            if ok:
                break
    if not ok:
        raise RuntimeError(f"cannot read last frame of {mp4}")
    last = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return first, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap clips (smoke runs)")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "bank_endpoint_captions.json"
    done: dict[str, dict] = json.loads(out_path.read_text()) if out_path.exists() else {}

    clips = load_bank()
    if args.limit:
        clips = clips[: args.limit]
    todo = [c for c in clips if c["clip_id"] not in done]
    print(f"[caption] bank={len(clips)} already_done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("[caption] nothing to do")
        return

    print(f"[caption] loading {GEMMA.name}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        GEMMA, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    processor = AutoProcessor.from_pretrained(GEMMA)

    def caption_one(img) -> str:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": INSTRUCTION}]}]
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        return processor.decode(gen[0][in_len:], skip_special_tokens=True)

    rejected = 0
    for i, c in enumerate(todo, 1):
        mp4 = BANK / c["mp4"]
        try:
            first, last = read_endpoint_frames_cv2(mp4)
        except Exception as exc:  # noqa: BLE001
            print(f"[caption][skip] {c['clip_id']}: {exc}", flush=True)
            continue

        caps = {}
        for role, img in (("s1", first), ("s2", last)):
            text = ""
            for attempt in range(args.max_retries):
                text = clean(caption_one(img))
                if text and not _BANNED.search(text):
                    break
                rejected += 1
                text = ""
            if not text:
                raise SystemExit(
                    f"[fatal] {c['clip_id']}/{role}: no leak-free caption in "
                    f"{args.max_retries} tries — a leaked caption would teach the model to read "
                    f"the effect off the text instead of the reference"
                )
            caps[role] = text

        done[c["clip_id"]] = {**caps, "source": c["source"], "label": c["label"]}
        if i % 10 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done, indent=1, sort_keys=True))
            print(f"[caption] {i}/{len(todo)} written (rejected so far {rejected})", flush=True)

    out_path.write_text(json.dumps(done, indent=1, sort_keys=True))
    print(f"[caption] DONE clips={len(done)} captions={2 * len(done)} rejected={rejected}")
    print(f"[caption] -> {out_path}")
    sample = next(iter(done.values()))
    print(f"[caption] sample s1: {sample['s1']}")
    print(f"[caption] sample s2: {sample['s2']}")


if __name__ == "__main__":
    main()
