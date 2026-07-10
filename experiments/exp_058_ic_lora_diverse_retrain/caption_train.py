"""exp_058 — type-blind endpoint captions for all training clips.

For every clip in the exp_058 training classes (design.md §3) that lacks a
caption in exp_056/exp_057 caption files, sends the standardized FIRST and
LAST frame stills to Gemini and composes the exp_056 format:
"<scene A>. The scene transforms into <scene B>."

Type-blind by construction: the model sees two stills, never the motion, and
is told to describe visible content only. A banned-word scrub catches
mechanism/class vocabulary; violations are reported for manual fixing.

Output: dataset/captions.json  {clip_stem: caption}  (merged reused+new),
        dataset/captions_provenance.json.
Login-node safe (CPU + API). Idempotent: cached per-clip responses.
"""

import json
import os
import pathlib
import sys
import time

import cv2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT_DIR = pathlib.Path(__file__).parent / "dataset"
CACHE = OUT_DIR / "caption_cache"

TRAIN_CLASSES = [
    # two-sided (exp_056 originals minus raven)
    "air_bending", "display_transition", "earth_wave", "firelava", "flame",
    "flying_cam_transition", "melt_transition", "shadow_smoke", "water_bending",
    # one-sided, exp_057-standardized
    "animalization", "fire_element", "giant_grab", "money_rain",
    "plasma_explosion", "portal", "shadow", "super_fast_run", "wireframe",
    # one-sided, exp_058-standardized
    "color_rain", "cotton_cloud", "earth_element", "live_concert",
    "luminous_gaze", "monstrosity", "mystification", "nature_bloom",
    "polygon", "saint_glow", "sakura_petals", "water_element", "wonderland",
    "run_set_on_fire",
]

# mechanism/class words that must not appear (type-blindness scrub)
BANNED = [
    "transition", "transform", "morph", "dissolve", "vanish", "disintegrat",
    "portal", "wireframe", "polygon", "hologram", "effect", "vfx", "cgi",
    "animation", "timelapse", "sequence",
]
# note: "transforms" appears once in the fixed template joiner, which is fine —
# the scrub applies to the Gemini-written scene descriptions only.

PROMPT = (
    "Describe this still image in ONE sentence for a video caption. Describe "
    "only what is visibly present: subject, appearance, clothing, setting, "
    "lighting. Do not speculate about motion, causes, or what happens next. "
    "Do not use words about effects, transformations, or editing. Plain "
    "declarative sentence, no preamble."
)

MODEL = "gemini-3.5-flash"
_RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")


def api_key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    p = pathlib.Path.home() / ".config/gemini/api_key"
    if p.exists():
        return p.read_text().strip()
    sys.exit("no GEMINI_API_KEY and no ~/.config/gemini/api_key")


def endpoint_frames(clip: pathlib.Path) -> tuple[bytes, bytes]:
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    for want in (0, n - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, fr = cap.read()
        assert ok, f"{clip} frame {want} unreadable"
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert ok
        frames[want] = buf.tobytes()
    cap.release()
    return frames[0], frames[n - 1]


def describe(client, types_mod, jpg: bytes, cache_file: pathlib.Path) -> str:
    if cache_file.exists():
        return cache_file.read_text().strip()
    part = types_mod.Part(inline_data=types_mod.Blob(mime_type="image/jpeg", data=jpg))
    cfg = types_mod.GenerateContentConfig(temperature=0.0)
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=[part, PROMPT], config=cfg)
            text = resp.text.strip().rstrip(".")
            cache_file.write_text(text)
            return text
        except Exception as e:  # noqa: BLE001
            if any(k in str(e) for k in _RETRYABLE) and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)

    reused = {}
    for f in [
        REPO_ROOT / "experiments/exp_056_ltx2_ic_lora_transition_transfer/dataset/captions.json",
        REPO_ROOT / "experiments/exp_057_ic_lora_unseen_eval/dataset/captions.json",
    ]:
        reused.update(json.loads(f.read_text()))

    from google import genai
    types_mod = __import__("google.genai.types", fromlist=["types"])
    client = genai.Client(api_key=api_key())

    captions, provenance, violations = {}, {}, []
    todo = []
    for cls in TRAIN_CLASSES:
        for clip in sorted((STD / cls).glob("*.mp4")):
            todo.append((cls, clip))
    print(f"[info] {len(todo)} training clips across {len(TRAIN_CLASSES)} classes")

    for i, (cls, clip) in enumerate(todo):
        stem = clip.stem
        if stem in reused:
            captions[stem] = reused[stem]
            provenance[stem] = "reused"
            continue
        first, last = endpoint_frames(clip)
        a = describe(client, types_mod, first, CACHE / f"{stem}__first.txt")
        b = describe(client, types_mod, last, CACHE / f"{stem}__last.txt")
        cap = f"{a}. The scene transforms into {b[0].lower()}{b[1:]}."
        captions[stem] = cap
        provenance[stem] = "gemini"
        low = (a + " " + b).lower()
        hits = [w for w in BANNED if w in low]
        if hits:
            violations.append((stem, hits))
        if (i + 1) % 20 == 0:
            print(f"[info] {i + 1}/{len(todo)}")

    (OUT_DIR / "captions.json").write_text(json.dumps(captions, indent=1))
    (OUT_DIR / "captions_provenance.json").write_text(json.dumps(provenance, indent=1))
    n_new = sum(1 for v in provenance.values() if v == "gemini")
    print(f"[done] {len(captions)} captions ({n_new} new via Gemini, "
          f"{len(captions) - n_new} reused)")
    if violations:
        print(f"[WARN] banned-word hits ({len(violations)}) — fix manually:")
        for stem, hits in violations:
            print(f"  {stem}: {hits}")


if __name__ == "__main__":
    main()
