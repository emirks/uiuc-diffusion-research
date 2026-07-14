"""exp_061 — generate missing item captions with the EXACT exp_058/exp_060
type-blind captioner (same PROMPT, model gemini-3.5-flash, temperature 0,
two-sentence '<A>. The scene transforms into <b>.' format). Reuses the exp_060
per-frame cache pattern. Idempotent, login-node safe.

Writes dataset/captions_extra.json and patches dataset/selection.json in place.
"""

import json
import os
import pathlib
import sys
import time

import cv2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
CACHE = EXP / "dataset/caption_cache"

# verbatim from exp_058/caption_train.py (via exp_060/caption_missing.py)
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


def endpoint_frames(clip: pathlib.Path):
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    for want in (0, n - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, fr = cap.read()
        assert ok, f"{clip} frame {want} unreadable"
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert ok
        out[want] = buf.tobytes()
    cap.release()
    return out[0], out[n - 1]


def describe(client, types_mod, jpg: bytes, cache_file: pathlib.Path) -> str:
    if cache_file.exists():
        return cache_file.read_text().strip()
    part = types_mod.Part(inline_data=types_mod.Blob(mime_type="image/jpeg", data=jpg))
    cfg = types_mod.GenerateContentConfig(temperature=0.0)
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=MODEL, contents=[part, PROMPT], config=cfg)
            text = resp.text.strip().rstrip(".")
            cache_file.write_text(text)
            return text
        except Exception as e:  # noqa: BLE001
            if any(k in str(e) for k in _RETRYABLE) and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    sel = json.loads((EXP / "dataset/selection.json").read_text())
    missing = [it for it in sel["items"] if not it["caption"]]
    if not missing:
        print("[done] no missing captions")
        return
    print(f"[info] {len(missing)} missing caption(s): {[it['clip'] for it in missing]}")

    from google import genai
    types_mod = __import__("google.genai.types", fromlist=["types"])
    client = genai.Client(api_key=api_key())

    extra_path = EXP / "dataset/captions_extra.json"
    new_caps = json.loads(extra_path.read_text()) if extra_path.exists() else {}
    for it in missing:
        cp = REPO_ROOT / it["clip_rel"]
        stem = it["clip"]
        first, last = endpoint_frames(cp)
        a = describe(client, types_mod, first, CACHE / f"{stem}__first.txt")
        b = describe(client, types_mod, last, CACHE / f"{stem}__last.txt")
        cap = f"{a}. The scene transforms into {b[0].lower()}{b[1:]}."
        new_caps[stem] = cap
        print(f"[caption] {stem}: {cap}")

    extra_path.write_text(json.dumps(new_caps, indent=2))
    for it in sel["items"]:
        if not it["caption"]:
            it["caption"] = new_caps[it["clip"]]
            it["caption_provenance"] = f"gemini:{MODEL} (exp_061, exp_058 pipeline)"
    (EXP / "dataset/selection.json").write_text(json.dumps(sel, indent=2))
    print("[done] patched selection.json")


if __name__ == "__main__":
    main()
