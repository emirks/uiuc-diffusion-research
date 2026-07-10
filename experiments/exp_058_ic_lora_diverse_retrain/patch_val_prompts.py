"""exp_058 — after caption_train.py: fill the __VAL2/3_PROMPT__ placeholders.

VAL2 = ICTRANS + caption(water_element_0)   (trained one-sided class)
VAL3 = ICTRANS + caption(hero_flight_0)     (HELD-OUT class -> captioned here
                                             via the same Gemini path, cached)
Patches config_ic2.yaml AND config_ic2_sanity.yaml in place. Idempotent.
"""

import json
import pathlib

from caption_train import CACHE, OUT_DIR, api_key, describe, endpoint_frames

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).parent
STD = REPO_ROOT / "data/processed/transitions_std121"


def caption_for(stem: str, cls: str, captions: dict) -> str:
    if stem in captions:
        return captions[stem]
    from google import genai
    types_mod = __import__("google.genai.types", fromlist=["types"])
    client = genai.Client(api_key=api_key())
    first, last = endpoint_frames(STD / cls / f"{stem}.mp4")
    a = describe(client, types_mod, first, CACHE / f"{stem}__first.txt")
    b = describe(client, types_mod, last, CACHE / f"{stem}__last.txt")
    return f"{a}. The scene transforms into {b[0].lower()}{b[1:]}."


def main():
    captions = json.loads((OUT_DIR / "captions.json").read_text())
    val2 = "ICTRANS " + caption_for("water_element_0", "water_element", captions)
    val3 = "ICTRANS " + caption_for("hero_flight_0", "hero_flight", captions)
    print(f"[val2] {val2}\n[val3] {val3}")
    import yaml
    assert '"' not in val2 + val3, "caption contains a double quote — patch manually"
    for name in ("config_ic2.yaml", "config_ic2_sanity.yaml"):
        p = EXP / name
        t = p.read_text()
        t = t.replace("__VAL2_PROMPT__", val2).replace("__VAL3_PROMPT__", val3)
        assert "__VAL" not in t, f"placeholder left in {name}"
        cfg = yaml.safe_load(t)  # must still parse
        assert cfg["validation"]["samples"][1]["prompt"].startswith("ICTRANS ")
        p.write_text(t)
        print(f"[done] {name}")


if __name__ == "__main__":
    main()
