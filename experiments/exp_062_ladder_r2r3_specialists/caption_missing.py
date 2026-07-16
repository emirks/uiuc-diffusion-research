"""exp_062 — caption the roster train clips missing from exp_058 captions.json,
using the EXACT exp_058/060/061 type-blind captioner (gemini-3.5-flash, temp 0,
two-sentence '<A>. The scene transforms into <b>.'). Login-node safe, idempotent,
per-frame cached. Writes dataset/captions_r2.json (only the newly captioned clips).

The 8 R4-tier roster classes are already fully covered by exp_058 captions.json;
this fills the 3 held-out R5-tier classes (gas_transformation, hero_flight,
illustration_scene = 24 train clips) so their specialists can train.
"""
import io, json, os, pathlib, sys, time
import av

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
CACHE = EXP / "dataset/caption_cache"
STD = REPO_ROOT / "data/processed/transitions_std121"
EXISTING = REPO_ROOT / "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json"
OUT = EXP / "dataset/captions_r2.json"

ROSTER = ["shadow", "portal", "super_fast_run", "shadow_smoke", "polygon",
          "wireframe", "animalization", "color_rain",
          "gas_transformation", "hero_flight", "illustration_scene"]

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
    """First and last decoded video frames as JPEG bytes (PyAV, headless-safe)."""
    with av.open(str(clip)) as container:
        frames = list(container.decode(video=0))
    assert frames, f"{clip} has no decodable frames"

    def jpg(fr) -> bytes:
        buf = io.BytesIO()
        fr.to_image().save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    return jpg(frames[0]), jpg(frames[-1])


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
    existing = json.loads(EXISTING.read_text())
    split = json.loads((STD / "split_v1.json").read_text())
    cls = split.get("classes", split)

    def stem(x):
        return x.split("/")[-1].replace(".mp4", "")

    missing = []
    for c in ROSTER:
        for t in cls[c].get("train", []):
            s = stem(t)
            if s not in existing:
                missing.append((c, s))
    print(f"[info] {len(missing)} train clip(s) need captions: {[s for _, s in missing]}")
    if not missing:
        OUT.write_text(json.dumps({}, indent=2))
        print("[done] none missing")
        return

    from google import genai
    types_mod = __import__("google.genai.types", fromlist=["types"])
    client = genai.Client(api_key=api_key())

    new_caps = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c, s in missing:
        if s in new_caps:
            continue
        clip = STD / c / f"{s}.mp4"
        first, last = endpoint_frames(clip)
        a = describe(client, types_mod, first, CACHE / f"{s}__first.txt")
        b = describe(client, types_mod, last, CACHE / f"{s}__last.txt")
        cap = f"{a}. The scene transforms into {b[0].lower()}{b[1:]}."
        new_caps[s] = cap
        print(f"[caption] {s}: {cap[:90]}…")
        OUT.write_text(json.dumps(new_caps, indent=2))  # incremental save

    print(f"[done] {len(new_caps)} captions -> {OUT}")


if __name__ == "__main__":
    main()
