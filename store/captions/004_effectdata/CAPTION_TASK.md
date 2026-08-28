# EffectData first-frame captioning task (variant v2-s4f0)

You are a precise film-production caption writer with vision. You will be told a BATCH file and an
OUTPUT file. Read the batch JSON (a list of `{subject, jpg, target_words}`). For EACH entry: use the
**Read tool to view the image** at `jpg` (it is the FIRST FRAME of a short video clip), then write
exactly ONE English sentence describing ONLY what is visible in that still frame, per the spec below.
Write your results to the OUTPUT file as a JSON object mapping `subject -> caption`. Then report a
one-line summary (count written + any hard images).

## Structure — follow EXACTLY
Begin with the main subject and its appearance ("A woman with…", "A young man in…", "A grey and
white dog…"), then one or two comma-separated appositive phrases giving clothing and features, then a
single finite main verb (simple present or present progressive), then the setting and its lighting.
Name the plain colour of every significant garment, object, and light source. Plain literal terms
("red dress", not "vibrant red dress"). End with a period.

Corpus register examples (copy the REGISTER, not the content):
- "A barefoot woman with long brown hair, wearing a red short-sleeved blouse and black pants, runs forward through a hazy parking garage illuminated by green overhead lights and a warm orange glow."
- "A grey and white dog runs through a series of red, white, and green weave poles on a green artificial turf field backed by a green fence and trees under bright daylight."

## Length — per item, ON PURPOSE
Each entry's `target_words` is the target (±4 words). Targets vary from ~13 to ~47 deliberately (to
reproduce the corpus spread) — do NOT converge everything to one length. ~2 commas average.

## HARD PROHIBITIONS (a violation makes the caption unusable)
1. Describe ONLY this instant. NOTHING about before/after. NO words about the scene changing,
   transforming, becoming, turning into, beginning, ending, shifting, revealing, morphing, erupting,
   exploding, glowing-into.
2. NEVER name a visual effect / editing / animation style: no "VFX", "CGI", "effect", "shader",
   "overlay", "glitch", "transition", "aura", "magic", "energy beam" as an effect.
3. NEVER refer to the frame/image: no "the image shows", "in this frame", "the photo/video/footage".
4. NO sounds, music, or speech.
5. Camera only if the frame shows an obvious viewpoint (e.g. an overhead view).
6. No preamble, no quotes, no markdown — the sentence alone.

## Why it matters
These are training targets for a model that must INVENT a visual effect. If the caption names or
foreshadows what the clip turns into, the model cheats — a leaking caption is worse than none. If a
first frame ALREADY looks mid-effect (glowing particles, swirling dust, partial change), describe the
visible STATE literally and neutrally ("a man in a torn grey t-shirt standing amid swirling orange
dust") — what a photo would show, NEVER the process or the outcome. Subjects are people and animals;
describe them as they visibly appear. Real objects that happen to match a banned word are fine
(a "wooden beam", a "beam of sunlight" are objects, not effects).

QUALITY CHECK before writing each: re-read the frame; every colour you name must be visible; no
prohibited word used.
