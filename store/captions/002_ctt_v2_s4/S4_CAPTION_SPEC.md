# S4 first-frame caption spec (role A, prompt variant `v2-s4f0`)

You are writing one-sentence descriptions of **single still frames** for a film-production
shot list. Each frame is the **first frame** of a short clip — the only frame the model is
conditioned on. Write exactly ONE English sentence describing **only what is visible in that
frame**.

## Structure — match this exactly

Begin with the main subject and its appearance (`A woman with…`, `A young man in…`), then one
or two comma-separated appositive phrases giving clothing and features, then a single finite
main verb, then the setting and its lighting. Name the plain colour of every significant
garment, object and light source. Use plain literal terms (`red dress`, not `vibrant red
dress`). Verbs in simple present or present progressive.

Real examples from the certified corpus — copy this register, not the content:

> A barefoot woman with long brown hair, wearing a red short-sleeved blouse and black pants, runs forward through a hazy parking garage illuminated by green overhead lights and a warm orange glow.

> A man with short brown hair, wearing a beige long-sleeved shirt, light blue jeans, and white sneakers, balances on one hand on a grey gravel driveway while kicking one leg high into the air in front of a stone wall lined with potted purple flowers.

> A grey and white dog runs through a series of red, white, and green weave poles on a green artificial turf field backed by a green fence and trees under bright daylight.

## Length — this is a measured target, not a suggestion

Aim for **about 34 words**; 26–41 is the healthy band, 16–50 the hard limit. Average about
**2 commas** per sentence — the appositive structure above produces them naturally. End with a
period.

## Hard prohibitions

1. **Describe only this instant.** Nothing about what happens before or after. No language
   about the scene changing, transforming, becoming, turning into, beginning, ending,
   shifting, revealing, morphing, or exploding.
2. **Never name a visual effect, editing technique, or animation style.** No `VFX`, `CGI`,
   `shader`, `overlay`, `glitch`, `transition`, `zoom effect`, `slow motion`.
3. **Never refer to the frame itself.** No `the image shows`, `in this frame`, `the photo`,
   `the video`, `the footage`, `the shot`.
4. **No sounds, music, or speech.**
5. **Camera only if the frame itself shows an obvious viewpoint** (e.g. an overhead view).
6. **No preamble, no quotes, no markdown** — the sentence alone.

## Why 1 and 2 matter

These clips are training targets for a model that must *invent* a transition. If the caption
names or foreshadows what the clip turns into, the model reads the outcome off the text
instead of learning it. A caption that leaks is worse than no caption. When a first frame
already looks mid-effect, describe the visible state literally and neutrally
(`a man in a torn grey t-shirt standing in swirling orange dust`) — never the process.
