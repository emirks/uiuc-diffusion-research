# LTX-2 Prompting Notes — exp_024

Researched 2026-05-05 for the prompt sweep experiment.  
Primary sources: LTX official docs (ltx.io), LTX-2.3 prompt guide, Crepal AI prompting guide.

---

## Format rules

- **Single flowing paragraph** — no bullet points, no shot lists.
- **Present tense throughout** — "glides", not "glided".
- **Length: 4–8 sentences, ≈ 150–200 words.** Longer prompts consistently outperform short ones for videos over 5 s, but past ~200 words quality degrades from over-specification.
- **Chronological order** — describe events as they progress from start to end.

---

## Sentence structure (SSACAL)

Each prompt should cover these in roughly this order:

| Element | Notes |
|---------|-------|
| **S**hot framing | "Wide shot", "medium close-up", "over-the-shoulder" |
| **S**ubject | Specific physical detail — plumage color, clothing, vehicle make, posture |
| **A**ction | What the subject does; keep to one primary action |
| **C**amera | One camera movement only — "tripod-locked", "slow dolly in", "handheld track". Do not combine movements. |
| **A**udio cue | "Silent" or a single sound descriptor. Model generates audio; naming it anchors the output. |
| **L**ight | Direction, color temperature, quality ("dappled warm", "flat overcast", "hard overhead sun") |

---

## Cinematographic language that works

LTX-2 responds well to explicit cinematographic terms. Naming these increases output fidelity:

- **Lens focal length**: "35mm lens at f/4" — affects perceived depth and compression.
- **Camera state**: "tripod-locked" vs "handheld" vs "slow dolly in".
- **Lighting**: "golden hour backlighting", "diffused overcast from above", "hard overhead sun", "rim light".
- **Texture/surface**: naming what the subject moves on ("loose mountain rock", "sunlit rippling water") anchors spatial coherence.

---

## Subject transformation and morphing (C2V context)

For clip-to-video with two conditioning endpoints, the model interpolates a latent path between the start and end conditions. The positive prompt guides what that path looks like.

**The key principle: describe the mechanism of transformation, not the fact of it.**

| Weak (what, not how) | Strong (mechanism) |
|---------------------|-------------------|
| "the black swan transforms into a mallard" | "the black plumage brightens at the edges, individual feathers lightening from jet to warm chestnut, a shimmer of iridescent green growing along the crown, the neck shortening incrementally" |
| "the longboard becomes a kiteboard" | "the tarmac beneath the wheels softens into ocean water, the wheels lifting free as the board reshapes beneath the rider's feet into a kiteboard riding on open swell" |
| "the scene transitions" | "the tree canopy thins and pulls back from both sides of the road, the diffuse forest light compressing into a single hard overhead source" |

**Rules for transformation prompts:**

1. **Describe both end states with equal detail.** The model needs a target to aim at.
2. **Name the physical bridge.** What carries the change forward? (spray, feathers rearranging, fabric gaining weight, canopy thinning)
3. **Avoid cut language.** Never use "then", "suddenly", "a new scene", "the camera cuts to", "transitions to a". These train the model toward hard cuts.
4. **Avoid dissolve language.** Never use "dissolves", "fades", "blurs into", "overlaps with", "cross-fades", "replaces". These train the model toward dissolves.
5. **Keep the subject doing something continuous.** The subject's own motion should carry the transformation — the swan gliding, the woman walking, the longboarder leaning. Do not stop the action to announce the change.
6. **One transformation mechanism per prompt.** Don't mix spray + shrink + re-color in one description; the model distributes attention and you lose clarity on each mechanism.

---

## Negative prompt guidance

The standard negative prompt for quality artifacts:

```
flicker, jitter, stutter, shaky camera, erratic motion, temporal artifacts,
frame blending, low quality, jpeg artifacts, text, watermark, logo, cartoon, anime, CGI
```

**Do not include "morphing" or "warping" in the negative prompt when the positive prompt asks for semantic transformation.** These terms suppress fluid organic motion at the text-embedding level and fight against the transformation you're prompting for.

Include "morphing" in the negative only when you want a purely static scene (e.g. for baseline categories).

---

## LTX-2 quirks relevant to C2V prompting

- **`enhance_prompt` not available in `LTX2ConditionPipeline`** (Diffusers stack). Only in the vendored `ltx-pipelines` stack.  
- **Guidance scale 3.2** — lower than the LTX-2 default (4.0) reduces over-saturation. Higher guidance increases prompt adherence but also increases contrast artifacts.
- **Transition LoRA** — A community LoRA (`valiantcat/LTX-2.3-Transition-LORA`) adds a trigger word `zhuanchang` that strengthens transformation output. Not used in exp_024 (base model only), but worth testing in a follow-up experiment.
- **Timed language in prompts** (Category E) — "at 4 seconds…", "by 6 seconds…" — the model has some temporal awareness but alignment to exact times is imprecise. Useful as a soft timing guide, not a hard one.
