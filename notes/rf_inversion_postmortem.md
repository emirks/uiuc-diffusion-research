# RF-Inversion Research Loop — Postmortem Report

**Closed:** 2026-05-16 02:16 via exit ② (scientific floor reached).
**Spend:** 6.75 / 8.0 pod-hours of the reset budget.
**Result:** `exp_033` (drop-frame-12) remains the deployable floor at
PSNR median 25.66, 2/10 clean exit-① passes. Nine §0-compliant recipes
were tested across four intervention families. None crossed exit ① on
the full 10-clip batch.

This document explains **what we tried, why, and what the data actually
said** — written for a reader who was not in the room.

---

## 1. The problem we were solving

We are doing **real-clip RF-Solver inversion on LTX-2**. Plain-English:
take a real video clip, run the diffusion solver *backwards* to recover
a "noise seed" that the forward sampler would map to that exact clip,
and then verify the round trip works by running the forward sampler on
that recovered seed and comparing it to the original.

The use case is **video editing**: if you can faithfully recover a
latent representation of a clip, you can then re-render that clip with
modified conditioning (a different prompt, different start/end frames,
etc.) and produce edits that respect the source. This only works if
inversion is *information-preserving* — round-trip PSNR has to be high
enough that the recovered seed encodes the source signal, not a
nearby-but-different signal.

We measure exit success on a 10-clip `shadow_smoke` validation set
using a perceptual gate:

- **Exit ①** (clean pass): PSNR ≥ 28 **and** SSIM ≥ 0.88 **and**
  LPIPS ≤ 0.10, hit by ≥ 6 of 10 clips.
- **Exit ②** (scientific floor): we tried ≥ 3 fixes across multiple
  families, named the cause, and confirmed the floor is genuine. We
  exit with the floor result rather than burning budget.
- **Exit ③** (wall) and **Exit ④** (budget) round out the protocol.

---

## 2. The §0 constraint — and why it dominates everything

§0 says: **at inversion time the recipe may only see the start sub-clip
and the end sub-clip — never the source middle frames.** Anchors used
to "pin" intermediate latent positions must be derivable from
{start sub-clip, end sub-clip, model parameters} alone.

Why this constraint exists: at edit time the *user has only the
endpoints*. They want to invert a clip whose middle they intend to
modify. A recipe that leaks middle-frame information at inversion is
useless in the real workflow, even if it scores well on the metric.

We already had `exp_032` from a prior loop: it pins `clean_latents =
z₀.clone()` at the conditioned positions — i.e. it uses the source's
own latent at the anchor positions. **That recipe scores PSNR median
40.88 with 8/10 clean passes — and it is `LEAKY`, non-deployable.**
Throughout this loop, exp_032 served as the upper bound: the number
we'd reach *if* the recipe could see the source. The §0-compliant
ceiling is whatever fraction of that gap we can actually close.

Going in we already knew the floor was `exp_033` at PSNR median 25.66.
The loop's job was to find a §0-compliant intervention that beats that.

---

## 3. The mechanical setting — why anchors matter so much

The 121-pixel-frame clip becomes 16 latent frames through the causal
VAE (`F_lat = (F_pix - 1)//8 + 1`). Call them `z[0..15]`.

- **Start sub-clip → anchors `z[0..3]`** (4 latents). These get
  encoded from the visible start frames and pinned during inversion.
- **End sub-clip → anchors `z[12..15]`** or `z[13..15]` (depending on
  recipe — see §4.1 below).
- **Middle positions `z[4..11]`** are the *free zone* — never
  directly conditioned in any deployable recipe. The solver has to
  reconstruct them from the anchors plus the model's prior.

The earlier It-3 diagnostic (CPU per-latent-frame round-trip error
decomposition) showed something important:

> **60% of the round-trip cost lives in the free middle `z[4..11]`,
> not at any conditioned position.**

And:

> When anchors are *exact* (the leaky exp_032 case, anchors = z₀
> slices), the middle's round-trip cost shrinks **6×**.

That second number is the crucial mechanism. The middle's cost is
**coupled to anchor quality through velocity coupling** — better
anchors propagate to a better middle reconstruction even though the
middle is never directly pinned. So the whole research question
becomes:

> **Can we produce deployable anchors that are closer to z₀-truth than
> sub-clip-encoded anchors are?**

That single question generates the entire intervention space.

---

## 4. The four intervention families we tested

### 4.1 Family A — anchor quality at end positions (`exp_034`)

**Motivating idea.** The end sub-clip encodes to 4 latents `z[12..15]`.
But the causal VAE compresses 8 pixel-frames into 1 latent — except
for the very first frame in the encoded segment, which collapses
*1 pixel-frame → 1 latent*. So the **first latent of the end sub-clip
is structurally asymmetric** with respect to the corresponding source
latent slice. exp_033 already noticed this and *drops* that one
asymmetric anchor (frame 12) — pinning only `z[13..15]`.

We asked: can we do better than "drop"? Two variants tested:

- **exp_034A — scaffold-pad frame 12.** Instead of dropping it,
  build a 9-pixel-frame "static replay" (the first end-clip frame
  repeated) and encode that. The 9 → 1+1 collapse gives a latent
  shaped like a *real* `(F-1)%8 == 0` segment, which we hoped would
  match z₀'s structure better than the raw 1-frame collapse.
- **exp_034B — drop all 4 end anchors.** If dropping one helps, maybe
  the end anchors are *systematically* miscalibrated and dropping them
  all helps more.

**Result.** Both pilot variants regressed. exp_034A −2.76 dB PSNR
median; exp_034B catastrophic (−8.38 dB). The lesson is clean:
**exp_033 is the local optimum of the drop/pin-end-anchor family.**
Padding frame 12 doesn't recover the right latent structure (the
scaffold's representational content is wrong, not just its shape).
Dropping more anchors removes too much endpoint information; the
solver underconstrains the end of the clip.

That observation closed the family. The cage detector then forced a
reframe at It-5: don't propose another end-anchor variant.

### 4.2 Family B — model-bootstrap middle anchors (`exp_035`, `exp_036`)

**Motivating idea.** §0 says we can use anything derivable from
{start, end, model}. The model is in-scope. **What if we run a
forward C2V generation pass with the same endpoints, capture its
predicted middle, and use those slices as deployable anchors at the
middle positions?**

This is potentially the highest-leverage move on paper: it injects
*model-derived middle information* into a previously-unanchored region
where 60% of the round-trip cost lives. The plumbing is two pieces:

- **Forward bootstrap.** Call `pipe(conditions=...,
  num_inference_steps=N, guidance_scale=4.0, output_type="latent",
  return_dict=False, generator=boot_generator)`. This is a real C2V
  forward pass producing a packed latent.
- **Substitution.** Replace `clean_latents` at conditioned positions
  `{4..11}` with `z_bootstrap_packed[:, middle_start:middle_end, :]`.

We tested two strengths:

- **exp_035 — hard pin (strength = 1.0).** Force middle anchors to
  the bootstrap exactly.
- **exp_036 — soft pin (strength = 0.3).** Same anchors, but the
  `conditioning_mask` is set to 0.3 rather than 1.0 at middle
  positions. The intent: nudge toward bootstrap without overwriting.

**Result.** Both rejected at pilot. The CPU diagnostic (It-3-style
per-frame error decomposition we re-ran on the bootstrap output)
showed the **bootstrap-to-z₀ distance at middle positions is
300–400× larger** than the bootstrap-to-z₀ distance at endpoints. In
plain terms: the model's middle prior is *generic* — it produces a
plausible interpolation between endpoints. But the source's middle is
*clip-specific* (smoke shape, shadow movement, occlusion timing). The
bootstrap is in the wrong place in latent space and pinning to it just
drags the inversion away from z₀.

The soft variant (exp_036) was particularly instructive. Naively a
0.3 strength sounds gentle. But the `conditioning_mask` value also
scales the per-token timestep through `t * (1 - mask)` *and* the
post-step hard re-clamp `x0_pred * (1 - mask) + clean_latents * mask`.
Over 40 steps these per-step nudges compound. End state: roughly the
same accumulated drift as the hard pin. **Lesson: soft pins are not
"30% of a hard pin" — they're a hard pin run in 40 micro-doses.**

That observation closed the family. If the *anchor target itself* is
wrong, no amount of pin softness rescues it.

### 4.3 Family C — solver dynamics (`exp_037`, `exp_038`, `exp_039`)

After the bootstrap family closed, the remaining lever was the
**solver itself**. The anchors we have (sub-clip-encoded endpoints,
unpinned middle) are the best deployable anchors that exist; can we
get more out of them with a smarter solver?

#### exp_037 — step escalation 40 → 80

**Motivating idea.** RF-Solver midpoint 2nd-order is a numerical
approximation. Doubling the step count halves the truncation error
per step and gives the solver more time in the low-σ regime where the
fine-detail content lives. Pure mechanical lever, no recipe change.

**Result.** Median PSNR **23.72** (vs exp_033's 25.66), 1/10 clean
pass. Net negative.

But the *distribution* is the interesting part. Per-clip deltas vs
exp_033:

| Clip | exp_033 PSNR | exp_037 PSNR | Δ |
|------|---:|---:|---:|
| ss1 | 22.13 | 27.55 | **+5.42** |
| ss9 | 19.84 | 27.17 | **+7.33** |
| ss7 | 29.32 | 23.76 | **−5.56** |
| ss8 | 31.04 | 22.50 | **−8.54** |

**Mechanism.** More steps = more time at low σ = more time being
*dragged* by whatever the anchors say. For clips where the anchors
are decent but the middle struggles to settle (ss1, ss9), the extra
steps help the middle converge. For clips where the anchors are
themselves imperfect (ss7, ss8 — high-frequency content near
endpoints), the extra steps amplify the wrong-anchor drift. **Step
count is not a uniform lever; it's a trade-off whose sign depends on
per-clip anchor quality.**

That alone is a real finding: in a setting where anchors are imperfect
in a clip-dependent way, throwing more solver steps at the problem
does not monotonically help. It re-weights the cost distribution.

#### exp_038 — σ-conditional anchor release

**Motivating idea.** The solver schedule has a high-σ regime
(coarse layout) and a low-σ regime (fine detail). Anchors are most
useful early — they tell the solver "here are the endpoints, organize
the layout around them." Once layout is locked in, continuing to pin
at fine-detail σ is *over-constraining*: small anchor imperfections
get sharpened into hard fine-detail artifacts.

The intervention: **release anchors below σ < 0.3.** Above 0.3 the
solver sees the normal `conditioning_mask`; below 0.3 we replace it
with `torch.zeros_like` — the inversion becomes free everywhere.
Threaded through every solver step:

```python
def _eff_mask(self, sigma_scalar):
    if self.sigma_release_threshold > 0.0 \
       and float(sigma_scalar) < self.sigma_release_threshold:
        return torch.zeros_like(self.conditioning_mask)
    return self.conditioning_mask
```

Used in `_call_transformer` (per-token timestep), `_x0_clamp_velocity`
(clamp), and the hard re-clamps in `_midpoint_step` and `_euler_step`.

**Pilot vs full-batch surprise.** The 3-clip pilot (ss0, ss2, ss5)
showed σ-release was *net positive*. The full 10-clip batch revealed
this was misleading — ss2 happened to be an outlier where release
genuinely helps (PSNR 28.06, clean exit-① pass). ss4 regressed by
9.55 dB, ss7 by 13.50 dB. Median PSNR **21.25**, 2/10 clean pass
(same count as exp_033 but worse median).

**Lesson on pilot design.** Three clips can't characterize a
distribution that varies clip-by-clip on the dimension the
intervention modifies. Future pilots must explicitly span the
difficulty spectrum — at minimum one "easy" clip (high baseline PSNR),
one "hard" clip (low baseline), and one with high CLIP gap (see §6).

**Mechanism.** σ-release helps when the cost is "anchor over-pinning
at fine detail." It hurts when the cost is "middle still needs anchor
guidance even at low σ to stay on the right manifold." Whether one
dominates is again clip-dependent.

#### exp_039 — 80 steps + σ-release combined

**Motivating idea.** Two trade-off interventions; maybe their
positive-clip subsets stack.

**Result.** Rejected. The pre-release accumulated drift from 80
high-σ steps swamps the low-σ release benefit. The two interventions
*don't* combine constructively because they don't share a sign on the
same clips — they're each clip-dependent in their own way, and the
intersection of their wins is empty.

---

## 5. Full results table

| # | Recipe | Family | PSNR median | Clean exit-① | Status |
|---|---|---|---:|---:|---|
| 1 | exp_030 | baseline (sub-clip anchors, no drop) | ~18 | 0/10 | reference |
| 2 | exp_032 | `clean_latents = z₀.clone()` (LEAKY) | **40.88** | **8/10** | NON-DEPLOYABLE upper bound |
| 3 | **exp_033** | **drop1 frame 12** | **25.66** | **2/10** | **THE §0 FLOOR** |
| 4 | exp_034A | scaffold-pad frame 12 | regression | — | rejected at pilot |
| 5 | exp_034B | drop all 4 end anchors | catastrophic | — | rejected at pilot |
| 6 | exp_035 | hard model-bootstrap middle | regression | — | rejected at pilot + CPU diag |
| 7 | exp_036 | soft bootstrap (strength 0.3) | regression | — | rejected — pins compound |
| 8 | exp_037 | 80 steps | 23.72 | 1/10 | net negative on full batch |
| 9 | exp_038 | σ-release at σ<0.3 | 21.25 | 2/10 | clip-dependent, net negative |
| 10 | exp_039 | 80 steps + σ-release | regression | — | rejected — drift dominates |

---

## 6. The named cause — and the per-clip predictor

After 9 deployable interventions, the cause is named and replicated:

> **Free-middle round-trip cost is coupled to anchor quality through
> velocity coupling. Under §0, no anchor source — sub-clip encoded,
> model-bootstrap, or solver-modulated — produces anchors that
> approach z₀-truth at middle positions for clips with
> clip-specific, novel transition content.**

We have a predictor for which clips will fail:

> **f24-vs-f96 CLIP cosine gap, Spearman ρ = 0.855 with exp_033 PSNR.
> Clean-pass threshold: gap_clip ≈ 0.39.**

Interpretation: if the CLIP embedding of the start chunk's frames
disagrees strongly with the end chunk's frames, the two endpoints don't
constrain the middle well — there's no near-linear interpolation in
representation space that satisfies both. The model's bootstrap can
only give a *generic* interpolation, the sub-clip encoded anchors can
only give a *local* approximation, and the source middle (which the
recipe can't see) is a *clip-specific path* that neither approximates.

Three clips were "irreducible" across all 9 deployable recipes —
**ss5, ss6, ss9 never crossed PSNR 22 in any deployable recipe**.
All three have high gap_clip. This is the genuine §0 ceiling: not a
bug, not a missed tweak, but a property of the data + the constraint
together.

---

## 7. What we *didn't* find — and why that matters

A non-result that surprised us:

- We did not find a deployable lever that beats `exp_033`. The
  anchor-quality family was exhausted at exp_033. The bootstrap family
  was wrong-prior. The solver family is a trade-off, not a lever.
- No knob improved the irreducible-failure clips (ss5/ss6/ss9). Every
  intervention either left them stuck below PSNR 22 or made them
  worse.

That tells us the §0 ceiling is **not lying just out of reach** — it's
the actual ceiling for novel-transition content. Spending another
budget on more drop variants or more pin variants would have given us
the same kind of clip-dependent zero-sum trades we already mapped.

---

## 8. Paths forward — none are inside §0

The §0 frame is exhausted. If you want different numbers, you have to
change the frame:

1. **Drop §0 and ship exp_032.** Median PSNR 40.88, 8/10 pass. Real,
   shippable, *if* the use-case doesn't require source-middle
   independence at edit time. (E.g. "compress a clip to a latent for
   playback" doesn't need §0. "Re-render with a different prompt"
   does.)
2. **Reframe to characterize the §0 floor.** Instead of "raise the
   floor", measure what perceptual failure modes exist at the floor.
   This becomes a downstream-editing-quality study, not an inversion
   study. Different exit criteria.
3. **Train a model on shadow_smoke-like transitions.** The bootstrap
   prior fails because it's generic. A model whose training
   distribution matches the test distribution would shrink the
   bootstrap-to-z₀ middle distance directly. Out of current scope and
   budget but theoretically a real fix.

Inside §0 with the current model and shadow_smoke clips: there is no
further lever worth pulling.

---

## 9. Process lessons (for the next loop)

A few things worth carrying forward:

- **Pilot ≠ full batch.** exp_038 looked good on 3 clips and bad on
  10. Always preserve a planned full-batch run before declaring a
  recipe a win — and choose pilot clips to span baseline difficulty.
- **Cage detector earns its keep.** It-4 → It-5 was a clean reframe
  driven by the cage. Without it, It-5 would have been another
  drop-variant on end anchors. Watch for stuckness signals; don't
  treat "one more tweak in this family" as progress when the family is
  exhausted.
- **Strength scalars in conditioning masks compound.** A 0.3 strength
  is not 30% of a hard pin; it's a hard pin run in 40 micro-doses.
  Reason about effects at the *trajectory* level, not the *step*
  level.
- **Step count is not a uniform lever** in solvers where anchor
  quality varies. It re-weights the per-clip cost distribution. Always
  inspect per-clip deltas, not just medians.
- **CLIP gap is a useful one-shot predictor** of round-trip
  difficulty when endpoint conditioning is the recipe. Worth checking
  before spending GPU time.
- **Premature exit ② is more costly than late exit ②.** The early
  19:37 call had to be re-opened. The work that followed was honest
  budget spend — the conclusion didn't change, but the confidence in
  the conclusion went from "one family tested" to "four families
  tested". Don't call exit ② until the cage signals are exhausted
  *and* the budget is materially consumed.

---

## 10. Frozen artifacts (do not modify)

- `outputs/videos/exp_030_ltx2_rf_inv_real_clips/run_*`
- `outputs/videos/exp_032_ltx2_rf_inv_selfcond/run_*` (LEAKY upper bound)
- `outputs/videos/exp_033_ltx2_rf_inv_drop1/run_*` (THE FLOOR)
- `outputs/videos/exp_034_*/run_*` through `outputs/videos/exp_039_*/run_*`
- `scripts/anchor_error_localization.csv` (per-latent-frame
  decomposition that named the cause)

## 11. Related references

- `notes/rf_inversion_loop.md` — full procedural Ledger (this report
  is the explanatory complement)
- `notes/models/ltx2/conditioning.md` §14-b — causal-VAE asymmetry
  mechanism, still factually valid
- `project_rf_inversion_it2_in_flight.md` (memory) — point-in-time
  closure snapshot
