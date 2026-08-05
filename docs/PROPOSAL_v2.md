# Creative Transition Transfer — Project Proposal v2

*2026-08-05 · supersedes the June proposal (`docs/Project proposal.md`) · self-contained — every term is defined where it first appears.*

---

## In one paragraph

**Creative Transition Transfer (CTT):** given one reference video demonstrating a creative transition — a smoke morph, a portal, a whip-pan — re-perform *that* transition between a new pair of shots, with no training for the specific style. The project is organized around three questions: can we **measure** the task, is the **ceiling reachable**, and can one model **transfer**. Where they stand: measurement is solved for our purposes (a calibrated, pre-registered instrument); the ceiling is reached (per-style *specialist* models score ≈100% of ground-truth level); and transfer — a single *generalist* model that reads the operator from the demo — is partially solved: training on our new 56k-pair corpus improves every generalization cell, and naming the effect in words closes most of the remaining gap. That localizes the open problem precisely: **the model can already exploit an operator it is told about in words; it cannot yet extract the same operator from the demo video.** Closing that gap is the current front.

---

## 1 · The task

A *transition* connects the last moments of shot A to the first moments of shot B. Professional VFX transitions are not cuts or crossfades: something happens in the middle — smoke swallows the scene, the world folds into a portal — and that *something* is a reusable style, independent of what A and B contain. We call it the **transition operator**.

The task: **given endpoints (A, B) and one reference video demonstrating an operator, generate the in-between video that connects A to B using that operator.** One model, any operator, no per-style training at test time.

Concretely, the base generator is an endpoint-conditioned video model: it receives prefix frames from A — and, for two-sided transitions, suffix frames from B — plus the reference video in its context, and must synthesize the middle.

## 2 · Why it is hard

Two difficulties — one shallow, one deep.

**Collapse.** Ask an endpoint-conditioned video model to connect two contextually distant shots and it produces a dissolve: the frames drift along a near-straight path in the model's latent space between the endpoints (early on we verified that these generated dissolves are visually close to a literal latent-space crossfade). The model is not wrong — a dissolve *is* the cheapest valid in-between — it is just never creative. Any method must first beat this default.

**Entanglement.** The reference video shows the operator only as applied to *its own* content. To transfer it, a model must separate *what the transition does* from *what these particular clips contain* — keep the operator, discard the content. Nothing in standard video-model training rewards that separation, and (§6) it does not fall out of the model's internal features for free.

## 3 · Scope and assumptions

- **Frozen base model.** LTX-2 19B, an endpoint-conditionable video generator. We train only small adapters (LoRA); the base is never updated.
- **The demo is the only operator channel.** Training captions are deliberately *content-only*: they describe the endpoint shots and mark where the transition goes with a neutral trigger token — never *which* transition. If captions named the effect, the model could route operator identity through text and ignore the demo; the closest prior work does exactly this (§5), and avoiding it is a design principle here — a text route caps the method at effects that already have names.
- **Endpoints are given** as conditioning frames (one- or two-sided).
- **Success is what the instrument says.** Every claim rides on a pre-registered metric with calibrated thresholds (§7.1), not on visual inspection.
- **Per-style training is not the product.** We do train per-style models — but only as a ceiling measurement and as a data engine (§7.3), never as the deliverable.

## 4 · The three questions

- **Q1 — Measurement.** Can we score "the intended transition happened here" reliably enough to steer research? Without this, nothing else is knowable.
- **Q2 — Ceiling.** Can the base generator express these operators at all? If even a model trained on a single style cannot perform it, the task is hopeless.
- **Q3 — Transfer** (the goal). Can one adapter read the operator from the reference video and apply it to new endpoints?

The rest of the document follows this order: what we built for each question (§7) and where each stands (§8). Status up front: **Q1 yes · Q2 yes · Q3 partially.**

## 5 · Closest prior work: refVFX

refVFX (*Tuning-free Visual Effect Transfer across Videos*, arXiv 2601.07833) is the nearest published system: a large LoRA on a 14B first/last-frame video model (Wan 2.1 FLF2V) that transfers a reference effect to a new clip. The essential difference is the routing: refVFX carries effect identity substantially through **text** — per-effect triggers and effect descriptions in the prompt — with the reference video as support. Our task definition forbids that route (§3).

We run refVFX as an external baseline on our own benchmark (via its public community reimplementation), in two configurations: **A**, its intended setup with the effect described in text; **B**, restricted to our content-only text budget. Its released dataset also contributes one training stratum (§7.2). Empirically it behaves as designed: removing the text description costs it ~9 points (§8.3) — its operator identity does ride on text.

## 6 · The first branch, and the pivot

The project's first branch was **training-free**: invert the reference video through the frozen model, extract internal features along the reconstruction path (attention K/V, velocity fields), and re-inject them while generating between new endpoints. It was implemented end-to-end, and it is closed. Three reasons: injection fidelity plateaued well below usable reconstruction (attention-feature injection stalled near its ~9 dB PSNR baseline against a pre-set 14 dB milestone); every new sample needed its own fragile inversion pipeline; and — decisively — operator and content proved inseparable in raw internal features: steering with them dragged the demo's content along with its operator. That is the entanglement problem of §2 in concrete form, and it convinced us the separation must be **learned**. Everything below is the training branch.

## 7 · What we built

### 7.1 The instrument (Q1)

The idea: a transition class has many real examples, so a generation claiming that class should resemble the real examples of the class — in its middle frames, where the transition lives — while genuinely connecting its own endpoints and not merely replaying the demo. Each aspect is one plainly-stated metric:

| | the question it answers | reading |
|---|---|---|
| **M1a** · appearance & dynamics | Does the generated middle look — and evolve — like real examples of the intended transition? | 0–1, higher better · **the headline** |
| M1b · camera motion | Does the camera move the way it moves in the reference? | distance, lower better |
| M1c · object motion | With camera motion removed, do the subjects move like in the reference? | distance, lower better |
| M2a · copy guard | Did the model just replay the demo clip? | flags above a calibrated threshold |
| **M2b** · style margin | Is the output closer to the *intended* style than to any other style in the corpus? | positive = intended style wins |
| M3a · endpoint fidelity | Does the video truly begin (and end) on the given endpoint frames? | similarity, higher better |
| M3b · seam flag | Is there a visible snap where generation meets the given frames? | flags |

Discipline, in one sentence: every threshold is pre-registered before any result exists and calibrated against a case that must fail it; there is no composite score; and the instrument is versioned and certified before any model claim is read off it (current version: v4).

**The headline scale — "% of ceiling."** M1a is reported on an interpretable scale. For each generation: score it against a pool of real clips of the intended class — never including the demo itself or the item's own endpoints, so copying cannot inflate it — giving the **raw** score. Score the real clips of that class against *each other* the same way, giving the **ceiling**: what a perfect generation would get. Report **raw ÷ ceiling** as a percentage. Anchors for intuition: a real clip of the class ≈ 100% · a plain crossfade 48% · a frozen frame 22%.

**Does it measure the concept, not the clip?** The validation we lean on: score every real clip's middle against every *other* same-class clip as its reference, all pairs. Same-class pairs score 0.870 on average versus 0.494 for wrong-class (a large clean gap, d′ = 1.71) — while *swapping which same-class clip serves as the reference* moves the score by only ±0.044 (±0.017 when averaging a pool of ~7). Identifying a clip's class from its score alone succeeds 86% of the time. So M1a responds to the transition concept and is nearly indifferent to the particular demo chosen to represent it — not a strict proof, but a strong signal, and it is what licenses pool scoring and the ceiling definition above. (Each metric also had to earn its place on a label-free retrieval exam over the 223-clip real corpus; the appearance metric's exam accuracy rose 0.67 → 0.81 during that selection.)

### 7.2 The dataset — CTT v2 (v2.1.0, frozen)

One training row = (target clip, reference clip demonstrating the **same** operator) + endpoint conditioning + a content-only caption. Rows are grouped by operator; the reference is always a *different* clip of the same operator group — so the only consistent signal linking reference to target is the operator itself.

| stratum | what it is | clips | operators | rows |
|---|---|---|---|---|
| **S0** · curated real | hand-curated real VFX transitions, 26 classes — the project's ground-truth corpus | 139 | 26 | 385 |
| **S1** · specialist-generated | produced by the 11 specialists (§7.3) applying their operator to unfamiliar endpoints, then hand-screened clip-by-clip (13.5% rejected) | 1,225 | 12 | 3,675 |
| **S2** · procedural | shader transitions (the open *gl-transitions* family) rendered between pairs of real clips — exact operator labels, operator diversity at scale | 15,436 | 1,590 | 46,308 |
| **S4** · external real | real-VFX clips from the refVFX release, 42 effects | 2,000 | 42 | 6,000 |
| | **total** | **18,800** | **1,670** | **56,368** |

152 GB precomputed (latents + text embeddings), endpoint-disjoint from every evaluation set; the sampling mix across strata is a training-time knob, not baked into the data. The design bet: S0/S1/S4 supply *creative* operators, S2 supplies operator *variety* at scale — together forcing the in-context mechanism to bind to the demo rather than memorize a catalogue of styles.

### 7.3 The models — specialists and generalists

The arms, in the order the logic requires:

- A **specialist** knows *one operator*. One LoRA per transition class, trained on that class's real clips (11 exist). A specialist cannot transfer — it is exactly the per-style training the task forbids — but it answers Q2 (*is the operator expressible at the quality we need?*) and doubles as a **data engine**: applying a specialist to content its class never touches manufactures new, clean examples of the operator (stratum S1).
- A **generalist** knows *the mechanism*. One in-context LoRA over all operators: at inference it receives the reference video in context and must read the operator out of it. This is the deliverable of the task (Q3). Two exist — **G1**, trained on the original curated corpus only, and **G2**, trained on CTT v2. A third arm, **G2+text**, additionally names the effect in one clause after the trigger token — deliberately breaking the content-only rule as a *probe*: it measures how much headroom sits in operator-reading itself.
- **Base controls.** The untrained base model given the *same* effect described in words — ⓪ prompt only, ① prompt + endpoint conditioning. Deliberately strong controls: they are told in text what the generalists must read from the demo.
- **External baseline:** refVFX A/B (§5).

All arms generate the same frozen 152-row evaluation grid (× 2 seeds): trained classes on unseen endpoints, endpoints borrowed from other classes, zero-shot classes never trained on, out-of-corpus footage, and control rows — including one whose demo is deliberately *mismatched*, the cell that reveals whether a model treats the demo as an instruction or as decoration.

## 8 · Where we stand

### 8.1 Q1 — Measurement: solved for our purposes

Instrument certified (v4); the reference-swap validation above; copy guard calibrated. One fact from practice worth stating: across all trained arms below, **zero** generations trip the copy guard — no score in this section is earned by replaying the demo.

### 8.2 Q2 — Ceiling: reached

Specialists perform their operator at ground-truth level:

| endpoints given to the specialist | % of ceiling | vs base twin |
|---|---|---|
| endpoints from training (fit anchor) | 100.4% | +39.5 pp |
| **unseen endpoints, own-class content** | **99.7%** | **+40.0 pp** |
| endpoints borrowed from another class | 94.9%* | +39.2 pp |
| out-of-corpus footage (DAVIS) | 63.1%* | +18.9 pp |

*11 specialists pooled; class ceilings for this grid average 0.863 (raw = % × ceiling). Starred cells are content-capped: the endpoints' content can never fully resemble the class, so the absolute level is ranking-only and the honest claim is the gap over the base twin. Anchors: crossfade 48% · prompt-only base 67%.*

Reading: with training, the base model performs a creative operator on new content at essentially ground-truth level — **the ceiling exists and is reachable**, and whatever limits the generalists, it is not the base model's capacity. The drop on far-out-of-corpus footage is real and expected: the operator is being asked to bind to radically unfamiliar content.

### 8.3 Q3 — Transfer: real progress, honest gap

Same instrument, same grid, all arms (1,842 scored items per arm, one machine). Same-class cells are the fair headline; cross/foreign cells are content-capped (ranking-only); the style margin is M2b — positive means the intended style beats every other.

| arm | same-class cells | cross/foreign cells* | style margin |
|---|---|---|---|
| base ⓪ · prompt only (effect described) | 80.0% | 81.4% | −0.000 |
| base ① · prompt + endpoints (effect described) | 84.9% | 83.0% | +0.001 |
| **G1** · generalist, demo only | 83.1% | 62.0% | −0.029 |
| **G2** · generalist on CTT v2, demo only | 82.5% | 66.7% | −0.011 |
| **G2+text** · demo + effect described | **91.3%** | **86.4%** | **+0.044** |
| refVFX A · its own config (text) | 42.4% | 41.3% | −0.032 |
| refVFX B · our text budget | 33.0% | 26.7% | −0.064 |

*Shared same-class ceiling 0.872 (raw = % × ceiling). refVFX rows are indicative rather than apples-to-apples: different base model, 33-frame native output, community reimplementation.*

Four findings, in order:

1. **Generalists do read the demo.** On the control rows whose demo is deliberately mismatched, the generalists follow the demo at 68.8% / 69.2% (G1/G2) — clearly above the base model's 56.3% / 61.0% when the same mismatched instruction is given in words. The in-context mechanism works *as a mechanism*.
2. **G1 is not good enough.** Trained on the 26 curated classes alone, it holds up on familiar territory (83.1%) but drops hard exactly where transfer is tested — cross-content and foreign cells (62.0%) — and its style margin is negative: on average some *other* class matches its output better than the intended one.
3. **CTT-v2 training moves everything the right way — modestly.** G2 improves the style margin on **all four** cross/foreign generalization cells (mean +0.04, best cell +0.066) against a pre-committed target of +0.10 — recorded as directional progress, not success. (Kept honest: corpus, adapter capacity and schedule changed together, so this is a systems comparison, not a clean ablation.)
4. **Naming the effect closes most of the remaining gap.** One clause of text on top of G2 gives the best arm across the board (91.3% / 86.4%, margin +0.044), with the largest gains precisely on the hardest cells (zero-shot classes, foreign footage). Symmetrically, taking text *away* from refVFX costs it ~9 pp.

Finding 4 is the diagnosis the project now runs on: **the generator can already *use* an operator specified in words; it cannot yet *extract* an equally usable operator from the demo video.** The G2 → G2+text gap is, precisely, the unsolved part of Q3 — and the specialists (Q2) prove it is a *representation* problem (how the demo is encoded and delivered to the generator), not a capacity problem, and Q1 ensures it is not a measurement problem.

## 9 · Next steps

1. **Make the demo channel as instructive as the text channel.** In flight: redesigning how the reference is represented to the generator — compressing the demo into a compact operator code the generator can natively read. This is the direct attack on the §8.3 diagnosis.
2. **Close the pre-committed margin target** (+0.10 on the cross/foreign claim cells) with the demo-only generalist, and re-measure seed noise under the current instrument so every reported delta carries a minimum-detectable-effect bar.
3. **Consolidated report** of the full ladder — mid-August.
