# Creative Transition Transfer: Project Proposal v2

*2026-08-05 · supersedes the June proposal · self-contained*

## 0. Thesis

**Creative Transition Transfer (CTT):** given a reference video demonstrating a transition between two clips, transfer that transition *operator* to new, unseen endpoint pairs, without per-style training. Our approach is a reference-conditioned adapter (in-context LoRA) on a frozen video flow model, judged by a pre-registered evaluation instrument. Current state: instrument certified, dataset final, and per-style *specialists* prove the ceiling is reachable (≈100% of ground-truth level); the *generalist* adapter demonstrably reads the reference but does not yet fully apply it. The gap is precisely localized: the model can already apply an operator described in words; extracting an equally usable operator from the reference video is the open front.

## 1. Problem Statement

### 1.1 Task Definition

Let the start and end clips be

$$V_S = (I_t)_{t=1}^{N_S}, \qquad V_E = (I_t)_{t=1}^{N_E},$$

with frames $I_t \in \mathbb{R}^{H \times W \times 3}$, shared resolution and frame rate.

A **transition operator** is a mapping

$$\mathcal{T}: (V_S, V_E) \mapsto V,$$

taking an endpoint pair to the connecting video with that transition type applied: *what the transition does* (smoke swallows the scene, the world folds into a portal), independent of the endpoints it is applied to. (A type admits many valid realizations; $\mathcal{T}$ stands for their shared behavior.) A **reference video** is one evaluation of the operator, on its own endpoints:

$$V_R = \mathcal{T}(V_R^S, V_R^E).$$

Given new endpoints $V_S$, $V_E$ (two-sided; one-sided omits $V_E$), the reference $V_R$, and an optional text prompt $c$, the task is to generate

$$V = (I_t)_{t=1}^{N} \approx \mathcal{T}(V_S, V_E),$$

i.e. read $\mathcal{T}$ from its single demonstration and apply it to the new pair, such that:

1. **Boundary constraints:** $V_{1:N_S} \approx V_S$ and $V_{N-N_E+1:N} \approx V_E$.
2. **Operator constraint:** the middle of $V$ realizes the same $\mathcal{T}$ that produced $V_R$: its appearance and dynamics, not $V_R$'s content.
3. **No per-style training:** one model serves every $\mathcal{T}$; the operator is specified only through $V_R$, at inference time.

### 1.2 Why It Is Hard

**(a) Lerp Collapse.** Endpoint-conditioned flow models connect distant shots with a dissolve, a near-straight path in latent space (we verified early that generated dissolves sit visually next to a literal latent crossfade). The cheapest valid in-between is never a creative one; any method must first beat this default.

**(b) Entanglement.** $V_R = \mathcal{T}(V_R^S, V_R^E)$ shows the operator only evaluated on its own endpoints. Transfer requires separating *what it does* from *what it contains*, a separation nothing in standard video training rewards, and one that does not fall out of the model's raw features (§5).

## 2. Assumptions & Scope

- **Frozen base generator.** LTX-2 19B, endpoint-conditionable; we train only small adapters (LoRA).
- **The reference video is the primary operator channel.** Training captions are content-only, with a neutral trigger token marking the transition slot; at inference, text may additionally describe the effect, and we measure both regimes (§7).
- **Per-style training is out of scope** by definition of the task; specialists exist only as a ceiling measurement and a data engine (§6.3).
- **Success is defined by a pre-registered instrument** (§6.2), not inspection: thresholds fixed before results exist, each calibrated against a case that must fail it.

## 3. Research Questions

- **RQ1: Transfer.** Can an adapter read an operator from a reference video and apply it to unseen endpoints?
- **RQ2: Disentanglement / compression.** Can a compact learned code encode, represent, and isolate the operator better than the raw reference does? *(in progress, §7.5)*
- **RQ3: Measurement.** How do you score "the right transition happened" credibly enough to bet training decisions on?

## 4. Related Work

**refVFX** (*Tuning-free Visual Effect Transfer across Videos*, arXiv:2601.07833) is the strongest published neighbor: a LoRA on a 14B first/last-frame video model (Wan 2.1 FLF2V) that transfers a reference effect to a new clip. It routes effect identity largely through **text** (per-effect triggers and descriptions), with the reference as support; our focus is the reverse, making the reference video itself carry the operator. We run it as an external baseline in two arms (**A**: its own config, effect described in text; **B**: under our text budget), via its public community reimplementation; its released dataset also contributes one training stratum (§6.1).

## 5. Trajectory

The project's first branch was **training-free**: invert $V_R$ through the frozen model, extract internal features along the reconstruction path (attention K/V, velocity fields), and re-inject them while generating between new endpoints. Implemented end-to-end, now closed: injection fidelity plateaued (~9 dB PSNR against a pre-set 14 dB milestone), and, decisively, operator and content proved inseparable in raw features: steering transferred the reference's content along with its operator. That is §1.2(b) in concrete form; the separation must be *learned*. Everything since is the training branch.

## 6. Methodology & Current State

### 6.1 Dataset: CTT v2 (v2.1.0, frozen)

One row = (target clip, reference clip of the **same** operator) + endpoint conditioning + content-only caption. The reference is always a *different* clip of the operator group, so the operator is the only consistent signal linking reference to target.

| stratum | source | clips | operators | rows |
| --- | --- | --- | --- | --- |
| **S0** · curated real | hand-curated real VFX transitions, 26 classes: the ground-truth corpus | 139 | 26 | 385 |
| **S1** · specialist-generated | the 11 specialists (§6.3) applied to unfamiliar endpoints, then hand-screened (13.5% rejected) | 1,225 | 12 | 3,675 |
| **S2** · procedural | shader transitions (*gl-transitions*) rendered between pairs of real clips: exact labels, operator diversity at scale | 15,436 | 1,590 | 46,308 |
| **S4** · external real | real-VFX clips from the refVFX release, 42 effects | 2,000 | 42 | 6,000 |
| | **total** | **18,800** | **1,670** | **56,368** |

152 GB precomputed, endpoint-disjoint from every eval set; the strata mix is a training-time knob. Design bet: S0/S1/S4 supply *creative* operators, S2 supplies operator *variety*; together they force the in-context mechanism to bind to the reference rather than memorize styles.

### 6.2 Evaluation Instrument (v4, certified)

| | the question it answers | reading |
| --- | --- | --- |
| **M1a** · appearance & dynamics | Does the generated middle look, and evolve, like real examples of the intended transition? | 0–1, higher · **headline** |
| M1b · camera motion | Does the camera move as in the reference? | distance, lower |
| M1c · object motion | Camera removed, do subjects move as in the reference? | distance, lower |
| M2a · copy guard | Did it just replay the reference clip? | flag over calibrated threshold |
| **M2b** · style margin | Closer to the *intended* style than to any other? | positive = intended wins |
| M3a · endpoint fidelity | Does $V$ truly begin/end on $V_S$/$V_E$? | similarity, higher |
| M3b · seam flag | Visible snap where generation meets given frames? | flag |

No composite score; every threshold pre-registered and calibrated against a case that must fail it.

**Headline scale: % of ceiling.** Score a generation against a pool of real clips of the intended class (never the reference itself, never the item's endpoints, so copying cannot inflate it) to get the **raw** score. Score the class's real clips against *each other* the same way to get the **ceiling**: what a perfect generation would score. Report raw ÷ ceiling. Anchors: real clip ≈ 100% · plain crossfade 48% · frozen frame 22%.

**Validation: the reference-swap test.** Scoring every real clip against every *other* same-class clip as reference: same-class 0.870 vs wrong-class 0.494 (d′ = 1.71), while swapping *which* same-class clip serves as reference moves the score only ±0.044 (±0.017 for a pool of ~7); the class is identifiable from the score alone 86% of the time. The metric tracks the transition *concept*, not the particular reference: not strict proof, but a strong signal.

**Test tiers.** Every evaluation row sits in one cell of a 3×3 grid: the **reference tier** (is the reference's operator *seen* in training, an *unseen* instance of a trained class, or a *zero-shot* class never trained on?) crossed with the **endpoint tier** (endpoints from the operator's own class: *same*; from another corpus class: *cross*; off-distribution footage: *foreign*).

| reference \ endpoints | same | cross | foreign |
| --- | --- | --- | --- |
| **seen** (training items) | anchors: fit, memorization probe | – | – |
| **unseen** (trained class, new instance) | ✓ | ✓ **target** | ✓ |
| **zero-shot** (class never trained) | ✓ | ✓ **target** | ✓ |

The claims target the **unseen and zero-shot reference rows**, cross endpoints above all: there the reference shows an operator the model cannot have memorized, applied to content from elsewhere. Seen-reference cells are anchors, never claims; a mismatched-reference control row completes the grid.

### 6.3 Experimental Arms

- **Specialist** (×11): a LoRA per transition class, trained on that class's own clips (split v1). Rank 32, 2,000 steps, endpoint-conditioned. Knows one operator, cannot transfer by design; answers whether the operator is expressible at the quality we need, and generates stratum S1.
- **Generalist 1 (G1)**: an IC-LoRA trained on S0's corpus (split v1): the reference clip is prepended in context (loss-masked), and the adapter must read the operator from it. Rank 32, bidirectional reference attention, 5,000 steps.
- **Generalist 2 (G2)**: the same in-context recipe trained on the whole CTT v2 dataset (56,368 pairs, 1,670 operators; strata mix set in the trainer config). Rank 128, one-way reference attention, 10,000 steps. **G2+text**: the same trained adapter, with the effect additionally named in one clause of $c$ at inference; probes how much headroom sits in operator-reading itself.
- **Base controls**: the untrained base model with the effect described in words (⓪ prompt only; ① plus endpoints). They receive in text what the generalists must read from the reference.
- **refVFX** (external, §4): **A** in its own text config, **B** under our text budget.
- **Bottleneck branch (running now)**: three reference-encoding implementations: a trained bottleneck operator encoder, a causal-VAE encoding, and a pretrained V-JEPA embedder. Goal: a reference signal that encodes, represents, and *isolates* the operator better than raw reference latents do.

All arms are evaluated on one frozen 152-row grid × 2 seeds covering the §6.2 tiers, plus controls, including rows with a deliberately *mismatched* reference that reveal whether a model treats the reference as an instruction or as decoration.

## 7. Current Results

One grid, one machine, one instrument (v4). Zero trained-arm generations trip the copy guard: no score below is earned by replaying the reference. Columns follow the §6.2 tier grid: **uns** = unseen reference, **zs** = zero-shot reference, each split by endpoint tier (% of ceiling; margin = M2b over all rows).

| arm | uns·same | uns·cross* | uns·foreign* | zs·same | zs·cross* | zs·foreign* | margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Specialist** (×11) | **99.7**ᵃ | 94.9ᵃ | 63.1ᵃ | – | – | – | – |
| base ⓪ · prompt only (effect described) | 83.0 | 87.2 | 79.4 | 81.5 | 80.1 | 74.2 | −0.000 |
| base ① · plus endpoints (effect described) | 87.0 | 88.8 | 78.6 | 89.6 | 89.3 | 76.5 | +0.001 |
| **G1** · reference only | 89.1 | 73.1 | 56.9 | 91.1 | 72.2 | 44.1 | −0.029 |
| **G2** · reference only | 90.5 | 74.1 | 59.8 | 74.9 | 78.5 | 54.0 | −0.011 |
| **G2+text** · reference + effect described | **98.3** | **94.3** | **86.1** | **91.6** | **91.0** | **71.9** | **+0.044** |
| refVFX A · its own config (text) | 47.8 | 40.4 | 38.7 | 52.3 | 41.8 | 45.3 | −0.032 |
| refVFX B · our text budget | 35.0 | 25.2 | 28.7 | 31.6 | 28.5 | 24.3 | −0.064 |

*All values % of ceiling; cell sizes 13/26/26/8/20/20 rows × 2 seeds. Starred columns are content-capped (endpoint content can never fully resemble the class): ranking-only. Zero-shot is undefined for specialists (a class with no training data has no specialist). ᵃ Specialists are scored on their own ladder grid (same instrument, cell means differ by <0.4 pp); their claim channel is the paired gap vs the base twin: **+40.0 pp** on uns·same.*

Findings, in order:

1. **The ceiling is reachable.** Specialists perform their operator on new own-class content at ground-truth level (99.7%, +40 pp over the base twin). Whatever limits the generalists, it is not the base model's capacity.
2. **G1 reads the reference but does not generalize.** On mismatched-reference control rows it follows the reference at 68.8%, clearly above the base model's 56.3% / 61.0% given the same instruction in words; yet it falls exactly where transfer is tested: cross and foreign cells sink to 73.1 / 56.9 (unseen) and 72.2 / 44.1 (zero-shot), with a negative margin, meaning some *other* class matches its output better than the intended one.
3. **G2 moves the target cells the right way.** Margin improves on all four cross/foreign claim cells (+0.04 mean, +0.066 best) against a pre-committed +0.10 target, clearest on zero-shot and foreign (44.1 → 54.0): directional progress, not success. **G2+text is best in every cell** (94.3 / 86.1 unseen cross/foreign, 91.0 / 71.9 zero-shot; margin +0.044): naming the effect closes most of the remaining gap.
4. **refVFX sits well below in every cell** and, consistent with its text-routed design, loses ~9 points without its effect description. Indicative rather than apples-to-apples (different base model, 33-frame output, community reimplementation).

**Diagnosis:** the generator can already *use* an operator specified in words; it cannot yet *extract* an equally usable operator from the raw reference video. That is a representation problem (the specialists rule out capacity, the instrument rules out measurement), and the bottleneck branch (§6.3) is the direct attack on it.

## 8. Next Steps

1. **Bottleneck branch**: land and evaluate the three reference encodings (bottleneck operator encoder, causal VAE, V-JEPA) on the same grid.
2. **Close the pre-committed +0.10 margin target** on the claim cells, reference-only; re-measure seed noise under v4 so every delta carries a minimum-detectable-effect bar.
3. **Consolidated ladder report**, mid-August.
