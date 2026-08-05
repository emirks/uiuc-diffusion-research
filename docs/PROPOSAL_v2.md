# Creative Transition Transfer: Project Proposal v2

*2026-08-05 · supersedes the June proposal · self-contained*

## 0. Thesis

**Creative Transition Transfer (CTT):** given a reference video demonstrating a transition between two clips, transfer that transition *operator* to new, unseen endpoint pairs, without per-style training. Our approach is a reference-conditioned adapter (in-context LoRA) on a frozen video flow model, judged by a pre-registered evaluation instrument. Current state: instrument certified, dataset final, and per-style *specialists* prove the ceiling is reachable (≈100% of ground-truth level); the *generalist* adapter demonstrably reads the reference but does not yet fully apply it. The gap is precisely localized: the model can already apply an operator described in words; extracting an equally usable operator from the reference video is the open front.

## 1. Problem Statement

### 1.1 Task Definition

Let the start and end clips be

$$V_S = (I_t)_{t=1}^{N_S}, \qquad V_E = (I_t)_{t=1}^{N_E},$$

with frames $I_t \in \mathbb{R}^{H \times W \times 3}$, shared resolution and frame rate. A **transition operator** $\mathcal{T}$ is the reusable part of a transition, *what it does* (smoke swallows the scene, the world folds into a portal), independent of the clips it connects. A **reference video** $V_R$ is one demonstration of $\mathcal{T}$, applied to $V_R$'s own endpoints.

Given $V_S$, $V_E$ (two-sided; one-sided omits $V_E$), the reference $V_R$, and an optional text prompt $c$, generate $V = (I_t)_{t=1}^{N}$ such that:

1. **Boundary constraints:** $V_{1:N_S} \approx V_S$ and $V_{N-N_E+1:N} \approx V_E$.
2. **Operator constraint:** the middle of $V$ realizes the operator demonstrated by $V_R$: its appearance and dynamics, not its content.
3. **No per-style training:** one model serves every operator; $\mathcal{T}$ is specified through $V_R$ at inference time.

### 1.2 Why It Is Hard

**(a) Lerp Collapse.** Endpoint-conditioned flow models connect distant shots with a dissolve, a near-straight path in latent space (we verified early that generated dissolves sit visually next to a literal latent crossfade). The cheapest valid in-between is never a creative one; any method must first beat this default.

**(b) Entanglement.** $V_R$ shows the operator only as applied to its own content. Transfer requires separating *what it does* from *what it contains*, a separation nothing in standard video training rewards, and one that does not fall out of the model's raw features (§5).

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

### 6.3 Model Arms

- **Specialist**: knows *one operator*. One LoRA per class, trained on that class's real clips (11 exist). Cannot transfer by design; it answers whether the operator is expressible at the quality we need, and doubles as the data engine behind S1.
- **Generalist**: knows *the mechanism*. One in-context LoRA over all operators; at inference it must read the operator from the reference. The deliverable. **G1** is trained only on S0's corpus; **G2** on the whole CTT v2 dataset; **G2+text** is G2 with the effect additionally named in one clause of $c$, a probe of how much headroom sits in operator-reading itself.
- **Base controls**: the untrained base model with the effect described in words (⓪ prompt only; ① plus endpoints), strong controls that receive in text what the generalists must read from the reference.
- **External**: refVFX A/B (§4).

All arms generate one frozen 152-row grid × 2 seeds: unseen endpoints, cross-class endpoints, zero-shot classes, out-of-corpus footage, plus controls, including rows with a deliberately *mismatched* reference that reveal whether a model treats the reference as an instruction or as decoration.

## 7. Cornerstone Experiments

One grid, one machine, one instrument (v4). Across all trained arms, zero generations trip the copy guard: no score below is earned by replaying the reference.

### 7.1 Specialists: the ceiling is reachable

| endpoints given to the specialist | % of ceiling | vs base twin |
| --- | --- | --- |
| from training (fit anchor) | 100.4% | +39.5 pp |
| **unseen, own-class content** | **99.7%** | **+40.0 pp** |
| borrowed from another class | 94.9%* | +39.2 pp |
| out-of-corpus footage (DAVIS) | 63.1%* | +18.9 pp |

*Starred cells are content-capped (endpoint content can never fully resemble the class): the level is ranking-only, the claim is the gap vs the base twin.*

With per-style training, the base model performs a creative operator on new content at ground-truth level. Whatever limits the generalists below, it is not the base model's capacity.

### 7.2 Baseline generalist: G1, trained only on S0

The remaining cornerstones read off one comparison table:

| arm | same-class cells | cross/foreign cells* | style margin |
| --- | --- | --- | --- |
| base ⓪ · prompt only (effect described) | 80.0% | 81.4% | −0.000 |
| base ① · plus endpoints (effect described) | 84.9% | 83.0% | +0.001 |
| **G1** · reference only | 83.1% | 62.0% | −0.029 |
| **G2** · reference only, CTT-v2-trained | 82.5% | 66.7% | −0.011 |
| **G2+text** · reference + effect described | **91.3%** | **86.4%** | **+0.044** |
| refVFX A · its own config (text) | 42.4% | 41.3% | −0.032 |
| refVFX B · our text budget | 33.0% | 26.7% | −0.064 |

*152 rows × 2 seeds per arm; shared same-class ceiling 0.872; starred column is content-capped, ranking-only.*

G1 holds up on familiar territory (83.1%) but falls exactly where transfer is tested: 62.0% on cross/foreign cells, with a negative style margin, meaning some *other* class matches its output better than the intended one. It does read the reference: on mismatched-reference control rows it follows the reference at 68.8%, clearly above the base model's 56.3% / 61.0% when the same instruction is given in words. The in-context mechanism works; the generalization does not.

### 7.3 CTT v2 generalist: G2, trained on the whole dataset

G2 moves all four generalization cells in the right direction (style margin +0.04 mean, +0.066 best) against a pre-committed +0.10 target: directional progress, not success. (Corpus, capacity and schedule changed together, so this is a systems comparison, not an ablation.) The **G2+text** probe is the sharpest datum: naming the effect in one clause makes it the best arm everywhere (91.3% / 86.4%, margin +0.044), with the largest gains on the hardest cells (zero-shot, foreign). **Diagnosis:** the generator can already *use* an operator specified in words; it cannot yet *extract* an equally usable operator from the raw reference video. That is a representation problem: the specialists rule out capacity, the instrument rules out measurement.

### 7.4 External baseline: refVFX

Well below all our arms on this benchmark (42.4% / 33.0% same-class), and consistent with its text-routed design: removing the effect description costs it ~9 points. Indicative rather than apples-to-apples: different base model, 33-frame native output, community reimplementation.

### 7.5 In progress: bottleneck reference encoding

Experimentation continues with bottleneck implementations. Following the §7.3 diagnosis, the current goal is a better reference signal: an encoding that represents and *isolates* the transition operator, instead of handing the generator the raw reference video.

## 8. Next Steps

1. **Bottleneck reference encoding**: a compact operator code the generator can natively read (in flight).
2. **Close the pre-committed +0.10 margin target** on the claim cells, reference-only; re-measure seed noise under v4 so every delta carries a minimum-detectable-effect bar.
3. **Consolidated ladder report**, mid-August.
