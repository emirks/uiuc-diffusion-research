# ladder2 — full campaign report

**Run:** 2026-07-22 evening → 2026-07-23 midday, UIUC campus cluster.
**Mode:** `/advised` — operator (Opus) executes and verifies; `fable-advisor` owns every judgment call.
**Single source of truth:** `experiments/ladder2/registry.jsonl`. Dossier: `$LAB/misc/ladder2_redesign/DOSSIER.md`. Code: branch `ladder2`.

**Completion:** training 12/12 · generation 888/888 · scoring complete (12,500 donor-pool rows + 2,211 recipient-pool rows + 420 dominance rows). **No kill rule (K0–K3) triggered on any model at any point.**

---

## 1. What the campaign asks

Does a transition learned from a corpus *transfer* — to new content, to a new demo of a known transition, and to transitions never trained on? And how much of that is capability versus lookup?

Two axes:

- **Reference novelty** — `seen` (exact demo trained) → `unseen` (new demo of a trained class) → `zero-shot` (held-out class). Specialists have no reference axis; their transition is in the weights.
- **Content** — `same` / `cross` / `foreign` (DAVIS, off-distribution real footage).

Endpoints are always untrained content, strictly sidedness-matched, except two deliberate fit anchors.

**Four tiers, all compared on byte-identical inputs** (keyed by `input_key` = endpoint + prompt + sidedness + reference):

| tier | gets |
|---|---|
| `text_floor` | prompt only |
| `base` | prompt + conditioning (+ the same reference where its twin has one), no adapter |
| specialist ×11 | + transition baked into weights |
| `ic_gen` | + transition supplied in-context as a demo |

---

## 2. The one thing that changed vs every prior ladder

Every previous run prompted with the **full caption**, which described the outcome:

> `"ICTRANS <S1>. The scene transforms into <S2>."`

That leaks the answer. ladder2 renders leak-free prompts where a neutral token holds the transition slot:

> one-sided `"{S1}. sksz."` · two-sided `"{S1}. sksz. {S2}."`

The outcome half is dropped entirely. Training captions come from the **same** `render_prompt()` call as the registry rows, so train == inference by construction.

**Gates passed before training:**

| gate | result |
|---|---|
| token `sksz` inert on base | effect/noise **0.21** (bar ≤1.0) |
| causal-VAE suffix bleed fix | rel-L2 **0.284**, reproducing exp_073's 0.280 |
| base accepts *and uses* an IC reference | **1.48×** seed noise |
| root assembly | 12 roots, equal source counts, 0 path mismatches |
| no embedding reuse | rel-L2 **0.966** vs the old leaky embeddings |
| leak audit | 0 leaks / 83 endpoints (2 promoted clips audited separately) |

**The new-variable gate passed at step 250** — 4× earlier than the abort rule. Identical prompt, prefix and seed: the untrained base shows no effect; after 250 LoRA steps the class effect is clearly present. The neutral token carried the trigger.

---

## 3. Results — donor-pool table (all 14 cells)

Pool-% = generation scored against every same-class clip of the **donor** class (copy-guarded), ÷ that class's GT ceiling. `%_same` is headline-eligible; `%_proxy` is content-capped and ranking-only.

| cell | n | %type | level | Δpp vs base | donors + | verdict |
|---|---|---|---|---|---|---|
| **SP-same** | 22 | same | **99.9%** | **+40.2** | **11/11** | **CLAIM PASSES** |
| **SP-cross** | 22 | proxy | (94.8%) | **+39.1** | **11/11** | **CLAIM PASSES** |
| SP-fit | 11 | same | 100.5% | +39.6 | 11/11 | sanity anchor |
| SP-foreign | 22 | proxy | (63.4%) | +19.3 | 8/11 | weak (below 9/11 bar) |
| text_floor | 12 | same | 67.4% | — | — | leak-proof floor |
| G-fit | 13 | same | 86.3% | −12.0 | 4/13 | ⚠ invalid |
| G-memo-probe | 13 | same | 83.1% | −15.3 | 1/13 | ⚠ invalid |
| G-unseen-same | 13 | same | 88.7% | −11.2 | 2/13 | ⚠ invalid |
| G-unseen-cross | 26 | proxy | (72.9%) | −27.2 | 0/13 | ⚠ invalid |
| G-zs-same | 8 | same | 90.8% | −14.1 | 1/8 | ⚠ invalid |
| G-zs-cross | 20 | proxy | (72.8%) | −30.3 | 0/10 | ⚠ invalid |
| G-ref-control | 13 | same | 69.0% | +4.1 | 10/13 | ⚠ invalid |
| G-unseen-foreign | 26 | proxy | (56.8%) | −42.1 | 0/13 | ⚠ invalid |
| G-zs-foreign | 20 | proxy | (44.3%) | −55.0 | 0/10 | ⚠ invalid |

**Specialist reading:** a specialist reaches its class ceiling on unseen endpoints of its own class (99.9%) and holds ~95% transferring to a different class's endpoints — both ~+40pp over base on identical inputs, 11/11 donor classes. SP-fit ≈ SP-same means no train-vs-test endpoint gap. DAVIS is positive on average (+19.3pp) but only 8/11 donors, so it fails its sign-test bar and is reported weak.

**Every `G-*` row is marked invalid** — see §4. Those numbers are retained, never deleted, because they were pre-registered.

---

## 4. Why the generalist rows are invalid — the copy confound

The generalist appeared catastrophic (−11 to −55pp, near-zero donors positive). Before reporting that, one pair was inspected directly (held-out donor `cotton_cloud`, demo `cotton_cloud_1`, recipient endpoint `animalization_0`):

- **`ic_gen` did the task**: kept the actual endpoint (woman, red bomber jacket, blue backdrop) and bloomed pink cotton-cloud material around her — donor *manner* on recipient *content*.
- **`base`, same input, no adapter, abandoned the endpoint** and reproduced the demo's own scene (a man on a couch) almost verbatim.

Since pool-% measures resemblance to the donor class, **base is rewarded for copying the demo and `ic_gen` is punished for honouring the endpoint.**

### Pass A — clip-level dominance (420 rows, both arms, every reference-bearing cell)

`ep_align` / `ref_align` = mean over generation *middle* frames of the best match to the endpoint clip / to the reference's non-core frames. Mean-of-best-match, not max-of-max — which is exactly why the harness's absolute `near_copy` flag (τ=0.858, calibrated for verbatim frame copies) never fired at cos ≈ 0.5.

| cell | arm | n | ep_align | ref_align | ref_dominated |
|---|---|---|---|---|---|
| G-fit | base | 26 | 0.292 | 0.838 | **96%** |
| G-fit | ic_gen | 22 | 0.662 | 0.212 | 9% |
| G-memo-probe | base | 26 | 0.321 | 0.864 | **100%** |
| G-memo-probe | ic_gen | 26 | 0.837 | 0.241 | 0% |
| G-ref-control | base | 26 | 0.143 | 0.900 | **100%** |
| G-ref-control | ic_gen | 26 | 0.786 | 0.160 | 0% |
| G-unseen-cross | base | 52 | 0.141 | 0.866 | **98%** |
| G-unseen-cross | ic_gen | 52 | 0.762 | 0.156 | 2% |
| G-unseen-same | base | 26 | 0.260 | 0.852 | **100%** |
| G-unseen-same | ic_gen | 26 | 0.819 | 0.207 | 0% |
| G-zs-cross | base | 40 | 0.171 | 0.857 | **100%** |
| G-zs-cross | ic_gen | 40 | 0.763 | 0.167 | 0% |
| G-zs-same | base | 16 | 0.408 | 0.846 | **88%** |
| G-zs-same | ic_gen | 16 | 0.773 | 0.243 | 12% |

A complete inversion, systematic across 7 cells: base's middle frames are made of the **demo**; `ic_gen`'s are made of the **endpoint**.

**Specialists are structurally immune** — verified 0 of 77 specialist base twins carry a reference, so there is nothing for them to copy. That is why their margins are clean.

---

## 5. Amendment-1 — the corrected, claim-bearing readout

Formulas, thresholds and headline wording were **locked in the dossier before any recipient-pool score existed**.

- **T** (donor manner arrived) = clip01( (D% − D_ep%) / (D_ref% − D_ep%) )
- **C** (endpoint content kept) = clip01( (R% − R_ref%) / (R_ep% − R_ref%) )
- **TI = min(T, C)** — min, not mean: transfer is a conjunction, and averaging lets copying buy back score
- 2×2 at (0.5, 0.5); anchor denominators < 5pp → `anchor_degenerate`, excluded

| cell | arm | n | T | C | TI | quadrants |
|---|---|---|---|---|---|---|
| G-unseen-cross | **ic_gen** | 25 | 0.39 | 0.33 | **0.22** | mush 11, ref-won 7, endpoint-won 5, transfer 2 |
| G-unseen-cross | base | 25 | 0.85 | 0.20 | 0.19 | **ref-won 21**, transfer 2, mush 2 |
| | | | | | **ΔTI +3.9pp** | **donors positive 9/13** |
| G-zs-cross | **ic_gen** | 20 | 0.36 | 0.29 | **0.16** | mush 9, ref-won 6, endpoint-won 4, transfer 1 |
| G-zs-cross | base | 20 | 0.90 | 0.26 | 0.24 | **ref-won 18**, mush 1, transfer 1 |
| | | | | | **ΔTI −8.3pp** | **donors positive 1/10** |

**The result, both directions:**

> In-context transfer **works within the trained transition vocabulary** — with a new demo of a class the model trained on, the IC-LoRA beats its base twin (+3.9pp, 9/13 donors). It **does not generalise to genuinely novel transitions** at this budget — with a held-out class it loses (−8.3pp, 1/10 donors).

Both verdicts stand. The instrument was fixed before the numbers existed, precisely so the negative one could not be explained away.

**Caveat on magnitude:** `ic_gen`'s absolute TI is low everywhere (0.16–0.22), with most items in "mush" — neither strongly donor-flavoured nor strongly endpoint-preserving. Base's high T (0.85–0.90) is bought entirely by copying.

---

## 6. The mechanistic finding (owner-gated — proposed, not written to FINDINGS.md)

> **Without the adapter, LTX-2 treats an in-context clip as *content to continue*. The IC-LoRA converts it into *an instruction to imitate*.**

Quantified on byte-identical inputs: applying the adapter moves `ref_align` **0.86 → 0.19** and `ep_align` **0.22 → 0.78**, across 420 rows and 7 cells (base 88–100% reference-dominated vs `ic_gen` 0–12%).

The advisor called this "arguably the campaign's best" finding. It is a statement about what the adapter *does*, not about how well it scores.

---

## 7. Bugs found and fixed during the run

Each was caught before it corrupted a result.

| # | bug | consequence if missed |
|---|---|---|
| 1 | clip→class derived by string-splitting | `action_run_setonfire_6` → class `run_set_on_fire`, `flame_transition_0` → `flame`; silently mislabelled rows |
| 2 | `item_id` collision in text_floor | seatbelt #1 caught it at build time |
| 3 | train-band endpoint rule was per-root, needed per-**arm** | held-out classes' train clips are untrained content for every arm |
| 4 | `process_captions.py` defaults `media_column` to `media_path` | chained prepare job would have died |
| 5 | YAML flow-style comma truncated a DAVIS caption mid-sentence | silent prompt corruption |
| 6 | `--export` is comma-separated → `CELLS` list truncated to its first element | base generation silently covered 11 of 50 rows |
| 7 | **`no_resume:false` does not enable resume** — `load_checkpoint` must point at the ckpt dir | 5 running jobs would have restarted from step 0 |
| 8 | that fix then crashed first runs (`_find_checkpoint` *raises* on a missing path) | `ic_gen` failed 7 min in; fixed by pre-creating the dir |
| 9 | scoring label reused per pass → **overwrote** prior `items.jsonl` | destroyed ~3000 scored rows (recovered via incremental re-plan) |
| 10 | `set -eo pipefail` before `source ~/.bashrc` | 4 scoring chunks died in 7 s with an empty log |
| 11 | stale `eval_c*.json` chunks re-scored | wasted GPU on already-scored rows |
| 12 | `car-turn` in the DAVIS roster has **no subject** in the portrait crop | it's filmed *from* the car; replaced by `hike` |
| 13 | smoke assert judged sidedness per root | would have killed `ic_gen` at startup (its tree mixes both) |

---

## 8. Process notes

- **Hardest-first generation** (owner call): zero-shot + reference-control generated before the easier cells, so the most valuable results landed first.
- **DAVIS expanded** from a 16-item token gate to 68 items (owner call). This *changed the conclusion*: at n=5 SP-foreign read 4/5 donors positive; at full n=22 it is **8/11 — below bar**. The gate-sized sample would have been reported as a clean positive.
- **Validation cadence**: `ic_gen` ran 21 inline validation rounds vs exp_073's 1, which is the entire reason it took ~6.7 h vs ~4.8 h — not slower training. Measured 18.3 min per 250-step round, of which ~14.4 min is pure training, matching the lineage exactly.
- **Resume chain**: `ic_gen` crossed a walltime boundary and resumed correctly from checkpoint 2500. Verified end-to-end earlier on `spec_color_rain` (cancelled at 750 → resumed → finished at 2000).
- **Amendment discipline**: two pre-registration amendments were made, both recorded as amendments with reasoning, neither swapped quietly. The 4500-vs-5000 inline eyeball was replaced by a scored checkpoint diagnostic; the donor-pool margin was replaced by the transfer index.

---

## 9. Open items

1. **Convergence diagnostic** (ckpt-4500 vs ckpt-5000 on G-unseen-same + G-zs-cross) — running. Pre-declared bar: UNDERSHOT if Δ ≥ +2.0pp **and** ≥2/3 of items improve. This decides whether the zero-shot loss is real or budget-confounded.
2. **F-block proposal** for §6 — awaiting owner approval; FINDINGS.md is owner-gated.
3. **2AFC** on the claim cells, if wanted — must use a content-aware question ("which shows *this transition* applied to *this scene*"), never "which looks more like class X", which inherits the same defect.
