# Eval ladder — SPEC

**Design version: see `VERSION` (currently `2.0.0`).** This file is authoritative. Where any
script, report, README or note disagrees with this document, this document wins and the other
thing is a bug.

The ladder measures **transfer**: does a transition learned from a corpus carry to new content,
to a new demo of a known transition, and to transitions never trained on — and how much of that
is capability rather than lookup?

---

## 1. Principles

These are the rules that keep the ladder simple. Simplicity here is not aesthetic; it is what
makes a result trustworthy a year later. Every one of these was bought with a bug.

1. **One source of truth.** `registry.jsonl` holds every generation, its cell, its GT pool, its
   % type, its priority and its base twin — *derived once* from the frozen inputs. Nothing
   downstream re-derives a fact. If you find yourself computing a row's class from its filename,
   stop: that exact instinct produced the worst bug in the lineage.
2. **One definition per concept, in one file.** One prompt renderer (`prompts.py`), one
   conditioning-window rule (`encode_conditioning.py`), one generator (`run_gen.py`), one
   evaluator (`run_eval.py`). Training and inference read the *same* function — never two
   implementations that are supposed to agree.
3. **Row × seed = one video.** The generator decides nothing. It loads an arm's adapter and
   renders that arm's rows. All selection is `--arm` and `--priority`/`--cells`.
4. **Derive, don't author.** Prompts, cells, pools and twins are computed. A hand-written list
   is a list that will silently drift from the thing it describes.
5. **Seatbelts are asserts, not comments.** A design invariant that is not machine-checked at
   build time is a wish. `build_registry.py` fails loudly rather than emitting a subtly wrong row.
6. **Compare only byte-identical inputs.** Every treatment row has a `base` twin keyed by
   `input_key` = endpoint + prompt + sidedness + reference. The join is exact; a treatment row
   without a twin is a hard error, never a silently dropped row.
7. **Freeze before you look.** Checkpoints, bars and formulas are pinned before any score
   exists. When something must change after the fact, it is an **amendment** — written down,
   reasoned, and labelled — never a quiet swap.
8. **Never delete an inconvenient number.** Pre-registered cells that turn out invalid stay in
   the report, marked `INVALID` with the reason. Fail-forward is the whole point.
9. **Prompts must not leak the answer.** The prompt describes the *starting* scene and holds the
   transition in a neutral token. If it describes the outcome, the ladder measures reading
   comprehension, not transfer. (This is the v1 → v2 defect: `docs/archive/eval_ladder_v1/PROMPT_REDESIGN.md`.)
10. **The report is the deliverable.** A run that produced numbers but no committed record did
    not happen.

---

## 2. Ontology

Two axes, one grid. A generation's cell is fully determined by where it sits.

**Reference novelty** — training exposure of the *transition source*:

| value | meaning |
|---|---|
| `none` | no reference; the transition is in the weights (specialists) |
| `seen` | the exact demo clip was in training |
| `unseen` | a new demo of a class that *was* trained |
| `zero_shot` | a demo of a **held-out** class, never trained |

**Content** — relationship between the endpoint and the donor (transition) class:

| value | meaning | % type |
|---|---|---|
| `same` | endpoint belongs to the donor class | `%_same` |
| `cross` | endpoint belongs to a different corpus class | `%_proxy` |
| `foreign` | endpoint is off-distribution real footage (DAVIS) | `%_proxy` |

Endpoints are **always untrained content** (test band, held-out class, or DAVIS) and **strictly
sidedness-matched** (one-sided endpoints only ever pair with one-sided rows) — except the two
deliberate *fit anchors* (`SP-fit`, `G-fit`), which use train-band endpoints to measure the
memorisation ceiling.

Two gaps fall out of the grid and are the reason it is a grid:

- `G-memo-probe − G-unseen-same` = **demo-instance memorisation** (endpoint novelty held fixed)
- `G-unseen-* − G-zs-*` = **class generalisation**

---

## 3. Arms

| arm | gets | n rows |
|---|---|---|
| `text_floor` | prompt only — no conditioning at all | 12 |
| `base` | prompt + conditioning (+ the same reference where its twin has one), **no adapter** | 203 |
| `spec_<class>` ×11 | + the transition baked into weights | 77 |
| `ic_gen` | + the transition supplied in-context as a demo clip | 152 |

`base` is not a baseline in the loose sense — it is the **counterfactual**: the identical input
with the adapter removed. `text_floor` is the leak-proof floor: if it sits near the pool floor,
the prompt is not carrying the answer.

Arms and their pinned checkpoints live in `arms.yaml`. **Checkpoints are pinned before any score
is seen** — fixed-checkpoint selection, never post-hoc best-checkpoint picking.

---

## 4. Frozen inputs

A run is reproducible only against these. Each report pins their hashes.

| input | identity |
|---|---|
| split | `data/processed/transitions_std121/split_v1.2.json`, sha `c694659d` |
| caption corpus | `clip_captions` via `prompts.captions()` |
| DAVIS roster | `davis.yaml` (block style — a flow-style comma once truncated a caption mid-sentence) |
| arms | `arms.yaml` |
| instrument | `src/diffusion/transition_eval` at a pinned version + commit |

---

## 5. Prompt rule

Rendered, never authored. `prompts.render_prompt()` is the only renderer, and it produces the
training captions **and** the registry rows, so train == inference by construction.

```
one-sided   "{S1}. sksz."
two-sided   "{S1}. sksz. {S2}."
```

`S1` is the starting scene; `S2` (two-sided only) is the ending scene, which the model is *also
given as suffix conditioning* — so stating it leaks nothing. **The outcome half of every caption
is dropped entirely.** `sksz` is a neutral token holding the transition slot; it must be probe-verified
inert on the base model (effect/noise ≤ 1.0) before any training.

Clip → class comes from the **frozen split**, never from string-splitting the clip name.
`action_run_setonfire_6` belongs to class `run_set_on_fire`; `flame_transition_0` to `flame`.

---

## 6. Conditioning rule

One window definition, used by trainer, generator and scorer alike (`encode_conditioning.py`):

| | frames |
|---|---|
| `PX_PREFIX` | 9 |
| `PX_SUFFIX` (encoded) | 9 |
| `SUFFIX_GEN_FRAMES` | 8 |
| total | 121 |

**Causal-VAE bleed fix (load-bearing).** The prefix is clean by causality (rel-L2 8.3e-5) but the
suffix *bleeds* when encoded in context (0.28). `write_cond_clean()` encodes each window in
isolation and asserts bitwise; both trainer and generator read the same `cond_clean_latents_dir`.
Without this, training and inference disagree about what the suffix anchor is.

Conditioning is a **pure function of the row** — never of the class label. The eval mask uses the
same rule, so no stage can disagree about which frames were given.

---

## 7. Registry + seatbelts

`build_registry.py` derives every row and then asserts the design. A build that violates an
invariant **fails**; it never emits a suspect row.

1. no `item_id` collisions
2. every prompt equals `render_prompt()` of its own row; no outcome marker; token present
3. no held-out class in the train roster; no quarantined clip anywhere
4. **keyed join**: every treatment row has exactly one `base` twin by `input_key`
5. the conditioning mask is a pure function of the row's conditioning
6. conditioning windows exist for every endpoint (one rule, checked)
7. cell derivation is self-consistent (novelty, content, reference ≠ endpoint, zero-shot refs untrained)
8. every eval endpoint is leak-audited; train-band endpoints appear only in fit anchors

---

## 8. Evaluation

### 8.1 The yardstick — pool percentage

Score each generation against **every same-class corpus clip of the donor class** (the class the
arm was supposed to produce), average, and divide by that class's **GT ceiling** — the same-class
off-diagonal mean of the instrument's distance matrix.

```
pool-%  =  mean_over_pool( app_ref )  /  ceiling(donor_class)
```

The ceiling carries the same class-spread penalty as the score, so it cancels; that is what makes
% comparable *across* classes. Pool references are **copy-guarded**: never the reference clip,
never the endpoint. Deterministic first-8 by clip name.

Always report **raw · ceiling · %** together. A bare % hides which of the two moved.

### 8.2 The %-typing firewall

| type | when | status |
|---|---|---|
| `%_same` | endpoint class == donor class | fair, cross-class comparable, **headline-eligible** |
| `%_proxy` | `cross` / `foreign` | content-capped: the generation can never fully resemble the donor class because its *content* comes from elsewhere. Absolute level is **ranking-only**; the claim is the margin Δpp vs the base twin, where the identical cap cancels |

`run_eval.report()` prints % everywhere and **refuses to aggregate across % types** (hard assert).

### 8.3 Unit of analysis

The **donor class**, not the item: per-donor mean of the margin, then a sign test.
`PASS` requires ≥ 80 % of donor classes positive; otherwise `weak`.

### 8.4 Dedup

Scoring is incremental and runs in several passes. A row planned twice — a generation that landed
between one pass's plan and its score — is written to both passes' `items.jsonl`. **Both reporters
dedup on the eval id.** Counting a row twice silently reweights that generation's pool mean.

---

## 9. Claim cells and bars

Only these carry claims. Everything else is descriptive or diagnostic.

| claim cell | bar |
|---|---|
| `SP-same` | Δpp > 0 with ≥ 80 % of donor classes positive |
| `SP-cross` | same |
| `G-unseen-cross` | ΔTI > 0 with ≥ 80 % of donor classes positive |
| `G-zs-cross` | same |

Support cells (`SP-fit`, `G-fit`, `G-memo-probe`, `G-ref-control`, `text_floor`) are **anchors and
controls** — they make the claim cells interpretable and never carry a claim themselves.

`G-ref-control` deserves its name: its demo is deliberately *mismatched*. Without it you cannot
show that a model uses the in-context demo as an instruction rather than as generic context.

---

## 10. Kill rules

Checked during training; a fired rule stops the run and is recorded in the report.

| rule | condition |
|---|---|
| **K0** | every job must log its `cond_clean` smoke assert before step 250 |
| **K1** | at each model's step 500: loss NaN, loss not decreasing, or the control sample shows a spurious effect |
| **K2** | ≥ 50 % of models reaching step 500 fail K1 (or both pilots fail) → cancel the fleet, consult |
| **K3** | neither pilot shows ID class-effect onset by step 1000 → kill the fleet, consult |

**Never respond to a fired kill rule by reverting to leaky prompts.**

---

## 11. Amendments

A pre-registered quantity may be replaced only by a written amendment that states: what is being
replaced, why the original is invalid, the new formula/bar **in full**, and whether it was written
before or after the corrective numbers existed. Amendments live in the run's report and are
labelled `pre-registered` or `outcome-aware`.

**Amendment-1 (design v2.0.0)** — the donor-pool margin is invalid for reference-bearing cells,
because the base twin can win by reproducing the demo. Replaced by the **transfer index**:

```
T  = clip01( (D% − D_ep%) / (D_ref% − D_ep%) )     donor manner arrived
C  = clip01( (R% − R_ref%) / (R_ep% − R_ref%) )     endpoint content kept
TI = min(T, C)
```

`min`, not mean: transfer is a **conjunction**, and averaging lets copying buy back score.
2×2 quadrants at (0.5, 0.5); anchor denominators < 5 pp → `anchor_degenerate`, excluded.

---

## 12. Versioning

Two things are versioned, and they are versioned differently on purpose.

### 12.1 The design — semver in `VERSION`

| bump | when |
|---|---|
| **MAJOR** | comparability-breaking: the ontology, a cell definition, or the prompt-rendering rule changes. Cross-run comparison dies at this boundary. |
| **MINOR** | additive: a new cell, a new seatbelt, a new metric surfaced. Old cells stay comparable. |
| **PATCH** | docs, scripts, refactors that change nothing measured. |

A **run never bumps the version.** The design is what is versioned; a run is an observation of it.

### 12.2 A run — a monotonic sequence number

Each run gets the next `R<N>`, never reset across design bumps, so chronology is `ls`-visible.
A report is named:

```
reports/v<DESIGN>-R<N>.md          e.g.  reports/v2.0.0-R1.md
```

**The instrument version is recorded inside the report, never in the filename** — two versions in
one filename is noise, and the instrument is pinned by its own tag and commit.

### 12.3 Which run is current

The newest `R<N>` whose report says `VALID` or `VALID-WITH-AMENDMENTS`. `reports/README.md`
carries a one-line index, newest first — the same convention as the instrument's
`certifications/README.md`.

### 12.4 Validity, not pass/fail

A certification passes because the *instrument* meets bars. A ladder run *measures models*, and a
model scoring badly is a valid result — so there is deliberately **no overall run PASS/FAIL**;
that would be a category error and an invitation to game bars.

Verdicts live on **cells**:

| cell verdict | meaning |
|---|---|
| `PASS` | met its pre-registered bar |
| `FAIL` | missed it — a real, reportable negative |
| `INVALID(reason, amendment)` | the statistic does not measure what it was meant to; retained, never deleted |

The run itself carries one line: `VALID` / `VALID-WITH-AMENDMENTS` / `INVALID-RUN` (reserved for a
fired kill rule or a broken instrument pairing).

---

## 13. Lineage

- **v1** — `docs/archive/eval_ladder_v1/`. Three generations of item lists and viewers. Retired
  because every arm was prompted with the **full caption, which described the outcome**; the
  forensic record is `PROMPT_REDESIGN.md` in that archive.
- **v2.0.0** — this design. Leak-free rendered prompts, derived registry, keyed base twins,
  strict sidedness matching, the %-typing firewall, `G-ref-control`.
- The standing pool-yardstick definition is `POOL_YARDSTICK.md` beside this file.
