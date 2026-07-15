# E1′ — γ-scalar signature · §2.4 KILL FIRED · INSTRUMENT VALID · WORKBENCH CLOSED

Run 2026-07-14. Authority `E1PRIME_DIRECTIVE.md`. Pre-registration frozen at
**`aace78d`**, before any candidate number existed. Raw-space min-D floor frozen at
**`6930eae`**, before any candidate distance existed.

**Terminal outcome: §2.6 case 1 — KILL (IV pass).** The endpoint-normalization program
is dead at the appearance level, **adjudicated this time**. Workbench closed. No third
cycle under any outcome.

---

## 1. The rules that fired

**§2.3 instrument validity — BOTH PASS. The kill therefore BINDS THE HYPOTHESIS.**

| precondition | form | accuracy | floor | verdict |
|---|---|---|---|---|
| **IV1** effect vs nothing | binary LOO 1-NN, {223 clips, 223 rendered nulls} | **0.9357** | 0.90 | **PASS** |
| **IV2** snap vs nothing | binary LOO 1-NN, {37 Bar-6 hard cuts, 37 matched lerps} | **1.0000** | 0.90 | **PASS** |

**§2.4 kill rule (verbatim):** *"If the candidate fails to beat pinned `m1a__v3_sided`
(d 1.522006, misretrieved 73/223) on **both** Cohen's d and misretrieved count → KILL,
one appendix paragraph, workbench closed."*

| | candidate | incumbent | beats? |
|---|---|---|---|
| Cohen's d | **0.797854** | 1.522006 | **NO** |
| misretrieved | **183 / 223** | 73 / 223 | **NO** |

**Both fail. KILL.** The candidate is not marginal: it loses by a factor of 1.91 on d
and 2.51 on misretrieved count.

## 2. The arms (§2.2's closed list — nothing else was computed)

| arm | role | acc | Cohen's d | coverage | misretrieved | hubness |
|---|---|---|---|---|---|---|
| **A (â, b̂, m̃) raw** | **THE GATING ARM** | 0.1905 | **0.7979** | 0.9417 | **183** | PASS |
| B (â, b̂, m) — no null subtraction | diagnostic control | 0.2143 | 0.6794 | 0.9417 | 178 | PASS |
| C (â, b̂, m̃) Ledoit-Wolf whitened | diagnostic, non-gating | 0.2048 | 0.8047 | 0.9417 | 180 | PASS |
| D m̃ alone | diagnostic | 0.0905 | 0.4336 | 0.9417 | 204 | PASS |
| — *A under σ_emb* | *NON-GATING σ column* | *0.2048* | *0.7763* | *0.9417* | *180* | *PASS* |
| `m1a__v3_sided` (pinned incumbent) | — | 0.672646 | **1.522006** | 1.0000 | **73** | PASS |

Chance = 0.067. The signature carries **real class signal** (0.19 ≈ 2.8× chance) and is
**far** below the incumbent (0.67).

**The hubness gate PASSES on every arm** (skew −0.10 – 0.41 against the 3.0 ceiling;
entropy 0.899 – 0.920 against the 0.70 floor; max-pred 0.076 – 0.090 against the 0.25
ceiling). The failure is **not** a hub collapse, and no sink is suppressing the score.

## 3. The pre-declared σ sensitivity column — the verdict is NOT σ-sensitive

P1 named one genuinely open reading (the σ-parameterization) and pre-declared a
non-gating column for the alternative, **before any number existed**, so that a
σ-sensitive verdict would be adjudicable without a re-run.

It is not σ-sensitive. Gating σ (arc length of the signature curve): **d 0.7979**.
σ_emb (arc length of the raw 768-d embedding path): **d 0.7763**. Both are less than
**half** the incumbent's 1.522006, and both fail the kill rule on both legs. **The open
reading cannot change the verdict.** It is hereby closed as a live question.

## 4. The IV disclosure columns — the instrument is valid on its merits, not on the tautology

PREREG §P5a disclosed, before the IV ran, that every "nothing" object has **m̃ ≡ 0
exactly** (`make_lerp` is idempotent on its own endpoints — proved by test), so the IV is
partly tautological on that channel, and pre-declared a non-gating column restricted to
the two channels that are **not** structurally constant for a null.

| | full signature | (â, b̂) only — NON-GATING |
|---|---|---|
| IV1 | 0.9357 | **0.8476** |
| IV2 | 1.0000 | **1.0000** |

Both columns are far above chance (0.50). **The IV passes do not rest on the
tautological channel** — the signature separates effect-from-nothing at 0.85 and
cut-from-crossfade at 1.00 with the m̃ channel removed entirely. The disclosure was
made in advance and the answer came back reassuring; the instrument is valid on its
merits, and the kill binds.

IV1 pool coverage 0.9417 (420/446 defined; the 13 undefined clips and their 13 nulls).
IV2 pair coverage **1.0000** over all 37 n≥2 classes.

## 5. §7 adoption conditions, as computed facts

| condition | value | threshold | pass |
|---|---|---|---|
| 1. Cohen's d ≥ 1.772006 | 0.797854 | 1.772006 | **FAIL** |
| 2. misretrieved < 73 | 183 | 73 | **FAIL** |
| 3. hubness gate | skew 0.41 / H 0.920 / max-pred 0.076 | 3.0 / 0.70 / 0.25 | PASS |
| 4. coverage not *materially* narrower | 0.9417 vs 1.0000 | — | **OWNER-RESERVED** |
| 5. full probe battery | not run | — | n/a (kill test, not an adoption run) |

**§7 ALL-PASS: FALSE**, on conditions 1 and 2, independently and terminally.

**Condition 4 is escalated, not decided.** "Materially" is not a threshold, and §2.7
forbids the executor to invent one. The **fact**: coverage 0.9417 = 210/223, with 13
undefined rows — **12 low_D** + **1 empty core mask**. The shortfall is entirely a
consequence of the **frozen §1.2 min-D guard**, which the γ-signature triggers and E1's
delta did not (the delta contains no D; every γ channel divides by it). **Moot for the
§7 call**: conditions 1 and 2 fail on their own.

## 6. §5 predictions, re-checked in registered form (DESCRIPTIVE — nothing here gates)

**P1 — sibling γ-distance < clip-to-own-null γ-distance, per n≥4-eligible class.**
**9 / 28** eligible classes have *every* clip closer to a sibling than to its own null.
At the clip level, **155 / 189** clips (82%) satisfy it. As registered ("for every
n≥4-eligible class"), **P1 fails**. (e0's cruder m-mean analogue gave 8/29 — the
directly-checkable form lands in the same place.)

**P2 — sidedness recoverable from s-asymmetry.** Registered statistic (e0's, verbatim:
`mean(â) − mean(1 − b̂)` over σ): **rank AUC 0.1017**.

Stated precisely: this is **not** "no separation" (which would be 0.50). The two
sidedness populations are **strongly separated** — a classifier on the *negation* of the
registered statistic scores **0.8983** — but in the direction **opposite** to the one
the registered statistic's phrasing implies. Sidedness *is* recoverable from the
asymmetry; the registered statistic's sign is inverted relative to the prediction.
**No mechanism claim is attached.** P2 is descriptive and non-gating.

(e0's figure was 0.523 — computed on per-channel arc-length curves that linearize any
monotone channel; see `E1PRIME_AMENDMENTS.md` §A. This one is on E1′'s shared-σ curves.
Both are descriptive; neither ever gated anything.)

**P4 — one-sided classes concentrate excursion mass in early σ.** On the registered m̃
channel: σ-centroid **0.3263** (one-sided) vs **0.4626** (two-sided) → one-sided
**earlier**. **P4 HOLDS.** On the raw m channel (for e0 comparability): 0.5132 vs
0.4991 → one-sided *later*; P4 does not hold there.

## 7. Arm C — what Ledoit-Wolf says about escalation (a)

Escalation (a) was adjudicated by the owner before this cycle ran, and E1-as-specified
is never re-run. This is recorded as a **fact bearing on that already-made
adjudication**, not as a reopening of it.

E1's whitener used an **executor-chosen** `eig_floor_ratio = 1e-6`, and its whitened
no-subtraction control scored **0.0628** (chance) against a raw control's **0.6054**.
Arm C substitutes **Ledoit-Wolf**, which has **no free parameter** (shrinkage
δ = 0.007576, computed from the data; condition number 1.968e7 → 4.062e3; no eigenvalue
floor anywhere).

Under Ledoit-Wolf the whitened arm scores **d 0.8047 / acc 0.2048** — i.e.
**indistinguishable from the raw gating arm** (0.7979 / 0.1905), not collapsed to chance.

The mechanism is isolated in a unit test
(`test_ledoit_wolf_does_not_amplify_near_null_directions_but_the_eig_floor_does`): an
eigenvalue floor divides near-null directions by `sqrt(floor)` and amplifies pure noise
to unit variance, whereas Ledoit-Wolf divides them by `sqrt(δ·m)` and leaves them small.

**The fact, stated without extension:** a parameter-free whitener does **not** flatten
this signature, while the floored whitener flattened E1's delta to chance. This bears
on which of the two candidate explanations in escalation (a) — "whitening" versus "the
executor-chosen floor" — the evidence favours. **The owner has already adjudicated
escalation (a) and E1-as-specified does not re-run; nothing here changes that.**

## 8. Consequences (rule text + numbers)

- **§2.4 KILL fires**, and **§2.3 makes it BIND THE HYPOTHESIS** (both IVs pass, and
  pass on their non-tautological channels too).
- **§2.6 case 1:** endpoint-normalization is **dead at the appearance level,
  adjudicated**. One appendix paragraph. **Workbench closed.**
- **No third cycle under any outcome.** No E2, no E3, no rescue variant, no repair.
- Phase 1 (motion) remains closed and terminal (`2de4835`). M1b_flow / M1c_flow remain
  analysis-tier.
- `m1a__v3_sided` remains the appearance incumbent, unchallenged and unchanged.

## 9. Determinism

The full run was executed **twice**. Every gating number — all four arms' accuracy,
Cohen's d, coverage and misretrieved count, and both IV accuracies — is **bit-identical
across re-runs**.

## Artifacts

`e1prime.json` (arms, IVs, kill rule, §7, predictions) · `mind_raw_frozen.json` (the
corpus-only min-D calibration) · `A_gating_distance_matrix.npz` ·
`B_no_null_sub_distance_matrix.npz` · `C_ledoit_wolf_distance_matrix.npz` ·
`D_m_tilde_alone_distance_matrix.npz` · `A_gating__sigma_emb_distance_matrix.npz` ·
`$WB_CACHE/iv2/manifest.json` + 37 hard-cut probes + 37 matched lerps.
