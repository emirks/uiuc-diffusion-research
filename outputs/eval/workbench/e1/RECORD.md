# E1 — effect-delta vector · §4.1 KILL TEST · RECORD

Rung: E1 (RUNBOOK §4.1). Run 2026-07-14. Verdict commit `72a5bd4`.
Gates frozen at `694afc7`, before any candidate computation.

---

## Verdict (§4.1, recorded, final)

RUNBOOK §4.1, verbatim: *"if delta fails to beat raw M1a on **both** Cohen's d and
misretrieved count, the endpoint-normalization program is dead at the appearance
level. One appendix paragraph, full stop. E2/E3 do not run."*

| | E1 delta | pinned incumbent `m1a__v3_sided` | beats? |
|---|---|---|---|
| Cohen's d | 0.358190 | 1.522006 | **NO** |
| misretrieved | 209/223 | 73/223 | **NO** |
| accuracy (1-NN) | 0.0628 | 0.672646 | — (chance = 0.067) |
| coverage | 1.0000 | 1.0000 | — |

**VERDICT: KILL. E2/E3 do not run.**

Independently, the §1.4 hubness gate: **FAIL** — skew 4.300 > 3.0; prediction
entropy 0.322 < 0.70; max prediction-class share 0.650 > 0.25; sink class
`mystification`. Definedness: 223/223 deltas defined, coverage 1.0000, so the
result is not a shrunken-support artifact.

Pinned pre-run choices (gates.yaml, before any E1 number existed): `v_null` is
pooled over the clip's own core frame indices; low_D clips are not excluded from
E1 (the delta contains no D).

---

## Instrument diagnostics (facts; they do not revise the verdict)

A below-chance accuracy has the shape of a plumbing bug, so the pipeline was
controlled. The control is the identical representation with **no null subtraction
at all** — i.e. containing zero endpoint-normalization, the thing §4.1 exists to
test.

| arm | accuracy | Cohen's d | misretrieved | hubness entropy |
|---|---|---|---|---|
| RAW `v_clip`, no subtraction | 0.6054 | 0.9875 | 88/223 | 0.882 |
| WHITENED `v_clip`, no subtraction | 0.0628 | 0.1068 | 209/223 | 0.042 |
| WHITENED delta (**the candidate**) | 0.0628 | 0.3582 | 209/223 | 0.322 |
| RAW delta (unwhitened) | 0.1345 | 0.6211 | 193/223 | 0.524 |

ZCA fit spectrum: 768 dims, eig_max 3.637e-02, eig_min 1.848e-09, condition number
~1.97e7. The frozen floor (1e-6 · λ_max = 3.64e-08) floors **1 of 768** dimensions.
Whitened norm means: null 35.52 vs clip 14.61 (raw: 0.912 vs 0.867).

Plumbing verified sound: nulls are genuinely distinct from their clips
(cos(clip_core_mean, null_core_mean) mean 0.700, min 0.113, max 0.988 — not an
accidental copy); 223/223 deltas defined.

Floor-sensitivity diagnostics (grid **pre-declared** in `CONSULTATIONS.md` C3
before being computed; `e1_floor_sensitivity.json`): across
`eig_floor_ratio ∈ {1e-6 … 1e-1}`, the whitened no-subtraction control stays at
accuracy 0.0628 from 1e-6 through 1e-3 and reaches 0.3722 only at 1e-1 (704 of 768
dims floored, 64 effective dimensions), still below the raw control's 0.6054. The
whitened delta does not exceed accuracy 0.0942 or fall below 202/223 misretrieved
at any floor in the grid. No floor is recommended or selected — that is
owner-reserved. No sweep row carries a verdict or is scored against any gate.

---

## Owner-reserved matter (escalated; track stopped)

RUNBOOK §1.1 mandates whitening but **does not pin its regularization**.
`eig_floor_ratio = 1e-6` was chosen by the executor and frozen in `gates.yaml`
(`694afc7`) before any candidate ran. Whether this frozen parameter confounds the
§4.1 test, and whether the recorded verdict bears on the endpoint-normalization
hypothesis, are outcome-aware threshold and kill-rule-interpretation questions —
owner-reserved under OPERATIONS §1(5) and §8. No workbench action revises the
recorded verdict. The floor-sensitivity diagnostics are attached so this can be
adjudicated without a re-run.

---

## Not run

- **E2** (γ-signature), **E3** (within-video Gram): not run, per the §4.1 recorded
  KILL. `gates.yaml` makes this structural — the driver refuses E2/E3 unless E1's
  recorded verdict is a pass.
- **§4.5 ablations**: not run — they apply to "whichever rung survives" (§4.5), and
  none did.

## Ran regardless

- **E0** (§4.2) — it "rides the E1 run, no gating numbers" and is not conditioned
  on E1 passing. See `../e0/`.
