# Grid v3 metric tables (preview twin)

_Generated 2026-09-18 by `scripts/build_metric_tables.py`._

These are the same numbers as `tab_A.tex` / `tab_B.tex` / `tab_C.tex`, in markdown, for review. Bold marks the best value per column per block (smallest for Copy rate and for the text-dependency $\Delta$). Numbers: transport and rates to 1 decimal, similarities to 3.

Inputs joined on `(item_id, seed)`:
- **per_gen**: 4 file(s), 1536 rows from `ic_gen_effect_v3`
- **handoff**: 0 file(s), 0 rows -- **ABSENT** (columns render `--`)
- **lens**: 0 file(s), 0 rows -- **ABSENT** (columns render `--`)

## Table A -- own arms across tiers (neutral prompt)
| arm | Id A | Id B | Mot A | Mot B | Seam% | Smooth | Copy% | Transport | n |
|---|---|---|---|---|---|---|---|---|---|
| **Seen** | | | | | | | | | |
| LTX-2 (no reference) | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| LTX-2 baseline LoRA | -- | -- | -- | -- | -- | -- | **0.0** | **82.2** | 52 |
| SEGUE w/o guidance | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| SEGUE | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **Unseen** | | | | | | | | | |
| LTX-2 (no reference) | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| LTX-2 baseline LoRA | -- | -- | -- | -- | -- | -- | **3.3** | **74.6** | 274 |
| SEGUE w/o guidance | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| SEGUE | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **Zero-shot** | | | | | | | | | |
| LTX-2 (no reference) | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| LTX-2 baseline LoRA | -- | -- | -- | -- | -- | -- | **0.9** | **62.2** | 442 |
| SEGUE w/o guidance | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| SEGUE | -- | -- | -- | -- | -- | -- | -- | -- | -- |

## Table B -- comparison with previous approaches (shared one-sided zero-shot set)
Shared set (intersection over the 6 Table-B arms): **n = 366** (target 366 = 183 triples x 2 seeds).
Per-arm bench coverage before intersection: `{'vap': 0, 'vfxmaster': 0, 'refvfx': 0, 'ic_gen': 366, 'dualforce_control': 0, 'dualforce_dcg_w6': 0}`; rows lost in the match per arm: `{'vap': 0, 'vfxmaster': 0, 'refvfx': 0, 'ic_gen': 0, 'dualforce_control': 0, 'dualforce_dcg_w6': 0}`.
| arm | Id A | Seam% | Smooth | Copy% | Transport | RefSim(VP) | MotFid | Aesth | n |
|---|---|---|---|---|---|---|---|---|---|
| Video-As-Prompt | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| VFXMaster | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| refVFX | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| LTX-2 baseline LoRA | -- | -- | -- | **1.1** | **61.6** | -- | -- | -- | 366 |
| SEGUE w/o guidance | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| SEGUE | -- | -- | -- | -- | -- | -- | -- | -- | -- |

## Table C -- text dependency (same shared set)
Shared set: **n = 366**.
| arm | T neutral | T effect | T Δ | VP neutral | VP effect | VP Δ | n |
|---|---|---|---|---|---|---|---|
| LTX-2 (no reference) | -- | -- | -- | -- | -- | -- | -- |
| LTX-2 baseline LoRA | 61.6 | **88.2** | **+26.6** | -- | -- | -- | 366 |
| SEGUE w/o guidance | -- | -- | -- | -- | -- | -- | -- |
| SEGUE | -- | -- | -- | -- | -- | -- | -- |
| Video-As-Prompt | -- | -- | -- | -- | -- | -- | -- |
| VFXMaster | -- | -- | -- | -- | -- | -- | -- |
| refVFX | -- | -- | -- | -- | -- | -- | -- |

## Table D -- ablation (guidance w-sweep)
**Not built tonight.** The w-sweep arms (w = 1, 1.5, 3) were not generated on grid v3, so there are no grid-v3 rows to aggregate. Table D stays as the paper's `tab_ablation.tex` draft until the sweep is regenerated on grid v3.

## Placeholder (`\ph{--}`) cells and why
Each line is a row whose listed columns render as `--` because the backing input row set is absent or empty:

- A / Seen / base_cond: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Seen / ic_gen: identity_a, identity_b, motion_a, motion_b, seam, smooth
- A / Seen / dualforce_control: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Seen / dualforce_dcg_w6: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Unseen / base_cond: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Unseen / ic_gen: identity_a, identity_b, motion_a, motion_b, seam, smooth
- A / Unseen / dualforce_control: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Unseen / dualforce_dcg_w6: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Zero-shot / base_cond: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Zero-shot / ic_gen: identity_a, identity_b, motion_a, motion_b, seam, smooth
- A / Zero-shot / dualforce_control: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- A / Zero-shot / dualforce_dcg_w6: identity_a, identity_b, motion_a, motion_b, seam, smooth, copy, transport  (no rows in scope)
- B / vap: identity_a, seam, smooth, copy, transport, vpref, motfid, aes  (no rows in shared set)
- B / vfxmaster: identity_a, seam, smooth, copy, transport, vpref, motfid, aes  (no rows in shared set)
- B / refvfx: identity_a, seam, smooth, copy, transport, vpref, motfid, aes  (no rows in shared set)
- B / ic_gen: identity_a, seam, smooth, vpref, motfid, aes
- B / dualforce_control: identity_a, seam, smooth, copy, transport, vpref, motfid, aes  (no rows in shared set)
- B / dualforce_dcg_w6: identity_a, seam, smooth, copy, transport, vpref, motfid, aes  (no rows in shared set)
- C / base_cond: t_neu, t_eff, t_d, v_neu, v_eff, v_d
- C / ic_gen: v_neu, v_eff, v_d
- C / dualforce_control: t_neu, t_eff, t_d, v_neu, v_eff, v_d
- C / dualforce_dcg_w6: t_neu, t_eff, t_d, v_neu, v_eff, v_d
- C / vap: t_neu, t_eff, t_d, v_neu, v_eff, v_d
- C / vfxmaster: t_neu, t_eff, t_d, v_neu, v_eff, v_d
- C / refvfx: t_neu, t_eff, t_d, v_neu, v_eff, v_d

## Shared-set / `--strict` checks
- No `--strict` violations: every present Table B/C column has a single $n$ across its rows.

### Known column-choice caveat
Our Transport rows carry Op-2's `transport_pct` (the S3 / `app_ref` column). The paper draft's refVFX zero-shot value 75.5 is from the `Look_u` column, not S3; the S3 refVFX value is ~62. Compare a system with itself, and only across rows on the same metric column.

