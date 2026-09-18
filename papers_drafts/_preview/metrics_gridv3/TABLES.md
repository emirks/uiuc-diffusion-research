# Grid v3 metric tables (preview twin)

_Generated 2026-09-18 by `scripts/build_metric_tables.py`._

These are the same numbers as `tab_A.tex` / `tab_B.tex` / `tab_C.tex`, in markdown, for review. Bold marks the best value per column per block (smallest for Copy rate and for the text-dependency $\Delta$). Numbers: transport and rates to 1 decimal, similarities to 3. A cell whose defining $n < 10$ renders `n/a` (with its $n$) and never bolds.

Inputs joined on `(item_id, seed)`:
- **per_gen**: 19 file(s), 7242 rows from `base_cond_effect_v3`
- **handoff**: 19 file(s), 7242 rows from `rows.jsonl`
- **lens**: 19 file(s), 7242 rows from `rows.jsonl`
- **copy**: 19 file(s), 7242 rows from `rows.jsonl`

## Table A -- own arms across tiers (neutral prompt)
| arm | Id A | Id B | Mot A | Mot B | Seam% | Smooth | Copy% | Copymax | Transport | n |
|---|---|---|---|---|---|---|---|---|---|---|
| **Seen** | | | | | | | | | |
| LTX-2 (no reference) | **0.948** | n/a (n=8) | 0.633 | n/a (n=6) | 82.7 | **0.991** | **0.0** | **0.247** | 57.9 | 52 |
| LTX-2 baseline LoRA | 0.877 | n/a (n=8) | **0.701** | n/a (n=8) | 94.2 | 0.988 | **0.0** | 0.332 | 82.2 | 52 |
| SEGUE w/o guidance | 0.938 | n/a (n=8) | 0.679 | n/a (n=1) | 96.2 | 0.986 | **0.0** | 0.340 | 91.1 | 52 |
| SEGUE | 0.908 | n/a (n=8) | 0.631 | n/a (n=1) | **98.1** | 0.981 | **0.0** | 0.401 | **97.1** | 52 |
| **Unseen** | | | | | | | | | |
| LTX-2 (no reference) | **0.935** | **0.960** (n=64) | 0.743 (n=269) | **0.695** (n=59) | 90.1 | **0.990** | **0.7** | **0.205** | 52.1 | 274 |
| LTX-2 baseline LoRA | 0.930 | 0.931 (n=64) | **0.757** | 0.441 (n=47) | **97.8** | 0.986 | **0.7** | 0.269 | 74.6 | 274 |
| SEGUE w/o guidance | 0.917 | 0.923 (n=64) | 0.725 (n=271) | 0.445 (n=35) | 95.6 | 0.983 | **0.7** | 0.348 | 86.3 | 274 |
| SEGUE | 0.862 | 0.880 (n=64) | 0.710 (n=273) | 0.399 (n=20) | 92.3 | 0.977 | **0.7** | 0.416 | **92.1** | 274 |
| **Zero-shot** | | | | | | | | | |
| LTX-2 (no reference) | **0.953** | **0.950** (n=76) | 0.718 (n=231) | **0.681** (n=63) | 64.7 | **0.993** | **0.9** | **0.198** | 45.5 | 442 |
| LTX-2 baseline LoRA | 0.950 | 0.926 (n=76) | **0.757** (n=236) | 0.462 (n=64) | 95.9 | 0.991 | **0.9** | 0.252 | 62.2 | 442 |
| SEGUE w/o guidance | 0.926 | 0.891 (n=76) | 0.745 (n=235) | 0.311 (n=41) | **97.1** | 0.986 | 1.4 | 0.405 | 88.5 | 442 |
| SEGUE | 0.867 | 0.850 (n=76) | 0.741 (n=234) | 0.368 (n=44) | 94.1 | 0.982 | 1.4 | 0.472 | **93.7** | 442 |

## Table B -- comparison with previous approaches (shared one-sided zero-shot set)
Shared set (intersection over the 6 Table-B arms): **n = 366** (target 366 = 183 triples x 2 seeds).
Per-arm bench coverage before intersection: `{'vap': 366, 'vfxmaster': 366, 'refvfx': 366, 'ic_gen': 366, 'dualforce_control': 366, 'dualforce_dcg_w6': 366}`; rows lost in the match per arm: `{'vap': 0, 'vfxmaster': 0, 'refvfx': 0, 'ic_gen': 0, 'dualforce_control': 0, 'dualforce_dcg_w6': 0}`.

**Copy rate source & frame-count caveat.** Copy rate is `100·mean(near_copy)` from the M2a copy eval (`*_copy_gridv3*`), which scores each generation against its OWN reference -- NOT per_gen's `near_copy`, which was scored against pool clips and is identical across arms (not a generation property). M2a takes the max over the generation's mid frames, so a longer generation has more chances to match: our clips are 121 f vs the externals' 49 f (VAP/VFXMaster) / 33 f (refVFX), which under-estimates the externals' copy rate -- disclosed, not corrected.

**Motion fid.** is NaN where no tracklets move (valid); n per cell.
| arm | Id A | Seam% | Smooth | Copy% | Copymax | Transport | RefSim(VP) | MotFid | Aesth | n |
|---|---|---|---|---|---|---|---|---|---|---|
| Video-As-Prompt | 0.879 | 76.0 | 0.979 | 1.6 | 0.396 | 81.1 | 0.958 | 0.137 (n=333) | 4.901 | 366 |
| VFXMaster | 0.964 | 85.5 | 0.987 | **0.5** | 0.412 | 85.8 | 0.962 | 0.192 (n=363) | 4.856 | 366 |
| refVFX | **0.971** | 93.2 | 0.988 | 1.1 | 0.352 | 62.2 | 0.957 | 0.161 (n=284) | **5.305** | 366 |
| LTX-2 baseline LoRA | 0.961 | 95.4 | **0.994** | **0.5** | **0.246** | 61.6 | 0.954 | 0.167 (n=312) | 5.227 | 366 |
| SEGUE w/o guidance | 0.936 | **96.4** | 0.990 | 1.1 | 0.406 | 87.6 | 0.963 | 0.203 (n=364) | 5.193 | 366 |
| SEGUE | 0.880 | 93.4 | 0.985 | 1.1 | 0.473 | **92.6** | **0.965** | **0.226** | 5.084 | 366 |

## Table C -- text dependency (same shared set)
Shared set: **n = 366**.
| arm | T neutral | T effect | T Δ | VP neutral | VP effect | VP Δ | n |
|---|---|---|---|---|---|---|---|
| LTX-2 (no reference) | 46.2 | 88.7 | +42.6 | 0.943 | 0.958 | +0.015 | 366 |
| LTX-2 baseline LoRA | 61.6 | 88.2 | +26.6 | 0.954 | 0.961 | +0.008 | 366 |
| SEGUE w/o guidance | 87.6 | 92.7 | +5.1 | 0.963 | 0.965 | +0.002 | 366 |
| SEGUE | 92.6 | **94.4** | **+1.9** | 0.965 | **0.966** | **+0.001** | 366 |
| Video-As-Prompt | -- | 81.1 | -- | -- | 0.958 | -- | 366 |
| VFXMaster | -- | 85.8 | -- | -- | 0.962 | -- | 366 |
| refVFX | -- | 62.2 | -- | -- | 0.957 | -- | 366 |

## Table D -- ablation (guidance w-sweep)
**Not built tonight.** The w-sweep arms (w = 1, 1.5, 3) were not generated on grid v3, so there are no grid-v3 rows to aggregate. Table D stays as the paper's `tab_ablation.tex` draft until the sweep is regenerated on grid v3.

## Placeholder (`\ph{--}`) cells and why
Each line is a row whose listed columns render as `--` because the backing input row set is absent or empty:

- C / vap: t_neu, t_d, v_neu, v_d
- C / vfxmaster: t_neu, t_d, v_neu, v_d
- C / refvfx: t_neu, t_d, v_neu, v_d

## Shared-set / `--strict` checks
- No `--strict` violations: every present Table B/C column has a single $n$ across its rows.

### Known column-choice caveat
Our Transport rows carry Op-2's `transport_pct` (the S3 / `app_ref` column). The paper draft's refVFX zero-shot value 75.5 is from the `Look_u` column, not S3; the S3 refVFX value is ~62. Compare a system with itself, and only across rows on the same metric column.

