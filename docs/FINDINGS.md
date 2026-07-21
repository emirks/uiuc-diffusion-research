# FINDINGS — cornerstone results registry

Curated, claim-grade findings destined for the paper's tables. **Governance:
entries are added only after explicit owner approval** — the agent proposes a
drafted F-block in conversation; it lands here on the owner's yes, never before.
Division of record: trajectory → `CHANGELOG.md`; mechanics/how-things-work →
`notes/`; approved claims → this file. Every entry carries the evidence files and
the exact script that regenerates its numbers.

Index:

| ID | claim (one line) | paper target | status |
|---|---|---|---|
| F-001 | Same-class references are interchangeable judges at class level; pool-mean scoring is a valid universal appearance yardstick | metrics/validity table | owner-approved 2026-07-20 |
| F-002 | The sided core-frame mask improves the appearance metric on every measure vs scoring all frames | metrics/validity table | owner-approved 2026-07-20 |

---

## F-001 · Cross-sample reference choice barely moves the appearance score — pool references are a valid universal yardstick

Scoring every corpus GT middle *as if generated*, against every other same-class
clip as reference (all pairwise, both instruments):

| | v3.0.0 (certified) | v4.0.0 (owner default) |
|---|---|---|
| same-class score | 0.362 ± 0.172 | 0.870 ± 0.184 |
| wrong-class score | 0.158 ± 0.082 | 0.494 ± 0.251 |
| separation d′ | 1.52 | **1.71** |
| single-reference swap noise σ_ref | ±0.079 (30% of gap) | **±0.044 (11% of gap)** |
| pool-mean (m≈7) swap noise | ~±0.030 | **~±0.017** |
| leave-own-out top-1 class retrieval | 84% (trusted classes) | 86% |

Consequences: (a) a generation can be scored against *any* same-class reference
pool — one uniform setting for base / specialists / IC arms, including the
foreign tier (donor pool; no GT needed); (b) a *single* random reference is a
noisy judge (rank agreement between two single references ρ≈0.35, range-restricted
lower bound) — use the pool mean; (c) the same-class mean defines a per-class
**GT ceiling**, and arm scores are reported as raw · ceiling · achieved-% (deltas
and tests always on the raw scale; % only for trust-map-trusted classes; >100%
annotated as prototype-mode, not celebrated).

- **Evidence:** `outputs/eval/certification/3.0.0-draft.8/analysis/distance_matrices.npz`
  (`m1a__v3_sided`), `.claude/worktrees/eval-v4-cert/outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz`
  (`m1a_S3`); pool-lane rows `outputs/eval/exp_072_pool_v4/`.
- **Regenerate:** `experiments/exp_072_pool_reference_rescore/ref_stability.py`
  (stability + ceilings), `local_pool_pilot_v4.py` (exact-kernel pilot,
  validated 8/8 vs the v4 sweep), `aggregate.py` (full table once chunks land).
- **Instrument:** v4.0.0 (secondary lane, pre-registered in exp_072 README before
  scoring); v3 bridge via `app_ref_v3`.
- **Status:** owner-approved 2026-07-20.

## F-002 · The sided core-frame mask makes the appearance metric strictly better than scoring all frames

Restricting appearance scoring to the *transition core* (frames where the video
is neither endpoint, sidedness-aware, conditioned windows excluded) beats scoring
all frames on every measure, replicated across two instrument generations:

| core mask | 1-NN style acc (v2 exam) | d (v2) | d′ (v3 matrices) | top-1 (v3) |
|---|---|---|---|---|
| **enabled (sided core)** | **0.927** [0.81, 0.97] | **2.04** | **1.52** | **69%** |
| disabled (all frames) | 0.780 [0.63, 0.88] | 1.11 | 1.27 | 54% |

Endpoint frames dilute style evidence with shared content; the mask removes
exactly that. The v2 ablation (exp_053) survived adversarial re-analysis
("isn't all-frames more robust?" — no: the honest maximum favors the mask), and
the v3 pairwise matrices independently reproduce the ordering on the full
222-clip corpus.

- **Evidence:** `notes/exp/exp_053_eval_harness_v2.md` (ablation table + Check A
  addendum); `outputs/eval/certification/3.0.0-draft.8/analysis/distance_matrices.npz`
  (`m1a__v3_sided` vs `m1a__all_frames`).
- **Regenerate:** `experiments/exp_072_pool_reference_rescore/ref_stability.py`
  (same-vs-cross machinery; pass `m1a__all_frames`).
- **Instrument:** v2 exam + v3.0.0 certification artifacts; mask definition in
  SPEC §3 (S — Structure).
- **Status:** owner-approved 2026-07-20.
