# Metric Workbench — Operating Contract

Execution mechanics for `RUNBOOK.md` (same directory — the frozen
pre-registered spec; read it first and in full). This file says HOW; the
RUNBOOK says WHAT and carries the pinned baselines (§B) and every gate/kill/
adoption rule. Where they disagree, RUNBOOK wins on science, this file wins
on paths and mechanics. The certified instrument is governed by `SPEC.md`
and is out of scope for edits entirely.

**Division of labor (binding):** this branch is executed by an
implementation agent whose entire mandate is build → test → run → **report
neutrally**. Interpretation, synthesis, strategy, and the §7 adoption call
happen OUTSIDE this branch, in owner-side review. See §8.

---

## 1. Topology — what is writable, what is read-only

| Location | Role | Access |
|---|---|---|
| `$LAB/diffusion-research/.claude/worktrees/metric-workbench/` (branch `eval/metric-workbench`, base `9b1a4cb`) | THIS worktree — all workbench code, docs, outputs, commits | read/write |
| `$LAB/diffusion-research/.claude/worktrees/eval-v3-spec/` | certified instrument's run artifacts, warm cache, corpus videos | **READ-ONLY. Never write, never touch mtimes.** |
| `$LAB/diffusion-research/` (main checkout) | user's working tree with an uncommitted experiment backlog | **DO NOT TOUCH — no commits, no file edits, no cleanup, no pulls.** |

`$LAB = /projects/illinois/eng/cs/jrehg/users/emirkisa`.

Hard prohibitions (each one protects a certified property):

1. **Never edit certified modules** — anything in `src/diffusion/transition_eval/`
   outside `workbench/` is frozen at the certified content. Candidates IMPORT
   deployed code; they never copy it and never patch it. Adoption happens
   later via v3.1 re-cert, not on this branch.
2. **Never run candidates through `certify/run_certification.py` or
   `certify/exam.py`.** The workbench has its own driver. Bars
   (`certify/bars.yaml`) stay untouched.
3. **The shared cache is read-only** (`…/eval-v3-spec/outputs/eval/cache`).
   New artifact types (flow, rendered nulls, ZCA) go in this worktree's own
   cache dir. Never write into the shared cache; a polluted entry can fail
   the certified warm-determinism bar later. (LPIPS is not needed by any
   workbench metric — do not compute it at all.)
4. **Kill rules are terminal.** A failed gate gets an honest negative record
   and the track stops. No rescue variants, no threshold adjustments, no
   second attempts — §3.6/§4.1/§9 of the RUNBOOK are the law.
5. **Owner decides health-assessment matters.** Anything not pre-registered
   in the RUNBOOK that would gate an outcome (a new threshold, a changed bar
   form, a baseline discrepancy) → stop, report neutrally, wait. Never
   self-iterate past an ambiguity.
6. **All candidate results are analysis-tier by construction.** No workbench
   number is a headline number; headline numbers come only from a certified
   checkout per the exp-eval rules.

## 2. Paths (absolute — cache keys are `abspath|mtime|size`-derived)

```bash
WB=$LAB/diffusion-research/.claude/worktrees/metric-workbench          # this worktree
EV=$LAB/diffusion-research/.claude/worktrees/eval-v3-spec              # certified artifacts (RO)

CORPUS_ROOT=$EV/data/processed/transitions_std121                      # 223 std videos + manifest (RO)
SHARED_CACHE=$EV/outputs/eval/cache                                    # warm dino_arr_*/tracks_* (RO)
BASELINE_DIR=$EV/outputs/eval/certification/3.0.0-draft.8              # analysis/ + exam/ (RO)
RECORD_DIR=$EV/outputs/eval/certification/3.0.0                        # regrade record (RO)

WB_CACHE=$WB/outputs/eval/workbench_cache                              # flow_*/null_*/zca (RW, new tags)
WB_OUT=$WB/outputs/eval/workbench                                      # per-rung run outputs (RW)
```

**Why these exact paths matter:** the 952 MB warm cache was keyed against the
`eval-v3-spec` worktree's video paths and mtimes. Copying the videos, or
reading them from any other checkout, silently misses the cache and forces a
full GPU re-extraction. Always resolve corpus clips against `$CORPUS_ROOT`.
The manifest (`$CORPUS_ROOT/corpus_manifest.json`) is also tracked in this
worktree at the same relative path — use it for class/sidedness facts, but
video PATHS come from `$CORPUS_ROOT`.

## 3. Environment

```bash
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $LAB/envs/diffusion
export PYTHONPATH=$WB/src        # CRITICAL in worktrees: the env's editable
                                 # install points at the main checkout and
                                 # silently shadows worktree code
```

Tests: `cd $WB && PYTHONPATH=$WB/src pytest -q tests/test_transition_eval*.py
tests/test_certify_v3.py` must be green before step 0 completes (proves the
certified package works from this worktree) and stay green throughout.

GPU work goes through Slurm (skills: `cc-slurm`, `exp-submit`, `exp-status`):
`HCESC-{H100,L40S}-*` with matching `--account=hcesc-h100|l40s`, or
cluster-wide `secondary` (4 h cap, ~100 s start). Login nodes: no compute.
CPU-heavy exam iteration: `srun` an interactive/batch CPU allocation rather
than the login node.

## 4. Step 0 — freeze verification (before ANY candidate computation)

Each item commits separately; the freeze is auditable from git history.

1. **Baseline regeneration + assert:** write `workbench/baselines.py` that
   reads `$BASELINE_DIR/analysis/{analysis.json,distance_matrices.npz}`,
   recomputes every number in RUNBOOK §B (acc, d, coverage, misretrieved per
   metric), verifies the npz sha256, and writes
   `$WB_OUT/step0/baselines.json`. Any mismatch with §B → STOP, report.
2. **Key alignment:** assert the npz `keys` array == sorted corpus keys from
   the manifest — candidate matrices must share the incumbent's row order.
3. **Warm-cache probe:** load all 223 bundles via
   `pipeline.process_video_file(path, cache_dir=$SHARED_CACHE, …,
   need_frames=False)` — must complete with zero decodes/GPU (all cache
   hits). A miss means paths are wrong; STOP before burning GPU.
4. **Bitwise round-trip:** rebuild the `m1a__v3_sided` distance matrix from
   the warm bundles with deployed code (`certify/exam.appearance_distance_matrix`)
   and compare to the npz copy. Must match bitwise (or report max |Δ| and
   stop for owner review if not exactly 0).
5. **Stratum backfill (RUNBOOK A4):** compute incumbent `m1b_camera`
   within-stratum recalls from the frozen npz using the workbench's own
   stratum-grouping code; record into `baselines.json` as the Phase 1
   targets.
6. **`workbench/gates.yaml`:** encode every numeric gate from the RUNBOOK
   (acceptance-test thresholds §3.4: corr ≥ 0.9, amp err ≤ 10%; definedness
   gates §3.2/§3.3; min-D 5th percentile §1.2; E1 kill rule §4.1 vs pinned
   73/223 and d 1.522006; adoption deltas §7: d +0.25, misretrieved < 73)
   with `frozen: true`, own commit. The workbench driver REFUSES to run any
   exam whose gates are not frozen, and refuses E2/E3 unless E1's recorded
   verdict is a pass — same reflex as `certify/exam.py`'s frozen-bars check.

## 5. Code layout and reuse seams (verified against the certified tree)

New code lives in `src/diffusion/transition_eval/workbench/` only:

```
workbench/
  RUNBOOK.md OPERATIONS.md          # this pre-registration (frozen)
  gates.yaml                        # frozen numeric gates (step 0)
  baselines.py                      # step-0 verifier
  whitening.py anchors.py           # §1.1 ZCA · §1.2 e_A/e_B, D dist, min-D
  hubness.py                        # §1.4 — DOES NOT EXIST anywhere yet; new code + unit tests
  flowcache.py                      # RAFT/SEA-RAFT extraction + flow_* cache (own tag)
  m1b_flow.py m1c_flow.py           # §3.2/§3.3 descriptors
  nulls.py                          # §4.0 rendered-lerp per clip + embedded curves (own cache tag)
  e1_delta.py e2_gamma.py e3_gram.py# §4.1/§4.3/§4.4
  acceptance.py                     # §3.4 probes (reversal + injected trajectory)
  run_workbench.py                  # driver: bundles → descriptors → exam → report; enforces gates.yaml
  reportgen.py                      # per-rung figures + workbench explorer (do NOT edit certify/figures.py
                                    # or certify/explorer.py — they hardcode the six certified metric ids)
```

Deployed functions to IMPORT (never copy):

- `pipeline.process_video_file(…, need_frames=False)` — warm bundle load
  (feats `[T,768]` L2-normed CLS, profile, tracks) with zero GPU.
- `s_structure.core_mask_v3(profile, sidedness)` — the S-mask; core frame
  indices = `np.flatnonzero(mask)`.
- `certify/probes.endpoint_vecs(bundle)` — e_A/e_B exactly as S defines them
  (whiten AFTER, via workbench ZCA).
- `report.retrieval_eval(D, labels)` — THE frozen exam kernel (LOO 1-NN,
  Cohen's d, NaN = undefined, coverage accounting). Candidate matrices:
  symmetric `[223,223]` float, NaN for undefined pairs/rows — definedness
  discipline (§1.5) falls out of the kernel's NaN handling; report coverage
  next to accuracy always.
- `certify/diagnostics.per_clip_rows / class_distance_matrices / clip_tags`
  — margins, class distances, tag strata.
- `certify/probes.reversed_cam / build_reversed_video / grade_reversal`
  patterns for §3.4's reversal probe; `controls.py`'s lerp construction for
  §4.0's rendered nulls (reuse the deployed alpha-blend, don't reinvent).
- `tests/test_certify_v3.py::fake_bundle` — synthetic bundle factory for
  unit-testing every new module without corpus or GPU.

New caches mirror the `features.feature_cache_path` pattern: own filename
prefix + own `CACHE_TAG` in the key (e.g. `flow_<sha>.npz` tag `raft-v1`,
`null_<sha>.npz` tag `lerpnull-v1`), stored under `$WB_CACHE`.

## 6. Execution order and compute shape

Front-load ALL GPU into one cache-build job; afterwards every descriptor
variant, exam, ablation, and acceptance test is CPU-on-cached-arrays
(seconds–minutes per candidate — the design loop never waits on Slurm again).

1. **Step 0** (§4 above) — CPU, commits the freeze.
2. **Shared infra + unit tests** — whitening/anchors/hubness on
   `fake_bundle`s; commit.
3. **GPU cache-build job** (single sbatch, L40S or `secondary` H100, well
   under 4 h): flow for 223 clips @ ~320 px adjacent pairs → `$WB_CACHE`;
   render 223 lerp nulls + DINO-embed their curves → `$WB_CACHE`; fit + persist
   ZCA and the D distribution. While it queues/runs: implement descriptors and
   acceptance tests against synthetic bundles.
4. **Phase 1 (motion):** acceptance tests (§3.4) → BOTH pass → exam via
   frozen kernel → hubness + definedness + stratum recalls vs backfilled
   baselines → verdict per §3.6 → committed report.
5. **Phase 2 (appearance):** E1 FIRST (§4.1 kill test, head-to-head vs
   pinned incumbent), E0 plots ride along → kill rule verdict → only on pass:
   E2 → (only if E2 adds over E1) E3 → ablations (§4.5) on the surviving
   rung → committed reports.
6. **Final:** `WORKBENCH_REPORT.md` — a neutral data package: every
   candidate vs pinned baselines, coverage next to accuracy, hubness
   verdicts, prediction checks (§5), each §7 adoption condition computed as
   a pass/fail FACT, and an explicit list of what was NOT run and why (kill
   rules honored are results, not failures). NO adoption recommendation, NO
   interpretation, NO strategy — the report ends at the facts; the catch is
   made in owner-side review (§8).

Phases 1 and 2 may interleave (both are cached-corpus work); the cache-build
job serves both.

## 7. Reporting discipline

- **Every rung commits:** code + `$WB_OUT/<rung>/` results (distance matrix,
  retrieval_eval json, hubness stats, definedness report, margins, figures) +
  a short md record stating verdict against the pre-registered gate. Negative
  results get the same quality of write-up as positive ones — E1 dying is a
  paper-grade finding per §9.
- **Neutral tone, no proposals in records.** Findings and verdicts against
  pre-registered rules only. Improvement ideas go in a separate
  `IDEAS_NEXT_CYCLE.md`, never into a verdict.
- **No composite scores, no silent drops:** a failing clip is an ERROR/
  undefined row that is counted, never skipped quietly.
- **Trust-map awareness:** class-level claims restricted to the n≥4-eligible
  scope (RUNBOOK §6); untrusted cells carry `†`.
- When blocked (owner decision, cluster failure, gate ambiguity): commit
  state, write the blocker into the running report, stop that track, continue
  any independent track.

## 8. Division of labor — executor reports, reviewer concludes

The executing agent's own evaluations are exactly the MECHANICAL ones:
computing gate verdicts against frozen `gates.yaml` numbers, running the
frozen exam kernel, checking §5 predictions as stated. Computing and stating
those pass/fail facts is reporting, not judgment — do it fully and precisely.

What the executor NEVER does, in any record, report, or commit message:

- draw conclusions beyond the computed verdicts ("this suggests…", "the
  program looks promising/dead…", "we should…");
- recommend adoption, rejection, or next steps;
- reframe, soften, or editorialize a result — a kill-rule fire is stated as
  the rule text + the numbers, nothing more;
- strategize about the paper, the v3.1 re-cert, or future cycles.

The standing order that governed the draft.8 certification applies verbatim:
**neutral report only, zero proposals — joint inspection is next.** The §7
adoption call, all synthesis, and all strategy belong to owner-side review
(the owner plus a separate reasoning session), which reads
`WORKBENCH_REPORT.md` after this branch's work is complete. Reports must
therefore be complete enough that the reviewer never needs a re-run to make
the call — coverage, hubness, margins, and definedness sit next to every
headline statistic, always.

`IDEAS_NEXT_CYCLE.md` is the single sanctioned non-neutral file: a parking
lot the executor may append to, clearly marked non-authoritative, never
referenced by any record or verdict.
