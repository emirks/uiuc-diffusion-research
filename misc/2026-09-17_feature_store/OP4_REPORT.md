# OP4 REPORT — competitor-lens pass + aesthetic + gate check

Op-4, branch `feature-store`. All code + tests + sbatch written and verified on CPU (login node
`gh-login01`). **No GPU jobs, no sbatch/srun submitted** — the coordinator submits. Python
`$LAB/envs-aarch64/ltx2/bin/python`.

## Deliverables (all committed, pushed; `ahead=0 behind=0`)
| file | what |
|---|---|
| `scripts/lens_pass_gridv3.py` | drives the FROZEN `their_metrics/score_batch.py --store` over the 19 grid-v3 variants (7242 gens); sharded, resumable; `--collect` merges per-gen JSON → per-arm `rows.jsonl` + `meta.yaml` + one INDEX line; `--plan` dry-run |
| `scripts/aesthetic_from_store.py` | LAION head (768→1024→128→64→16→1) on `clip_l14@r224`; merges an `aesthetic` column into the lenses `rows.jsonl` (idempotent; NaN+`missing` when absent) |
| `scripts/lens_gate_check.py` | recomputes lenses for eligible VAP author-native rows vs `rows_v3` and reports per-lens max\|Δ\| with the coordinator tolerances; exit 1 on a real port error |
| `misc/2026-09-17_feature_store/jobs/lens_pass.sbatch` | GPU array job for the lens pass (copies `extract.sbatch` flags + `--time=02:00:00`, env `NSHARDS`) |
| `tests/test_lens_pass_gridv3.py` | 16 tests: arm map, id reconcile, collect/join, aesthetic finite, gate comparator (all pass) |

Eval id (next free NNN): **`039_lenses_gridv3__dai__2026-09-18`** (scripts default to it; pass `--eval-id`
if 039 is taken before you run).

## The commands the coordinator runs

**Prereqs:** wave-2 extraction (`clip_b32` 3171050, `videoprism` 3171051, `raft_mag` 3171052,
`clip_l14` 3171053) must land for the population; the gate job (`fs_gate` 3170875) fills VAP + its 58
refs with `clip_b32/videoprism/raft_mag`. `cotracker3` is already 100% (7242/7242 gens + refs).

All commands assume:
```bash
export LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
source $LAB/envs-aarch64/activate
export HF_HOME=$LAB/cache/huggingface TORCH_HOME=$LAB/cache/torch HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $LAB/diffusion-research
export PYTHONPATH=$PWD/src
```

### (a) Gate check — the kill switch (run FIRST, after `fs_gate` 3170875 lands)
CPU; needs GPU only if features are missing (then it reports fewer eligible rows). Exit 1 = a real
port error → `scancel` the wave-2 lens work and investigate.
```bash
python scripts/lens_gate_check.py --device cpu
```
Expected on PASS: per-lens `max|Δ|` printed; embedding lenses `≤ 1e-3` (report bar 1e-4),
`det_motion_fidelity` reported separately at `≤ 1e-2`, `RESULT: PASS`, exit 0. Before `fs_gate` lands
it prints `eligible=0` and exits 0 (nothing to check) — re-run once the VAP+refs features are present.

### (b) Lens pass — GPU array (after wave-2 extraction lands for the population)
`NSHARDS` MUST equal the array width. 8 shards ≈ 905 gens/shard; store hits mean the only per-gen GPU
work is the fresh input-frame CLIP+VideoPrism embed, so a shard finishes well inside 2 h. Resumable
(skip-if-exists per gen) — safe to requeue/re-submit.
```bash
sbatch --array=0-7 --export=ALL,NSHARDS=8 misc/2026-09-17_feature_store/jobs/lens_pass.sbatch
# optional explicit id: add ,EVAL_ID=039_lenses_gridv3__dai__2026-09-18 to --export
```
Watch: `store/evals/039_lenses_gridv3__dai__2026-09-18/<arm>/rows/*.json` fill (7242 total).
Coverage/plan without models: `python scripts/lens_pass_gridv3.py --plan --shard 0 --num-shards 8`.

### (c) Collect + aesthetic + INDEX (CPU, login, one shot, after the array is done)
```bash
python scripts/lens_pass_gridv3.py --collect          # per-arm rows.jsonl + meta.yaml + INDEX line
python scripts/aesthetic_from_store.py                # merge the `aesthetic` column (needs clip_l14)
```
`--collect` is idempotent; the INDEX line is appended once. Re-run `--collect` then
`aesthetic_from_store.py` if you re-score. `rows.jsonl`/`meta.yaml` are store artifacts (NOT committed).

## rows.jsonl schema (one row per gen; joins on `(item_id, seed)`)
`item_id` (GRID id), `seed`, `arm` (harness arm, e.g. `vap_author_native`), `gen` (repo-rel path),
`motion_smoothness`, `videoprism_sim_ref`, `videoprism_sim_input`, `clip_sim_ref`, `clip_sim_input`,
`det_motion_fidelity`, `dynamic_degree_mean_mag`, `impl_sha`, `warnings`, and `aesthetic` (added by
step c). This is exactly the set Op-5's Table B/C expect (`videoprism_sim_ref, det_motion_fidelity,
clip_sim_ref, dynamic_degree_mean_mag, motion_smoothness, aesthetic`).

## The `(item_id, seed)` reconciliation (explicit, as required)
`score_batch`'s own `item_id` is the gen **filename stem** (`…__ref_…__s42`); Op-2/Op-3 key on the
**grid** `item_id` (no `__s<seed>`) + `seed`. `--collect` joins each scored gen to Op-2's
`per_gen.jsonl` (globbed from `store/evals/028_*/<arm>/` and `030_*/<arm>/`) **by gen path**
(`gen_video` ↔ `gen`, both normalized repo-relative) and takes the grid `(item_id, seed)` from there.
No per_gen match → fallback strips the trailing `__s<seed>` token and records a `no_pergen_join`
warning. The harness arm comes from each variant's own `meta.yaml` `harness_arm` field — the same
frozen stamp Op-2/Op-3 use — so arms line up across the three inputs (verified: 19 unique arms match
the 038 hand-off arms).

## Verification done (CPU / fixtures / dry-runs)
- **`--plan` dry-run** enumerated 19 arms, 7242 gens, shard 0/8 = 906; arm names match the 038 hand-off
  entry; store-coverage sample shows `cotracker3=hit`, `clip_b32/videoprism/raft_mag=MISS` (wave 2 not
  yet landed, as expected).
- **Gate check** now: `total=366 eligible=0 … Exiting 0` (features absent; correct behavior).
- **`tests/test_lens_pass_gridv3.py`**: `16 passed in 69.11s`.
- **collect smoke** (temp dir): well-formed `meta.yaml` (038 shape) + `rows.jsonl` with the exact keys;
  INDEX append idempotent (`added1/added2 = True False`), lands in the `## evals` section, real
  `store/INDEX.md` untouched.
- **aesthetic smoke** (mini-store fixture): present → finite (`5.37`); absent → `nan` +
  `missing clip_l14@r224 (aesthetic)`; idempotent (missing-warning count stays 1). Head shapes verified
  at load; `manual-vs-fn agree` and the head re-normalizes internally (unnormalized input scores equal).
- `Models(store)` constructs in `0.02s` with all backbones lazy (`None`); `score_one` inner imports
  (`common`, `clip_sim`, `motion`) resolve; `impl_sha = 8a808635e8a08cb4`.

## Notes / caveats for the coordinator
- **impl_sha**: the recompute path is `8a808635` (P3b `--store`); the frozen `rows_v3` are `d63935f4`
  (legacy in-process embedders). The metric ARITHMETIC is byte-identical between them (only
  `score_batch.py` bytes changed), so the gate should pass within tolerance; the sha difference is
  expected and printed.
- **No login-node extraction.** The feature store keys features BY VIDEO PATH (next to the video),
  independent of `--store-root`, so a real score/gate on the login node with a *missing* namespace
  would extract on CPU and write a `host=gh-login01` file that the GPU job then skips (host-mix /
  cross-machine drift). I therefore validated end-to-end scoring via fixtures + dry-runs only; the real
  scoring is the GPU job (b) and the gate (a) run only on **eligible** rows (features already present),
  which read the store and never trigger a write. The lens pass on GPU is self-healing for any missing
  reference-clip feature (extracts on the correct host).
- **score_batch not edited** — no flag was missing; `lens_pass_gridv3.py` imports its `score_one` /
  `Models` / `StoreLenses` / `impl_sha` so the arithmetic + impl_sha are the single frozen source.
- **Rows with a missing input-frame png** (some seen/unseen endpoints lack
  `misc/refvfx_baseline/frames/<endpoint>__first.png`) still score every non-input lens; `clip_sim_input`
  / `videoprism_sim_input` are null and a `missing input_frame …` warning is recorded (counted in the
  meta `missing_input_frame`). Never crashes.
- **Not done tonight** (out of Op-4 scope / dependencies): the real 7242-gen scoring, the real collect,
  the real aesthetic merge, and a PASS/FAIL gate number — all require wave-2 + `fs_gate` GPU features to
  land, which the coordinator submits. Commands (a)/(b)/(c) above are the complete runbook.
