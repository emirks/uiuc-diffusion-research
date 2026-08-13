# The artifact store — `$LAB/diffusion-research/store` (contract v2)

**One place, five shelves: `runs/` (training) → `gens/` (generation) → `evals/` (scoring), fed by
`datasets/` and `prompts/`.** Created 2026-07-30; contract v2 since the 2026-08-13 migration
(`MIGRATION.md` — old gen ids resolve forever via `gens/_legacy/`). `STORE=$LAB/diffusion-research/store`.
Code lives in git; artifacts live here; campaign narrative stays in `misc/<campaign>/`.
The store is the source of truth — other locations symlink INTO it, never hold copies.
Git tracks the store's *metadata* (README, INDEX, ARMS, MIGRATION, `meta.yaml`, `meta.v1.yaml`,
`grid.jsonl`, `config.yaml`); videos, checkpoints, and score files are gitignored.

```
store/
  INDEX.md               ← the ledger: one line per entry. Update it when you add an entry.
  ARMS.md                ← canonical arm registry: slugs, variants, harness_arm aliases, prompt families
  MIGRATION.md           ← the frozen v1→v2 map (2026-08-13)
  datasets/NNN_<slug>/   immutable dataset roots (or symlink stub) + meta.yaml
  prompts/NNN_<slug>/    canonical rendered prompt families: ARM-FREE grid.jsonl + meta.yaml (sha-pinned)
  runs/NNN_<slug>/       one training run:  meta.yaml · config.yaml · checkpoints/ · NOTES.md?
  gens/NNN_<arm>/KK_<variant>__<machine>/   one generation batch of one arm-variant:
                         meta.yaml · meta.v1.yaml? · grid.jsonl · videos/*.mp4 (flat, REAL files) · scores -> its eval arm-dir
  evals/NNN_<name>__<machine>__<date>/      one scoring pass:
                         meta.yaml · <harness_arm>/<label>/{items.jsonl,results.json}  (label = --label shard, c0..c7)
```

## Naming (v2)

- **Arm** = model/adapter lineage, canonical slug per `ARMS.md` (`ctt_v3`, not a campaign nickname).
- **Variant** = prompt setting: **`neutral`** (the arm's leak-free prompt) or **`effect`** (+ the
  shared clause); controls suffix it (`neutral_shufcode`). **Machine** always suffixes the subentry
  (`__cc`/`__eps`/`__dai`) — cross-machine drift is measured and identity-bearing.
- **`harness_arm`** is the frozen stamp inside scored artifacts (item_ids, `items.jsonl` `arm`
  field). Never renamed without re-scoring; joins use it, humans use canonical. New arms stamp
  `<arm>_<variant>` so the two coincide.
- `KK` = 2-digit inner seq in creation order (never reused): the arm dir's `ls` is its variant
  timeline, the shelf's `ls` is the arm timeline.

## The contract (all of it)

1. **Every entry is a directory with a `meta.yaml`.** Gen subentries carry: `id, seq, shelf, arm,
   harness_arm, variant, machine, created, inputs {run, step}, prompt_family, prompt_sha, grid_rows,
   videos` (+ `void:` where a control is unusable). Migrated ones keep the v1 meta as `meta.v1.yaml`.
2. **Entry dirs are numbered, numbers never reused; `ls` IS the timeline** — arm level and variant
   level. An eval's slug is `<name>__<machine>__<YYYY-MM-DD>`.
3. **An eval is scored on ONE machine, recorded in `meta.yaml`**, with a REQUIRED `arms_scored:`
   block mapping each arm subdir → `{gen: <gen id>, run: <run id>, rows: N}`. Record the *measured*
   sha256 of the reference artifact, not the declared constant.
4. **A gen pins everything**: adapter by `run_id` + `step` (+ checkpoint sha256), code by a
   `trainer:`/`code:` line (checkout · commit), prompts by `prompt_family` + `prompt_sha`, and the
   exact rendered rows in `grid.jsonl`. Arm identity lives in meta (`harness_arm`), never only in a
   path.
5. **Prompts have ONE source.** `prompts/` families are sha-pinned and arm-free;
   `eval_ladder/stamp_rows.py` derives an arm-stamped registry from a family — hand-written
   registries are a contract violation. `eval_ladder/prompts.py` remains the only renderer.
6. **Checkpoint retention:** shipped step(s) + final only; intermediates and training-state blobs
   die when the campaign closes. After close-out sha-verify, offsite/misc duplicate copies are
   DELETED in the same step.
7. **External models** stay in `$LAB/cache/`; their run entry is a stub (`external: true`).
8. **Wiring, not copying — and media lives HERE.** `videos/` holds real flat files
   (`<item_id>__s<seed>.mp4`); `outputs/`, viewers, and campaign dirs reach them via symlinks INTO
   the store. If you find yourself `cp`-ing out of the store, stop.
9. **Entries are immutable.** Registering = numbered dir + meta + INDEX row + CHANGELOG, **one
   commit, at close — not days later**. A re-run/re-score/fix is a NEW subentry with the next KK.
   (The 2026-08-13 migration was a one-time versioned event, not precedent.)

## How new work flows through it

- **Train** → scratch/offsite as usual; at close: `mkdir $STORE/runs/<id>`, move kept checkpoint(s)
  + exact `config.yaml`, write meta (trainer pin), sha-verify, delete duplicates, symlink the old
  location back if referenced.
- **Generate** → registry from `stamp_rows.py`; run with
  `--out-root $STORE/gens/NNN_<arm>/KK_<variant>__<machine>` (videos land flat in `videos/`);
  at close `scripts/store_register.py gen <subentry>` writes `grid.jsonl` + autofilled meta + the
  INDEX row.
- **Score** → scorer `--out-root $STORE/evals/<id>/<harness_arm>` per arm (per-arm out-roots — the
  shared-outroot collision is a known failure), `--label c<K>` shards. At close, materialize the
  eval↔gen 1:1: `ln -sfn <eval arm-dir> <gen subentry>/scores` (latest scoring wins; history in
  `arms_scored:`). Prompts: the shelf has exactly TWO sources — base/external variants are
  `stamp_rows.py` transforms with derived shas recorded in the family metas.
- **View** → `eval_ladder/viewer/build_runs.py` entries point at store paths; `ensure_external_media()`
  symlinks serve them.
- **Record** → one INDEX line; campaign dossiers reference store ids.
- **Check** → `scripts/store_fsck.py` validates schema, counts, and prompt shas; run it at every
  registration.

## What the store is NOT

Artifacts + provenance, not tooling. Tools live in git: trainers in `src/LTX-2-official` (+ linked
worktrees, one branch each), generation in `eval_ladder/run_gen.py` (+ `job_gen.sbatch`), scoring in
`src/diffusion/transition_eval` (v4 via the `eval-v4-cert` worktree), the viewer in
`eval_ladder/viewer/`. A store entry's meta names the code and commit that produced it, so
**entry + git checkout = the full reproduction recipe**.

**Path prefix rule (both clusters, one Taiga filesystem):** use `/taiga/illinois/...` in anything
absolute — it resolves on CC *and* DeltaAI. `/projects/illinois/...` is CC-only. See the `deltaai`
skill and the cc-cluster-layout memory.

## History

v1 (2026-07-30) seeded the five-arm refVFX comparison and retired the `$LAB`-level bridges (parked
at `$LAB/.retired-bridges/`). v2 (2026-08-13) restructured `gens/` arm-first, added `prompts/` +
`ARMS.md`, backfilled the nine stub gens, and remediated the drift documented in
`misc/2026-08-13_store_restructure/PROPOSAL.md`. Legacy campaigns before the store stay where they
are, frozen.
