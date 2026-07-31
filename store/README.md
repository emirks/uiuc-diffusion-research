# The artifact store — `$LAB/diffusion-research/store`

**One place, three shelves: `runs/` (training) → `gens/` (generation) → `evals/` (scoring).**
Created 2026-07-30. `STORE=$LAB/diffusion-research/store`. Code lives in git; artifacts live here;
campaign narrative stays in `misc/<campaign>/` (also inside the repo since the 2026-07-30 reorg).
The store is the source of truth — other locations may symlink INTO it, never hold copies.
Git tracks the store's *metadata* (README, INDEX, `meta.yaml`, `grid.jsonl`); videos, checkpoints,
and score files are gitignored like the rest of `outputs/`.

```
store/
  INDEX.md            ← the ledger: one line per entry below. Update it when you add an entry.
  datasets/<id>/      immutable dataset roots (or a symlink stub to a legacy location) + meta.yaml
  runs/<run_id>/      one training run:  meta.yaml · config.yaml · checkpoints/ · NOTES.md?
  gens/<gen_id>/      one generation batch:  meta.yaml · grid.jsonl · videos/*.mp4
  evals/<eval_id>/    one scoring pass:  meta.yaml · <arm>/<label>/{items.jsonl,results.json}
```

## The contract (all of it)

1. **Every entry is a directory with a `meta.yaml`.** Minimum keys: `id`, `created`, `machine`,
   `inputs` (the upstream store ids this was made from), `source` (campaign/script that made it).
   Everything else is per-shelf (see the seeded entries for live examples).
2. **IDs are snake_case slugs, unique per shelf, chosen at creation.** A gen is named after its
   arm (`ctt_v2_leaky`); suffix with `__<grid>` only if the same arm is generated again on another
   grid. Evals: `<name>__<machine>__<YYYY-MM-DD>`.
3. **An eval is scored on ONE machine, recorded in `meta.yaml`.** Never merge rows from two
   machines into one eval entry — measured 2026-07-30: v4 does not reproduce eps↔DeltaAI at the
   0.005 bar. Record the *measured* sha256 of the reference artifact, not the declared constant.
4. **A gen pins its adapter by `run_id` + `step` (+ sha256 of the checkpoint file)** and carries
   the exact rendered rows it used in `grid.jsonl` (prompt text included). Arm identity must never
   live only in a parent directory name — that is how the four identically-named 304-clip sets
   almost got merged.
5. **Checkpoint retention:** `runs/<id>/checkpoints/` holds the *shipped* step(s) and the final
   step only. Intermediates stay in the training scratch dir and die when the campaign closes,
   unless `meta.yaml` lists a reason to keep them.
6. **External models** (base weights, third-party adapters) stay in `$LAB/cache/`; a run entry for
   them is a stub: `meta.yaml` + `checkpoints/` symlink into the cache, `external: true`.
7. **Wiring, not copying.** The viewer, repo `outputs/`, and campaign dirs reach store content via
   symlinks. If you find yourself `cp`-ing out of the store, stop.

## How new work flows through it

- **Train** → point the trainer's `output_dir` at scratch as usual; at campaign close, `mkdir
  $STORE/runs/<id>`, move the kept checkpoint(s) + the exact `config.yaml` in, write `meta.yaml`,
  symlink the old location back if anything references it.
- **Generate** → `LADDER_OUT_ROOT=$STORE/gens/<id>/videos python eval_ladder/run_gen.py …`
  (the `relative_to` crash for out-of-repo roots was fixed 2026-07-30); drop `grid.jsonl` +
  `meta.yaml` beside it.
- **Score** → `LADDER_SCORES=$STORE/evals/<id>` (or `--out-root`); the scorer's native
  `<label>/{items.jsonl,results.json}` layout is stored as-is, one subdir per arm.
- **View** → `eval_ladder/viewer/build_runs.py` entries point at store paths; its existing
  `ensure_external_media()` symlink machinery serves them.
- **Record** → one line in `INDEX.md`, and the campaign dossier references store ids.

## What this replaces (context)

Before 2026-07-30 artifacts were scattered: checkpoints in `misc/ctt_v2_training/` and repo
`outputs/training/ladder2/`, videos across repo `outputs/videos/*` and three `misc/` campaign
dirs, scores in `misc/refvfx_baseline/eval/scores/`, with identity carried by parent paths and
hand-edited viewer lists. The five arms of the refVFX comparison (`ic_gen`, `ctt_v2`,
`ctt_v2_leaky`, `refvfx_A`, `refvfx_B`) were migrated in as the seed content; legacy campaigns
before that stay where they are, frozen.

Same reorg moved `misc/` and the four `LTX-2-*` trainer checkouts from `$LAB` into this repo dir,
with **symlinks left at the old `$LAB` locations** — those bridges are load-bearing: the
`envs-aarch64/ltx2` and `envs-aarch64/refvfx` venvs hold editable installs pointing at
`$LAB/LTX-2-official/packages/*` and `$LAB/misc/refvfx_baseline/code/refVFX_trainer`. Do not
delete the bridges without re-installing those packages.
