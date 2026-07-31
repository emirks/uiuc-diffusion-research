# The artifact store — `$LAB/diffusion-research/store`

**One place, three shelves: `runs/` (training) → `gens/` (generation) → `evals/` (scoring).**
Created 2026-07-30. `STORE=$LAB/diffusion-research/store`. Code lives in git; artifacts live here;
campaign narrative stays in `misc/<campaign>/` (also inside the repo since the 2026-07-30 reorg).
The store is the source of truth — other locations may symlink INTO it, never hold copies.
Git tracks the store's *metadata* (README, INDEX, `meta.yaml`, `grid.jsonl`); videos, checkpoints,
and score files are gitignored like the rest of `outputs/`.

```
store/
  INDEX.md               ← the ledger: one line per entry below. Update it when you add an entry.
  datasets/NNN_<slug>/   immutable dataset roots (or a symlink stub to a legacy location) + meta.yaml
  runs/NNN_<slug>/       one training run:  meta.yaml · config.yaml · checkpoints/ · NOTES.md?
  gens/NNN_<slug>/       one generation batch:  meta.yaml · grid.jsonl · videos/*.mp4
  evals/NNN_<slug>/      one scoring pass:  meta.yaml · <arm>/<label>/{items.jsonl,results.json}
```

## The contract (all of it)

1. **Every entry is a directory with a `meta.yaml`.** Minimum keys: `id`, `created`, `machine`,
   `inputs` (the upstream store ids this was made from), `source` (campaign/script that made it).
   Everything else is per-shelf (see the seeded entries for live examples).
2. **Entry dirs are numbered: `NNN_<slug>`** — `NNN` = zero-padded seq (max on the shelf + 1,
   never reused), slug snake_case. A gen's slug is its arm (`005_ctt_v2_leaky`); an eval's is
   `<name>__<machine>__<YYYY-MM-DD>`. **`ls` IS the timeline — the highest number is the
   latest**, no separate pointer.
3. **An eval is scored on ONE machine, recorded in `meta.yaml`.** Never merge rows from two
   machines into one eval entry — measured 2026-07-30: v4 does not reproduce eps↔DeltaAI at the
   0.005 bar. Record the *measured* sha256 of the reference artifact, not the declared constant.
4. **A gen pins its adapter by `run_id` + `step` (+ sha256 of the checkpoint file)** and carries
   the exact rendered rows it used in `grid.jsonl` (prompt text included). Arm identity must never
   live only in a parent directory name — that is how the four identically-named 304-clip sets
   almost got merged. **Code is pinned the same way**: a run's `meta.yaml` carries a `trainer:`
   block (checkout · branch · commit · entry script) and a gen's carries a `code:` line naming the
   stack that rendered it.
5. **Checkpoint retention:** `runs/<id>/checkpoints/` holds the *shipped* step(s) and the final
   step only. Intermediates stay in the training scratch dir and die when the campaign closes,
   unless `meta.yaml` lists a reason to keep them.
6. **External models** (base weights, third-party adapters) stay in `$LAB/cache/`; a run entry for
   them is a stub: `meta.yaml` + `checkpoints/` symlink into the cache, `external: true`.
7. **Wiring, not copying.** The viewer, repo `outputs/`, and campaign dirs reach store content via
   symlinks. If you find yourself `cp`-ing out of the store, stop.
8. **Entries are immutable.** `meta.yaml` carries `seq: N` matching the dir prefix. Registering
   = numbered dir + meta + INDEX row + CHANGELOG, one commit. A re-run/re-score/fix is a NEW
   entry with the next number — never overwrite written artifacts or numbers.

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

## What the store is NOT

It is **artifacts + provenance, not tooling**. The tools live in the repo, versioned by git:
trainers in `src/LTX-2-official` (+ its linked worktrees `src/LTX-2-{cond-bleed-fix,ctt-v2-train,
bneck}` — one branch each), generation in `eval_ladder/run_gen.py` (+ `job_gen.sbatch`), scoring
in `src/diffusion/transition_eval` (imported from the `eval-v4-cert` worktree), the viewer in
`eval_ladder/viewer/`. A store entry's `meta.yaml` names the code and commit that produced it, so
**entry + git checkout = the full reproduction recipe** — that is the self-containment contract,
and it is why tools are never copied into entries (copies rot; pins don't).

## What this replaces (context)

Before 2026-07-30 artifacts were scattered: checkpoints in `misc/ctt_v2_training/` and repo
`outputs/training/ladder2/`, videos across repo `outputs/videos/*` and three `misc/` campaign
dirs, scores in `misc/refvfx_baseline/eval/scores/`, with identity carried by parent paths and
hand-edited viewer lists. The five arms of the refVFX comparison (`ic_gen`, `ctt_v2`,
`ctt_v2_leaky`, `refvfx_A`, `refvfx_B`) were migrated in as the seed content; legacy campaigns
before that stay where they are, frozen.

Same reorg moved `misc/` (repo root) and the four `LTX-2-*` trainer checkouts (under `src/`)
from `$LAB` into this repo dir. The `$LAB`-level bridge symlinks that initially kept old paths
alive were **RETIRED later the same day**: the venvs' editable path files
(`envs-aarch64/{ltx2,refvfx}` `.pth` + `direct_url.json`) and the eval_ladder sbatch scripts +
`encode_conditioning.py` were rewritten to the canonical in-repo paths, import-verified, and the
five bridges parked at `$LAB/.retired-bridges/` (restore = move them back). The three
non-official `LTX-2-*` dirs are **linked git worktrees** of `src/LTX-2-official`; their gitdir
wiring was repaired on 2026-07-30 (it had pointed at dead `/projects` paths since the migration).

**Path prefix rule (both clusters, one Taiga filesystem):** use `/taiga/illinois/...` in anything
absolute — it resolves on CC *and* DeltaAI. `/projects/illinois/...` is CC-only, and on DeltaAI
`/projects/<code>` is a *different* filesystem (Delta project space). See the `deltaai` skill and
the cc-cluster-layout memory.
