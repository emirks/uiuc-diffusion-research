# ladder2 — the clean transition eval campaign

**`registry.jsonl` is the single source of truth for the eval ladder.** Every generation, its
cell, its GT pool, its % type and its base twin are derived there, once, from three frozen
inputs. Nothing downstream re-derives a fact. `REFERENCE.html` is the human-readable face of
the same design (serve it from the repo root so the sample videos play); the campaign decision
log is `$LAB/misc/ladder2_redesign/DOSSIER.md`.

## Question

Does a transition learned from a corpus *transfer* — to new content, to new demos of a known
transition, and to transitions the model never trained on — and how much of that is capability
versus lookup? Measured with leak-free prompts, so nothing the model produces can be credited
to the prompt describing the outcome.

## Design

Two axes, one grid.

- **Reference novelty** (training exposure of the *transition source*): `seen` (the exact demo
  was trained) → `unseen` (a new demo of a trained class) → `zero_shot` (a held-out class).
  Specialists have no reference axis — their transition is baked into the weights.
- **Content** (endpoint vs transition class): `same` / `cross` / `foreign` (DAVIS).

Endpoints are always untrained content (test band, held-out, or DAVIS), strictly
sidedness-matched (one↔one, two↔two) — except the two fit anchors, which deliberately use
train endpoints to measure the memorisation ceiling.

Two gaps fall out of the grid: `G-memo-probe − G-unseen-same` = demo-instance memorisation
(endpoint novelty held fixed), and `G-unseen-* − G-zs-*` = class generalisation.

## Setup

| | |
|---|---|
| split | `data/processed/transitions_std121/split_v1.2.json` (frozen, sha `c694659d`), 29 held-in / 10 held-out |
| prompts | rendered, never authored: `"{S1}. sksz."` / `"{S1}. sksz. {S2}."` — the outcome half of every caption is dropped |
| specialists | 11 × c2v LoRA, rank32/α32 attn, lr 1e-4, 2000 steps, ckpt/250 |
| generalist | 1 × IC-LoRA, rank32/α32 attn+FFN, lr 2e-4, 5000 steps, ckpt/500, 26 classes / 385 pairs |
| bleed fix | conditioning anchors VAE-encoded in **isolation** so train == inference (suffix only; prefix is clean by causality) |
| trainer | `$LAB/LTX-2-cond-bleed-fix` @ `cond-bleed-fix` |
| instrument | v4 (`.claude/worktrees/eval-v4-cert`), pool-% of the class GT ceiling |

## How to run

```bash
# 0. token (base-model inertness probe) -> arms.yaml
sbatch eval_ladder/token_probe/job_probe.sbatch
python eval_ladder/train/token_verdict.py --apply

# 1. build (no GPU)
python eval_ladder/train/inventory.py         # rosters + latent coverage
python eval_ladder/build_registry.py          # registry.jsonl + 8 seatbelts
python eval_ladder/train/make_configs.py      # 12 training configs
pytest tests/test_ladder2_prompts.py -q

# 2. precompute (GPU)
sbatch --export=ALL,MODE=cond-clean ... eval_ladder/train/job_precompute.sbatch
sbatch --export=ALL,MODE=text       ... eval_ladder/train/job_precompute.sbatch
python eval_ladder/train/assemble_roots.py

# 3. train — pilot first, then the fleet
python eval_ladder/submit.py train --models spec_shadow_smoke,spec_color_rain
python eval_ladder/submit.py train --fleet

# 4. generate + score
python eval_ladder/submit.py gen --arms spec_color_rain --priority P0 --chunks 4
python eval_ladder/run_eval.py --mode plan
sbatch ... eval_ladder/job_score.sbatch
python eval_ladder/run_eval.py --mode report
```

## Outputs

`registry.jsonl` (298 items → 596 generations: P0 432, P1 132, P2 32) ·
`outputs/training/ladder2/<arm>/` · `outputs/videos/ladder2/<arm>/<item_id>__s<seed>.mp4` ·
`outputs/eval/ladder2/<chunk>/items.jsonl` · the report table from `run_eval.py --mode report`.

## The seatbelts (why this ladder is trustworthy)

Every prior defect had one root cause: the same fact written down twice, by hand, with no
conformance check. So each of these is an assert, not a convention.

1. unique `item_id` — no collisions
2. prompts **rendered** by `prompts.render_prompt()`, never authored (training captions come
   from the same call, so train == inference)
3. contamination: no held-out class in the generalist's training set; quarantined clips barred;
   only fit anchors may use content their own arm trained on
4. **keyed join is exact** — a treatment row without its base twin is a hard error, never a
   silent drop (this is the defect that flipped two verdicts in the previous ladder)
5. mask = f(conditioning), never of the class label
6. one `encode_conditioning.py` defines the conditioning window for training *and* inference
7. cell derivation: novelty = f(reference) only; content class-disjointness; zero-shot
   self-reference guard
8. **%-typing firewall**: `%_same` (endpoint class == donor) is headline-eligible;
   `%_proxy` (cross/foreign) is content-capped and ranking-only — its claim is the margin
   Δpp vs base, where the identical cap cancels. `run_eval.py` refuses to mix the two.
