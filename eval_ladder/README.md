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

### Cell-assignment consistency across tiers (audited 2026-07-23)

Is a task's ontology cell the same for every tier that answers it? Half yes, half tier-relative
— by design, and now verified against the registry:

- **Content (same/cross/foreign) — consistent, always.** It is a pure function of the *task*
  (endpoint class vs donor class), so every tier on a card shares it. Audited: **0 mismatches**
  across all 39 shared specialist+generalist tasks. "Cross uses only test clips of other
  classes" holds for 66/68 cross endpoints; the other 2 (`display_transition_1`,
  `monstrosity_0`) carry a "train" split label but belong to **held-out classes** — no arm ever
  trained on them (the per-arm untrained-content rule), so they are equivalent where it matters.
- **Novelty (seen/unseen/zero-shot) — tier-relative by definition.** It grades the exposure of
  the tier's *variable input*: the **demo** for the generalist, the **endpoint** for the
  specialist (which has no demo). On the 39 shared tasks the two views coincide for 39 of 50
  spec–gen row pairs (unseen/unseen); the 11 exceptions are all **G-memo-probe** — same task,
  *trained* demo — where the misalignment is deliberate: it IS the memorisation probe. So a
  card can legitimately sit in "unseen" for its specialist row and "seen" for its memo-probe
  generalist row; the viewer places each row by its own tier's novelty.

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

## Infrastructure map (versioned like the instrument)

| piece | where | rule |
|---|---|---|
| design version | `VERSION` (semver) | bumps only with `SPEC.md` — see `VERSIONING.md` |
| run records | `reports/v<DESIGN>-R<N>.md` | append-only; **newest VALID\* record = the current result** |
| results viewer | `viewer/build.py` → `outputs/reports/ladder_viewer/index.html` | stable LATEST path; `--freeze <run-id>` writes the immutable per-run copy |
| decision log | `$LAB/misc/ladder2_redesign/DOSSIER.md` | campaign narrative, advisor rulings |

Rebuild the viewer any time (idempotent, ~10 s):

    python eval_ladder/viewer/build.py
    # serve from the repo root so the videos resolve:
    python3 -m http.server 8890   ->  http://localhost:8890/outputs/reports/ladder_viewer/index.html

## Side-by-side suitability (honest status, 2026-07-23)

The viewer's card unit is a **task** = (transition donor, endpoint, sidedness); every tier that
answered that task appears in one aligned row. Coverage today is partial **by design history,
not by accident**: specialist cells (SP-*) and generalist cells (G-*) drew their endpoints
independently, so only the overlap carries both tiers on one endpoint. Zero-shot donors are
held-out classes — no specialist can exist for them, structurally. The two clean baselines
(`base_prompt`, `base_cond`) were designed in v2.1.0 but their generation lane was stopped
before completion (owner call); where they are missing, cards fall back to an old no-reference
base twin when one exists, and the headline marks margins "baseline pending" rather than
silently substituting the base+demo copier (which is NOT a baseline — it reproduces the demo).

To make every non-zero-shot task fully 4-tier in the next design: draw SP and G cells from one
shared endpoint roster, and generate the two clean baselines on that same roster (v2.1.0's
`add_baselines.py` already dedups them to one video per endpoint — ~194 videos, ~4 GPU-h).

## TODO — next design (v3.0.0): CTT tasks, all tiers by construction

*Owner directive, 2026-07-23. This is the committed shape of the next MAJOR design bump; held
here so it survives until that campaign starts.*

The v2 defect to fix is structural: SP-* and G-* cells drew their endpoints independently, so
only 39/177 tasks are fully side-by-side. v3 inverts the build order — the **task roster comes
first, the arms are projections of it**:

1. **CTT task** (Creative Transition Transfer) = `(endpoints, target transition)` — the atomic
   unit. Endpoints are sidedness-typed (one/two); the target transition is a donor class (or a
   held-out class for zero-shot tasks).
2. **Every task generates on every tier it can support**, from one registry:
   ① prompt only · ② prompt + endpoints · ③ specialist (exists iff the donor is a trained
   class) · ④ generalist (IC demo). Same prompt, same conditioning windows, same seeds — the
   tier is the *only* variable. All four scored by the same instrument, same pool rule.
3. **Enumerate the greatest space first, then choose the subspace**: the full grid is
   endpoints × donors × sidedness × novelty(seen/unseen/zero-shot) × content(same/cross/foreign).
   Enumerate it exhaustively in `build_registry.py`, then select a budgeted subspace that keeps
   the process healthy — balanced per-donor and per-cell n (pre-register minimum n per claim
   cell), ≥2 seeds, hardest cells first, fit anchors kept tiny. Selection rule is written in the
   registry builder and pre-registered, never hand-picked.
4. Deliverable: near-every card in the viewer shows all four tiers; the only structural blanks
   are specialist × zero-shot (impossible by design) and mismatched-demo controls.

Cost note from v2: the clean-baseline half of this is already scaffolded (`add_baselines.py`,
video dedup per endpoint) — the missing piece is drawing all cells from the shared roster.
