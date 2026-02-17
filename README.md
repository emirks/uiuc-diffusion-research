# diffusion-research

Research workspace for diffusion models with clear separation between reusable code, disposable experiments, and research notes.

## Goal
Use this repository to run disciplined, reproducible diffusion research. Treat it as a lab notebook plus codebase:
- `src/` is stable and reusable.
- `experiments/` is where hypotheses are tested.
- `notes/` and `papers_drafts/` capture reasoning and communication.

## Folder scope (main folders)
- `data/`: datasets and derived data used by experiments.
- `src/`: reusable library code (core diffusion/video logic).
- `experiments/`: runnable experiment entrypoints (`config.yaml` + `run.py`).
- `outputs/`: generated artifacts only (videos, images, logs, checkpoints).
- `scripts/`: one-off utilities (download/prep/inspection helpers).
- `tests/`: sanity and regression checks.
- `notes/`: research thinking, paper notes, derivations, conclusions.
- `papers_drafts/`: writing and figure drafting for reports/papers.

## Current structure
```
diffusion-research/
├── README.md
├── environment.yml
├── pyproject.toml
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── src/diffusion/
│   ├── schedules.py
│   ├── forward.py
│   ├── reverse.py
│   ├── losses.py
│   ├── sampling.py
│   ├── utils.py
│   └── models/
├── experiments/
│   ├── README.md
│   ├── exp_001_forward_l2_norm_dynamics/
│   └── exp_002_loss_variants/
├── scripts/
│   ├── download_cifar_data.py
│   ├── inspect_scheduler.py
│   └── visualize_noise.py
├── tests/
├── notes/
└── papers_drafts/
```

## Environment setup
From `diffusion-research/`:

```bash
pip install -e .
pip install torch torchvision diffusers pyyaml pillow matplotlib pytest
```

`environment.yml` can be used as a template if rebuilding a clean env.

## Recommended research workflow
1. Define a precise question in `notes/ideas.md`.
2. Write a falsifiable hypothesis and minimal experiment design.
3. Implement only the required reusable math/code in `src/diffusion/`.
4. Add/extend tests in `tests/` for shape and math invariants.
5. Create a new numbered experiment folder in `experiments/` with:
   - `config.yaml`
   - `run.py`
   - optional local `README.md`
6. Run experiment and log all outputs under `outputs/`.
7. Summarize interpretation and limitations in `notes/`.
8. Move stable results/figures into `papers_drafts/`.

## Research guidelines (important)
- One experiment = one question.
- Never mutate old experiment configs/results; fork a new `exp_XXX_*` instead.
- Keep `src/` free of plotting, CLI glue, and dataset downloads.
- Keep side effects inside `experiments/` and `scripts/`.
- Always set and log random seed/device in experiment config.
- Store generated artifacts under `outputs/` only.
- Keep raw data immutable in `data/raw/`.
- Put derived/intermediate data in `data/processed/`.
- Record assumptions and interpretation in markdown, not only in code comments.

## Reproducibility checklist
For each experiment, ensure:
- Config file exists and is committed.
- Seed is fixed and recorded.
- Dependency-sensitive choices are explicit (scheduler, timesteps, loss target).
- Output filenames include experiment identity.
- Plots and tensor logs can be regenerated from code + config.

## Data workflow
Download CIFAR-10 into `data/raw/cifar10`:

```bash
python scripts/download_cifar_data.py
```

If needed, keep optional inspection exports in `data/processed/`.

### Start/end frame placement (simple rule)
- Put reusable input frames in `data/processed/` (recommended).
- Keep only tiny, one-off debug frames inside a specific `experiments/exp_XXX.../` folder.

Why:
- `data/` is for shared inputs.
- `experiments/` is for run logic/config and disposable experiment-local files.

## Running experiments
Example implemented experiment:

```bash
python experiments/exp_001_forward_l2_norm_dynamics/run.py
```

Config:
- `experiments/exp_001_forward_l2_norm_dynamics/config.yaml`

Artifacts produced include logs/figures/images under `outputs/`.

## Tests and quality bar
Run tests before or alongside major experiment changes:

```bash
pytest -q
```

Minimum bar for new reusable code in `src/`:
- shape correctness
- timestep/indexing correctness
- deterministic behavior where expected (with fixed seed)

## Notes and writing discipline
- Paper/theory reading summaries belong in `notes/papers/`.
- Derivations belong in `notes/theory/derivations/`.
- Draft narrative and figure planning belong in `papers_drafts/`.
- Keep conclusions scoped: separate evidence, interpretation, and speculation.

## What to improve next
Current repository gaps to address as research expands:
- Implement `src/diffusion/reverse.py`, `losses.py`, `sampling.py`, and model modules.
- Add stronger tests for numerical properties (not only shapes).
- Flesh out `experiments/exp_002_loss_variants/run.py` into a full training/evaluation pipeline.
- Implement utility scripts (`inspect_scheduler.py`, `visualize_noise.py`).
