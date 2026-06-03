# AGENTS.md

## Project Overview

Research workspace for diffusion models (Wan 2.1 and LTX-2 video generation). Not a production application — it's a lab notebook with reusable library code (`src/diffusion/`), numbered experiments (`experiments/exp_NNN_*/`), and research notes.

See `README.md` for folder layout, research workflow, and reproducibility checklist.

## Cursor Cloud specific instructions

### Environment

- Python 3.12 (system); no virtual env or conda needed in the Cloud VM.
- `~/.local/bin` must be on `PATH` for `pytest`, `ruff`, and other pip-installed CLI tools. Add `export PATH="$HOME/.local/bin:$PATH"` at the start of shell sessions if not already set.
- No GPU available in Cloud VMs — experiments run on CPU (the code auto-detects this). GPU-heavy experiments (Wan/LTX-2 video generation, exp_003+) will not complete without a CUDA GPU, but CPU-only experiments (exp_001, exp_002) and the test suite work fine.
- Model weights (`/workspace/cache/models/`) are not present in the Cloud VM. Experiments that load large models will fail; this is expected.

### Key commands

| Task | Command |
|------|---------|
| Install deps | `pip install -e . && pip install torch torchvision diffusers pyyaml pillow matplotlib pytest tqdm transformers accelerate safetensors einops av opencv-python scipy ruff` |
| Run tests | `python3 -m pytest -q` |
| Lint | `ruff check src/ tests/` |
| Run experiment | `python3 experiments/exp_001_forward_l2_norm_dynamics/run.py` |

### Gotchas

- First run of exp_001 downloads CIFAR-10 (~170 MB) to `data/raw/cifar10/`. Subsequent runs reuse the cached dataset.
- The `pyproject.toml` declares no runtime dependencies; all deps must be installed via pip separately (see `environment.yml` as a reference for the full list).
- Pre-existing lint issues (17 warnings in `src/diffusion/signals/`) are known and not blockers.
- The `diffusion` package is installed in editable mode (`pip install -e .`), so code changes in `src/` are reflected immediately.
