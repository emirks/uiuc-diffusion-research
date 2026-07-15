# diffusion-research Project Rules

## Changelog

`CHANGELOG.md` lives at the repo root. **Update it whenever you complete something meaningful** — a bug fix, a new experiment, a significant code change, a documented finding, or an infrastructure change.

Format: newest entry at top, under a `## YYYY-MM-DD` date heading. Each entry is a **`HH:MM` timestamp** followed by a plain-language sentence or two — what changed and why it matters. Get the current time with `date '+%Y-%m-%d %H:%M'` if needed. Include code specifics (file names, function names) only when they help locate the change. Do not log trivial edits.

---

## Commit & push discipline

**Local and remote must never drift.** After *any* checkpoint that changes the repo — a new experiment scaffold (`exp-design`), a finished run + write-up (`exp-finish`), an eval/certification (`exp-eval`), a doc/spec/knowledge-bank edit, an infra change — **commit and push immediately, in the same step**. End every such step with a clean `git status` and `ahead=0 behind=0`.

- **Atomic**: one logical change per commit; stage by pathspec, never `git add -A`. Message `type(scope): what` — `exp_NNN:`, `docs:`, `eval:`, `infra:`, `chore:`.
- **Push after every commit** (`git push`) — never leave commits sitting local.
- **Identity**: this environment exports a wrong `GIT_AUTHOR_EMAIL`, so commits mis-attribute. Always commit as `emirks <emirks88@gmail.com>` — guard once per shell: `export GIT_AUTHOR_NAME=emirks GIT_AUTHOR_EMAIL=emirks88@gmail.com GIT_COMMITTER_NAME=emirks GIT_COMMITTER_EMAIL=emirks88@gmail.com`.
- **Never commit** big/regenerable files — respect `.gitignore` (`wandb/`, `**/.precomputed*/`, `**/decoded_videos/`, weights). Dataset `*.mp4` go through Git-LFS (`.gitattributes`); needs `~/.local/bin` on PATH.
- **Branches**: do exploratory/spec work on a feature branch, same commit+push-immediately rule (first push `git push -u origin <branch>`). When the work concludes (certified, decision landed), `git merge --no-ff` it into `main` and push; leave the branch as the record. Keep every branch at `ahead=0 behind=0` vs its upstream.

---

## Knowledge bank workflow

`notes/INDEX.md` is the **single entry point** for everything learned in this project (model mechanics, API quirks, experiment findings, theory).

**Before implementing anything non-trivial:**
1. Open `notes/INDEX.md` and scan the quick-scan table.
2. If a relevant file exists, read it before writing code or answering.
3. This avoids re-deriving known facts (e.g. latent geometry, scheduler shift, conditioning index semantics).

**After validating new knowledge** (a mechanism was confirmed, a bug was understood, a finding was replicated):
1. Find the correct note file (or create one — see below).
2. Add the new content as a clean `##` section. Be atomic and concrete.
3. Update the entry in `notes/INDEX.md` (description or table row).

**Creating a new note file:**
- Choose the right subdirectory: `models/<name>/`, `theory/`, `papers/`, or root `notes/` for dataset/benchmark notes.
- Add a row to the quick-scan table in `notes/INDEX.md` and an entry in the relevant area section.

Do **not** dump raw chat transcript. Write clean, factual notes a future reader can trust.

---

## Repo layout (never violate)

- `src/diffusion/` — reusable, importable library code only. No CLI, no plotting, no dataset downloads.
- `experiments/exp_NNN_<slug>/` — one question per experiment. Contains `run.py`, `config.yaml`, `README.md`.
- `outputs/` — generated artifacts only (videos, images, logs). Never committed source or config edits.
- `data/raw/` — immutable. `data/processed/` — derived inputs.
- `tests/` — shape, indexing, and determinism checks for `src/`.

---

## Starting a new experiment

1. Next number = `max(existing exp_NNN) + 1`. Pad to 3 digits: `exp_012_…`.
2. Copy the closest prior experiment as a starting point or write a new one; never mutate old ones.
3. Every experiment must have:
   - `config.yaml`
   - `run.py`
   - `README.md` with `## Question`, `## Setup`, `## How to run`, `## Expected outcome` / `## Outputs`.
4. Output dir in config: `"outputs/videos/exp_NNN_<slug>"`.
5. Output filename must encode distinguishing hyperparams (seed, steps, cfg, anchor_frames, …).
6. Always save a `config_snapshot.yaml` alongside the video.

---

## config.yaml conventions

```yaml
model:
  model_id: "org/model-name"      # HF Hub ID

inputs:
  # paths relative to REPO_ROOT

inference:
  height: 512
  width: 768
  num_frames: 121
  num_inference_steps: 40
  guidance_scale: 4.0

runtime:
  seed: 42
  device: cuda
  dtype: bfloat16
  cpu_offload: true

outputs:
  dir: "outputs/videos/exp_NNN_<slug>"
  fps: 24
```

---

## run.py conventions

- `REPO_ROOT` anchored at `pathlib.Path(__file__).resolve().parents[2]`.
- All file paths resolved as `REPO_ROOT / cfg[...]` — never hardcoded strings.
- Always import boilerplate from `diffusion.exp_utils`:
  ```python
  from diffusion.exp_utils import (
      load_config, next_run_dir, resolve_resolution,
      load_clip_from_mp4,   # when loading MP4 clips
      compute_num_frames,   # when num_frames depends on anchor_frames
      TeeLogger,
  )
  ```
- Call `load_config(CONFIG_PATH)` (not a local `load_config()`).
- **Always use `TeeLogger`**: create `run_dir` first via `next_run_dir`, then wrap the rest of `main()`:
  ```python
  run_id, run_dir = next_run_dir(out_dir)
  with TeeLogger(run_dir / "run.log"):
      ...  # all print() output saved to run.log + shown in terminal
  ```
- **`num_frames`**: if the experiment sweeps `anchor_frames`, do NOT hardcode `num_frames` in config. Use `target_middle_frames: 24` and compute per-run:
  ```python
  num_frames = compute_num_frames(anchor_frames, cfg["inference"]["target_middle_frames"])
  ```
  For fixed-anchor experiments, `num_frames` may still be explicit in config.
- Resolution: use `resolve_resolution(cfg["inference"], mod_value, ref_image)`.
  - `ref_image` = first frame of start clip (MP4 experiments) or first PNG frame.
  - Handles both `max_area` and explicit `height`/`width` configs transparently.
- Pipeline always loaded with `vae` and `image_encoder` at `torch.float32`; transformer at config dtype.
- Always call `pipe.vae.enable_tiling()`.
- Generator: `torch.Generator(device=device).manual_seed(seed)`.
- Print `[info]` lines for key inputs, resolution, and `[done] run_id → path` at the end.

---

## What belongs in src/ vs experiments/

| Reusable math / pipeline logic | → `src/diffusion/` |
| One-off run script, config, I/O | → `experiments/exp_NNN_…/` |
| Dataset download / prep | → `scripts/` |
| Generated files | → `outputs/` |

---

## Compute: UIUC Campus Cluster (Slurm) — since 2026-07

Experiments run on the UIUC Campus Cluster (login node `cc-login3`), not
RunPod. The repo lives at `$LAB/diffusion-research` with
`LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa`; framework caches are
redirected to `$LAB/cache/*` via `~/.bashrc`; the research env is
`conda activate $LAB/envs/diffusion` (after `module load anaconda3/2024.10`).

**Before any cluster/experiment-lifecycle operation, invoke the matching
skill first — they are the source of truth for this workflow:**

- `cc-slurm` — partitions/accounts/GPUs, queue etiquette (high/normal/
  secondary), storage map, modules, monitoring commands
- `exp-design` — plan an experiment or sweep (resource plan, queue choice,
  preemption-readiness, array-friendly variant structure)
- `exp-submit` — validated sbatch template, job arrays, dependency chains,
  post-submit checks
- `exp-status` — job/progress monitoring, pending diagnosis, failure taxonomy
- `exp-finish` — completion verification, vc-bench eval, CHANGELOG write-up,
  cleanup
- `exp-eval` — evaluating transition generations with the certified
  transition-eval harness (`eval/v3.0.0`): certified-checkout rule,
  plan→infer→score, trust map, certification protocol; SPEC.md in
  `src/diffusion/transition_eval/` is the authority

Hard rules: long runs go through `sbatch` (never park `srun ... bash`);
`-high` queues require a `#cluster_high_priority` Slack announcement;
`-secondary` queues are preemptible — only submit resumable runs there.

### RunPod (legacy, pre-2026-07)

The pod-based flow is preserved in the `runpod-pod-init` skill (marked
legacy). The volume `s3://sy54sawkcs` still exists read-only pending
decommissioning. Pod IDs in older CHANGELOG entries refer to that era.
