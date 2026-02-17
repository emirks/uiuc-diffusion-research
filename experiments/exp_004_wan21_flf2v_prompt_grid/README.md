# exp_004_wan21_flf2v_prompt_grid

Runs Wan FLF2V over a small prompt/frame grid:
- 3 different `first_last_frames` pairs
- 3 different prompts
- Total: 9 runs

## Purpose
Fast comparative experiment for prompt sensitivity across multiple start/end pairs while keeping the same inference settings.

## Inputs
Configured in `config.yaml`:
- `inputs.frame_pairs_root`: folder containing pair subfolders with `first.png` and `last.png`
- `inputs.num_frame_pairs`: number of pair folders to use (if `frame_pair_ids` is empty)
- `inputs.frame_pair_ids`: optional explicit pair folder names
- `inputs.prompts`: prompt list
- `inputs.negative_prompt`: shared negative prompt

## Run
From repo root:

```bash
python experiments/exp_004_wan21_flf2v_prompt_grid/run.py
```

## Output
Root:
- `outputs/videos/exp_004_wan21_flf2v_prompt_grid/`

Per run (incremental):
- `run_0001/sample.mp4`
- `run_0001/config_snapshot.yaml`
- `run_0002/...`

Metadata log:
- `outputs/videos/exp_004_wan21_flf2v_prompt_grid/runs_metadata.jsonl`

Each metadata row includes: `pair_id`, `prompt_idx`, `prompt`, `seed`, paths, device, dtype, and full config.
