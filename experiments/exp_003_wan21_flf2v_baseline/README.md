# exp_003_wan21_flf2v_baseline

Minimal baseline for Wan 2.1 FLF2V-style inference using `diffusers`.

## Purpose
Get a first working run from `(start_frame, end_frame, prompt)` to generated video frames.
This experiment intentionally does **not** implement full VC-Bench hard-constrained denoising yet.

## Files
- `config.yaml`: model/input/runtime/output settings
- `run.py`: baseline inference entrypoint

## Run
From repo root:

```bash
python experiments/exp_003_wan21_flf2v_baseline/run.py
```

## First run checklist
1. Set a valid model repo in `config.yaml` (`model.repo_id`).
2. Provide real start/end images in:
   - `inputs.start_frame`
   - `inputs.end_frame`
3. Keep `runtime.dry_run: true` until paths/config are valid.
4. Switch to `runtime.dry_run: false` to run model inference.

## Output
- Manifest: `outputs/videos/exp_003_wan21_flf2v_baseline/run_manifest.json`
- Frames: `outputs/videos/exp_003_wan21_flf2v_baseline/frames/frame_XXXX.png`

## Notes
- The script uses signature introspection so it can adapt to minor API naming differences in FLF2V pipelines (`first_frame`, `start_image`, etc.).
- Includes `slerp(...)` utility for upcoming latent guidance work, but this baseline does not yet wire it into the denoising loop.
