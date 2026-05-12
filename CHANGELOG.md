# Changelog

## 2026-05-12

**13:09** — Revised exp_028 per visual-review feedback and re-ran on an A100 PCIe ($1.39/hr, EU-RO-1 secure, ~5 min total wall). Two changes: (1) the bridge in `hold_bridge_hold` is no longer a 57-pixel-frame cross-fade re-encoded together — it's now a latent-space lerp between two *single-frame VAE encodings* (last_start_frame alone → key-frame latent A, first_end_frame alone → key-frame latent B), filling the M middle slots with alphas (1..M)/(M+1). This drops the previous version's double-anchoring of `first_end_pixel` (it had appeared at both the bridge tail and `end_lat[0]`), so the second-clip onset isn't ambiguous anymore. (2) Added a length sweep `num_frames_sweep = [121, 89, 65]` → M ∈ {8, 4, 1}; N=65 with M=1 is the smallest possible bridge that still keeps both clips fully held. Output filenames now encode M (`s42_K25_N{nf}_M{m}_mode-{mode}.mp4`). 27 mp4s in `outputs/videos/exp_028_vae_latent_composition/run_0002/` covering 3 samples × 3 lengths × 3 modes. Per-output wall time 2-8s (no pixel-cross-fade encode pass, so the bridge mode is no longer the bottleneck).

**12:41** — Ran exp_028 on an A100-SXM4-80GB in EU-RO-1 (community-cloud sold out of all pre-Blackwell GPUs at 12:25, poller hit at cycle=7 / ~7 min later). All 3 samples × 3 modes (naive / hold_lerp_hold / hold_bridge_hold) wrote cleanly under `outputs/videos/exp_028_vae_latent_composition/run_0001/`. Per-mode wall time was 4-33s (naive/lerp are pure latent arithmetic; bridge adds a 57-pixel-frame VAE encode pass). The `hold_bridge_hold` mode produced exactly the expected `bridge_lat` shape `(1,128,8,16,24)` — confirms the `M_px = 8*(M-1)+1 → M` arithmetic. Logs/summary committed; videos are the visual A/B/C for the dissolve-cause diagnosis.

**12:14** — Built exp_028 (`experiments/exp_028_vae_latent_composition/`) — a fork of exp_023 that fixes two diagnosed problems with the VAE-only dissolve probe. (1) exp_023's `alpha = t / (T_total - 1)` ramps from 0 across the entire timeline, so the blend kicks in immediately at output frame 1 — the start clip never plays clean. (2) The middle blends mix `start_lat[-1]` (a *motion* latent encoding pixel frames 17-24 of the start clip) with `end_lat[0]` (a *key-frame* latent encoding only pixel frame 0 of the end clip) — arithmetic on two latents with incompatible temporal semantics, producing flicker. exp_028 runs three modes per sample for visual comparison: `naive` (exp_023 baseline), `hold_lerp_hold` (pure boundaries + latent lerp middle — isolates fix 1), and `hold_bridge_hold` (pure boundaries + a pixel-space cross-fade between `last_pixel_of_start` and `first_pixel_of_end` re-encoded by the VAE to exactly fill the middle — isolates fix 1 + 2). For default `num_frames=121` / `num_clip_frames=25`: T_total=16, T_clip=4, middle M=8, bridge clip = 57 pixel frames. 3 samples (easy/mid/hard).

**11:51** — Added `experiments/exp_027_ltx2_rf_inversion/CFG_AND_PROMPT.md` documenting the conditioning contract: generation runs at CFG=4 with positive + negative prompt and 80 NFE; inversion/reconstruction runs at CFG=1 with the positive prompt still active (negative dropped) and 60 NFE per direction — the "2 calls per step" come from the midpoint integrator, not CFG. Spells out the three independent reasons CFG=1 is required for inversion (non-conservative mixed field, NFE budget, downstream cache reusability), separates text CFG from LTX-2's visual conditioning (per-token timestep + x₀-clamp), and lists six future-pitfall warnings (the `inversion.guidance_scale` knob is currently a no-op past 1.0; bf16/fp32 boundary; σ<1e-4 short-circuit; etc.).

## 2026-05-11

**16:50** — Cleaned up `experiments/exp_023_vae_latent_lerp/`: replaced the stray exp_020 README with a real one describing the VAE-only dissolve-cause probe (hypothesis, interpolation scheme, sweep table for `num_frames` ∈ {121, 73, 57}, outputs layout); rewrote the misleading config comment that claimed `num_frames=121` when the value was 57; updated `run.py` output filename to `s{seed}_K{K}_N{num_frames}_lerp.mp4` so sweep outputs are distinguishable on disk per project convention.

**16:27** — Added section 2.5 (PCA of velocity-field frame embeddings) to `notebooks/exp021_02_velocity_field.ipynb`, mirroring the VAE PCA in Level 1.4 and the transformer PCA in Level 3.2 but with `v_pred` as the source signal. Grid is samples × {τ=0, 13, 26, 39} — early-step two-cluster splits indicate the model commits to the dissolve frame from the noisiest step. Also fixed the `NameError: add_gt_vline` in cells 2.2 and 2.4 by forcing `importlib.reload(trajectory_utils)` in the imports cell so a stale module cache cannot mask newly added helpers.

**11:44** — Built exp_027 (LTX-2 RF-Solver flow inversion, Step 6 of the editing pipeline). Custom denoising loop on top of `LTX2ConditionPipeline`: VAE-encode Stage-1 generated z₀ → invert via RF-Solver midpoint 2nd-order (30 steps, CFG=1, 60 NFE) → reconstruct → per-frame LPIPS gate at 0.05. Replicates the pipeline's per-token timestep + x₀-domain conditioning clamp, with a short-circuit guard at σ<1e-4 to avoid the schedule's degenerate first inversion step. Auto-retries at 50 steps on gate miss. 3 DAVIS pairs (easy/mid/hard). Cached artefacts: z₀, z₁, σ-checkpoints, source/recon videos, LPIPS stats. Ready for Step 7 (feature injection).

**11:18** — Opened placeholders for exp_025 (LTX-2 negative-prompt sweep) and exp_026 (LTX-2 seed × end-clip-strength sweep) with READMEs/configs but stubbed `run.py` — to be implemented after the next round of work, when we revisit which structural knobs (negative prompt, seed lottery, endpoint clamping) most affect transition creativity beyond the empty-prompt finding from exp_024.

**11:01** — Cleared saved cell outputs from `notebooks/exp021_trajectory_analysis.ipynb` (~176 MB → ~118 KB) so GitHub accepts the blob under its 100 MB limit; re-run the notebook locally to regenerate plots.

**10:57** — Published exp_023 (VAE latent interpolation) and exp_024 (LTX-2 prompt sweep) with configs and run scripts; added Jupyter notebooks under `notebooks/` for exp_021 trajectory analysis (`trajectory_utils`, programmatic notebook generator) and exp_024 prompt exploration. Documented LTX-2 19B P=1 patch geometry and token locality in `spatial_locality.md` with matching updates to `conditioning.md` and the knowledge index. Added root `CHANGELOG.md`, Cursor rule for keeping it current, repo-wide `CLAUDE.md` guidance, experiment-wide `experiments/CLAUDE.md`, and gitignore entries for `.claude/` and `.ipynb_checkpoints/`.

## 2026-05-05

**16:45** — exp_024 prompt update: rewrote all 10 Category B prompts to describe continuous semantic morphing (feathers changing color and shape, bus form growing from a car silhouette, clothing materializing mid-walk) instead of scene cuts. Removed "morphing" and "warping" from the negative prompt since they fight the new B intent. Added Stage 1 video save to `run.py` so each run now outputs both a Stage 1 preview (512×768, silent) and the final Stage 2 output (1536×1024), enabling faster iteration decisions. Created `ltx2_prompting_notes.md` in the experiment folder documenting the format rules, transformation mechanism principles, and what prompt language to avoid.

---

Newest first. Each entry has a timestamp and says what changed in plain language.
Code details only when they help locate the change.

---

## 2026-04-08

**12:57** — Changelog made timestamped and language-first per this update.

**12:45** — Notes folder fully restructured. Moved theory notes into `theory/`, dataset notes into `dataset/`, split the monolithic LTX-2 reference into three focused files (pipeline API, conditioning mechanics, denoising schedule), added Wan 2.1 model notes, removed all empty stub files, rewrote the knowledge index.

**12:30** — Cursor rule updated to require maintaining the changelog on every meaningful change, placed as the first instruction so it is never missed.

**12:15** — Learned and documented how the LTX-2 denoising schedule actually works: the sigma shift concentrates many small steps at the noisy end and few large steps near clean. This explains the counterintuitive pattern in the trajectory heatmaps where latent displacements are largest in the final denoising steps, not the first.

**11:45** — Fixed two plotting bugs in exp_022: the y-axis arrow on heatmaps pointed in the wrong direction after matplotlib's rotation, and the conditioning boundary lines were drawn at the wrong column position for curvature and angular features (which have shorter x-axes than the other panels).

**11:10** — Fixed a crash in exp_021 trajectory logging. LTX-2 passes latents to the scheduler in packed sequence format `[batch, tokens, channels]`, but the logger assumed a spatial layout. Added unpacking logic so the rest of the analysis works unchanged.

---

## [earlier — dates not recorded]

**exp_022** — Geometric feature extraction from trajectories. Computes per-frame norms, speed, curvature, and angular consistency across all denoising steps. The discrete Laplacian of the final clean latent turned out to be the best signal for locating the dissolve frame; works well for semantically distant clips, less so when the conditioning boundary dominates.

**exp_021** — Trajectory logging. Patches the scheduler to capture the full denoising trajectory — every latent state and velocity prediction at every step — and saves it for offline analysis.

**exp_020** — First working clip-to-video pipeline using Diffusers natively. Key discovery: Stage 2 requires the doubled spatial dimensions explicitly, otherwise the conditioning mask is built at the wrong size.
