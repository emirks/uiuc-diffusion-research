# Changelog

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
