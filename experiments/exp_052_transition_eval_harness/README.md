# exp_052 — Transition eval harness v1: build it, then make it pass its own exam

## Question

Can we measure "same transition style, different content" quantitatively? Pixel
comparison is dead on arrival (the task definition is *same program, different
pixels*), so the harness measures content-invariant signatures — each computed
relative to a video's **own** endpoints and frames. Before any method decision
rests on these metrics, the harness must pass its own exam on ground-truth data:
**if a metric cannot tell shadow_smoke from earth_wave on real clips, it cannot
evaluate transfer.**

Two sub-questions, pre-registered:

1. **Style discrimination** (zero generated videos needed): on the 41 real clips
  across 9 styles in `data/processed/transitions/` (higgsfield excluded —
   uncurated), does each metric retrieve style above chance (largest class =
   10/41 ≈ 24%) via leave-one-out 1-NN?
2. **Ladder sanity** ( consumers): scoring the 24 exp_051 outputs
  (base / t2v / i2v_ff05 / c2v × ew0-2 × trigger/no-trigger), does the metric
   ranking reproduce the visual findifirst realngs — base worst on transition fidelity,
   all LoRA arms ≈ full morphology, c2v best on boundary continuity?

## Setup

### Library (`src/diffusion/transition_eval/`) — the metric suite

All metrics operate on per-frame **DINOv2-base CLS embeddings** (cached per
video in `outputs/eval/cache/`) unless noted.

- **M1 Morph Profile** (`morph.py`) — the signature metric. Per-frame cosine to
own endpoint A (mean feature of first 9 px frames) and endpoint B (mean of
last 8) → curves `a(t), b(t)`. Curves are floor-normalized by
`cross = cos(eA, eB)` (the "unrelated content" baseline for that pair):
`â = (a − cross)/(1 − cross)`, so â≈1 means "is endpoint A", â≈0 means
"unrelated content" — content-invariant thresholds. Profile comparison:
resample to 96 pts, z-norm, 2-channel DTW (Sakoe-Chiba band 15%) + Pearson.
Derived scalars: **transformation depth** = 1 − min_mid max(â,b̂) (crossfade
→ ~0, true metamorphosis → ~1), departure/arrival timing, identity-hold
fraction, **core mask** = frames where max(â,b̂) < 0.5 ("neither endpoint";
fallback: argmin frame).
- **M2 Motion Fidelity** (`motion.py`) — the established metric (Yatim et al.,
Space-Time Diffusion Features): CoTracker3 tracklets (20×20 grid), per-step
velocity directions, normalized cross-correlation, bidirectional max-match
mean. Tracklet velocities resampled to 64 steps so 122- and 242-frame clips
compare. v1 = full frame (noted: static-background dilution; v2 = restrict
to effect region via core mask).
- **M3 Effect Appearance** (`appearance.py`) — DINO features of **core frames
only** (the frames that are neither endpoint, by M1's mask) compared
set-to-set (symmetric mean-of-max cosine) against the reference's core
frames. Isolates the medium (smoke/ravens/water) from endpoint content by
construction.
- **M4 Rubric VLM Judge** (`judge.py`) — checklist, not vibes: 5 fixed
questions (same-type transition present? dynamics/timing match? endpoints
preserved + seamlessly entered/exited? content leaked from reference?
artifacts?), each answered yes/no + evidence, JSON out, temperature 0.
Backend: local Gemma 3 12B-it (vision tower confirmed in config). Inputs:
8-frame contact strips of reference + generated, endpoint stills.
**Status: experimental until validated against human labels** — do not use
as a headline number yet; ~50-100 human-labeled outputs → Spearman ρ needed
first (target ≥0.8).
- **M5 Endpoint fidelity + boundary seams** (`endpoints.py`) — LPIPS + DINO
between generated conditioned frames (first 9 / last 8) and the condition
clips (resize-to-cover + center-crop, mirroring the ValidationRunner);
**seam detection** = temporal-LPIPS d(t, t+1) z-spike at the conditioning
handoff boundaries (px frames 8→9 and 112→113) vs the video's own d(t)
distribution.
- **M6 Leakage** (`appearance.py`) — max cosine of generated core frames
retrieved against ALL reference-style frames, contrasted with the same
retrieval against the other 8 styles (style-driven similarity must not read
as leakage); reports the argmax (gen frame, ref frame) pair for eyeballing.

### Normalization — every number gets a floor and a ceiling

Anchored controls run through the *identical* pipeline:

- **lerp control** (floor): synthesized crossfade — copy prefix 9 frames, linear
blend prefix[-1]→suffix[0] over the middle, copy suffix 8 frames
(`controls.py`). Built per endpoint pair (each real clip's own endpoints;
each eval item's condition clips).
- **base model, no LoRA** (floor #2): exp_051 `base` arm, already generated.
- **real-clip ceiling**: leave-one-out real clips of the same style (a real
shadow_smoke clip scored against the other 9 = the best any generator could
look under these metrics).
- Reported score = (raw − lerp floor)/(ceiling − lerp floor), clipped to [0,1],
alongside raws. **No single composite number**: headline = per-arm three-axis
table (transition fidelity / endpoint fidelity / leakage) + success rate.

### Manifest schema — condition-count aware from day one

`manifest.py`; JSON items: `{item_id, generated_video, style, n_endpoints, condition_prefix: {video, num_frames}|null, condition_suffix: {...}|null, arm, notes}`. 1-endpoint (portal-style) items drop `b(t)`, suffix fidelity, and
arrival timing; keep `a(t)`, hold, M2, M3, judge, leakage — with the honest
note that the 1-endpoint task contract is weaker (outcome underdetermined,
success leans on the judge).

### Resource plan

- **GPU/queue**: `HCESC-L40S-normal` (`--account=hcesc-l40s`) — analysis
workload, fits 48 GB easily (DINOv2-base ≈ 0.4 GB, CoTracker3 ≈ 1 GB, LPIPS
tiny, Gemma 3 12B bf16 ≈ 24 GB loaded only in the judge phase after the
others are freed). Fallback: cluster `secondary` (`gpu:H100:1`,
`--account=campusclusterusers`).
- **Walltime**: smoke ≈ 15 min; full ≈ feature pass ~20 min (≈110 videos:
41 real + 44 lerp/controls + 24 ladder + condition clips) + CoTracker ~40 min
  - judge ~30 min → 4 h with ×1.5 buffer.
- **Resumability**: every expensive artifact (DINO features, tracklets) is
content-hash cached on disk → a requeued job skips completed work naturally.
Normal (non-preemptible) queue anyway.
- **Weights**: DINOv2-base prefetched to `$HF_HOME`, CoTracker3 to
`$TORCH_HOME/hub` (login-node prefetch, minutes); Gemma + LPIPS already
staged.

## How to run

```bash
cd $LAB/diffusion-research
mkdir -p outputs/logs/slurm
# 1. smoke (3 styles × 2 clips, 48-frame cap, no judge) — always first
sbatch experiments/exp_052_transition_eval_harness/job_eval.sbatch --smoke
# 2. full validation exam + exp_051 ladder scoring (one job, two phases)
sbatch experiments/exp_052_transition_eval_harness/job_eval.sbatch
# CPU unit tests (login node, no GPU):
PYTHONPATH=src python -m pytest tests/test_transition_eval.py -q
```

## Expected outcome

Pre-registered:
(a) **Style discrimination**: Morph Profile and Effect Appearance ≥ ~2× chance
(≥50% 1-NN accuracy); Motion Fidelity above chance but weaker full-frame
(static-background dilution); styles with 2 clips (flame, jump) will be noisy.
(b) **Lerp floor**: transformation depth ≈ 0 for every synthesized crossfade,
well-separated from real clips' depth distribution (real transitions with a
"neither-endpoint" middle should score high).
(c) **Ladder sanity**: base arm scores lowest on M1-profile + M3 similarity to
shadow_smoke refs; all three LoRA arms cluster together and above base; c2v
shows the smallest boundary-seam spikes and best M5 endpoint fidelity; leakage
low everywhere (no reference content in outputs); judge (experimental) agrees
with the arm ranking directionally.
(d) Failure condition, stated in advance: any metric at/below chance in (a)
gets fixed or dropped before it is used in an ablation table — that is the
point of the exam.

## Outputs

- `outputs/eval/cache/` — reusable DINO feature + tracklet caches (content-keyed).
- `outputs/eval/exp_052/validation/` — distance matrices, confusion matrices,
per-style morph-profile figure, discrimination report (`report.md`,
`results.json`).
- `outputs/eval/exp_052/ladder/` — per-item scores (`items.jsonl`), per-arm
normalized table (`report.md`), judge JSONs, seam plots.
- W&B: project `creative-transition-transfer`, runs `exp052_validation`,
`exp052_ladder`.

## Outcome

**Completed 2026-07-06.** Jobs: smoke 9354322, full 9354400 (both
HCESC-L40S-normal ccc0440, EXIT 0), judge re-run 9355187 (LTX venv — the
diffusion env's torch 2.5 predates transformers' Gemma3 torch≥2.6 requirement).

Against the pre-registration: (a) **PARTIAL** — effect appearance 0.93 1-NN
(≥2× chance ✓), motion fidelity 0.46 (above chance, weaker full-frame as
predicted — but only after three transition-specific fixes found via the smoke:
mid-frame grid queries + per-step occlusion masking + duration-normalized
smoothed velocities; the naive frame-0 protocol was AT chance), morph DTW 0.34
(**below the pre-registered 50% bar** → scoped per the failure clause: it is a
transition-family/timing fingerprint — jump 1.0, display 0.75 — not a style
ID; style retrieval belongs to appearance). The three metrics fail on
complementary classes. (b) **REVISED** — lerp depth is 0.62, not ≈0 (a pixel
crossfade's double-exposure middle leaves DINO's endpoint neighborhoods);
separation from real clips (0.92 mean, 0.64 min) holds and `lerp_below_02=0`.
(c) **REPRODUCED** — base lowest on appearance (0.56 vs 0.69–0.79 norm) and
motion (0.47 vs 0.88–0.91); LoRA arms cluster; c2v tops profile fidelity
(1.00); NO seams in any arm (all robust-z < −0.55) and endpoint DINO ≈0.98
everywhere = the quantitative twin of "anchoring is mechanism-robust"; morph
profile near-saturates for all arms under conditioning (anchor-dominated
curves) so the discriminative axes are appearance/motion/judge. Bonus:
no detectable trigger effect at n=3/cell (appearance ± SHDWSMK; suggestive of
always-on style — probe before multi-style route decisions, see exp_053 note);
leakage max sim ≤0.78 → no near-copies. Details:
`notes/exp/exp_052_eval_harness.md`.