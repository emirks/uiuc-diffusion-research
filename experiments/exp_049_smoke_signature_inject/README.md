# exp_049 — σ-matched recon x̂₀-trajectory injection (self + donor)

## Question

exp_048 injected a clip's **own static `z0_recon`** into the production Euler
regen free-middle at every step and got **self free-mid PSNR 13.05 @ g=1.0** —
far below recon's own ~22–27. Forcing a *fully-sharp* clean prediction at high σ
is off-manifold. **Does injecting the σ-MATCHED step of the recon's own coarse→fine
trajectory `x̂₀(σ_i)` instead recover the reconstruction (self), and where in the
σ-schedule is the smoke transition actually carried (early / mid / late window)?**

## Setup

Producer — `experiments/exp_040_ltx2_feature_cache/config_recon_traj17.yaml`
(num_clip_frames=17, velocity-only, **recon all 40 steps**). Caches, per smoke
reference clip, `z_in`/`v_pred`/`sigma` per recon step. The consumer derives
`x̂₀(σ_i) = z_in − v_pred·σ_i` — **no library change**.

Consumer (this exp) — fork of exp_048. During Euler regen, for steps `i` in the
injection window, blend the free-token clean prediction toward `x̂₀(σ_i)` by
weight `g` (`_x0_clamp_velocity`); outside the window, no injection. recon & regen
share `_build_sigma_grid` → step `i` is σ-matched (asserted at load).

- **source**: `self` (clip's own trajectory — mechanism check) or `donor`
  (a same-orientation-grid donor — deployable).
- **window**: `all` / `early` [0,13) / `mid` [13,26) / `late` [26,40) over 40 steps.
- **g**: blend weight (self g=1.0 mechanism; donor g=0.8 perceptual).

Donor must share the orientation grid (token alignment): portrait 22×16
{ss0,2,3,6,8}, landscape 16×22 {ss1,5,7,9}, square 19×19 {ss4}.

## How to run

```bash
# 1. produce the cache (once, all 10 clips)
PYTHONPATH=src python experiments/exp_040_ltx2_feature_cache/run.py \
    --config experiments/exp_040_ltx2_feature_cache/config_recon_traj17.yaml
# 2. point cache_source.run_dir at the produced run_NNNN, then:
PYTHONPATH=src python experiments/exp_049_smoke_signature_inject/run.py
```

## Expected outcome

- **self / all / g1.0** → free-mid PSNR ≫ exp_048's 13.05, toward recon
  (~22–27). If not, the σ-matched-trajectory premise is wrong → stop.
- **window sweep** → locates where the transition lives; expect `late` to carry
  most of it (It-4).
- **donor** → judge perceptually (lum/tex/sat/tdiff + visual). PSNR-vs-target is
  information-limited (exp_045), informational only.

## Outputs

`outputs/videos/exp_049_smoke_signature_inject/run_NNNN/`:
`<sample>/source_video.mp4`, `<sample>/regen_<variant>.mp4`, `summary.yaml`,
`run.log` (per-variant full/free-mid PSNR + perceptual signals; median table).
