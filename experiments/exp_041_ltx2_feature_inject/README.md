# exp_041 — First Feature-Injection Experiment (MasaCtrl-style self-attn K,V)

First experiment on the feature-injection axis. Reads the recon-pass cache
from [`exp_040`](../exp_040_ltx2_feature_cache) (`config_recon_ss4.yaml`,
sample `shadow_smoke_4`) and injects the source self-attention K,V into the
free middle of a re-noised reconstruction.

Reusable mechanism lives in [`src/diffusion/feature_inject.py`](../../src/diffusion/feature_inject.py)
(`FeatureInjector`) — the read side of [`feature_cache.py`](../../src/diffusion/feature_cache.py).

## Question

Do source self-attention K,V (mid layers, descending recon steps), injected
into the **free middle** of a reconstruction whose free-middle noise was
reseeded, transport the source transition — i.e. pull the output back toward
the source recon despite a different free-middle noise code?

This is the prerequisite test for the real goal: transplanting a *good*
transition's features into *hard* shadow_smoke clips that go off-manifold.

## Where / what we inject, and why

* **Method — MasaCtrl-style K,V replacement.** Keep the edit pass's Q (it
  encodes "where am I in the layout"), replace K,V (they encode "what content
  lives at each position"). Lowest-risk, most-published injection; matches the
  cached `attn1_k` / `attn1_v`.
* **Spatial scope — free middle only** (latent frames 4–11 → packed tokens
  `[4·tpf : 12·tpf)`). The C2V anchors (start frames 0–3, end frames 12–15
  minus the drop1 token) are hard-pinned by the conditioning mask + x0-clamp
  at every step, so injecting there is pointless and fights the conditioning.
  The free middle is the only region with degrees of freedom and is where the
  transition lives.
* **Layers — the 4 mid layers cached `{11,19,28,37}`.** PnP/MasaCtrl
  consensus: early layers carry coarse layout, late layers carry fine
  appearance, mid layers carry transferable structure. We leave early/late to
  the edit pass and transplant the mid band.
* **Steps — all cached recon steps `{0,4,9,14,19,24,29,34,39}`.** Edit-step k
  ↔ cached-step k directly (both descending recon, no σ flip). Inject on the
  **predictor** substep only (the σ_curr call we cached); the midpoint
  corrector runs free.
* **Strength 1.0** — hard replace (MasaCtrl default).

The injection point is the output of `attn1.to_k` / `attn1.to_v` — pre-RMSNorm,
pre-RoPE. Because norm + RoPE are deterministic functions of token position and
the layout is identical between the cached pass and the edit pass, overwriting
the linear output is exactly equivalent to overwriting the K/V that attention
consumes.

## Setup

All passes are CFG=1 midpoint reconstruction (matching the cached recon pass),
on `shadow_smoke_4`, reusing the frozen `z1` and conditioning state from the
exp_040 cache run.

| Pass | z1 free-middle | Injection | Expected vs reference |
|---|---|---|---|
| reference | original | — | `z0_recon.pt` (source recon) |
| **B** perturbed baseline | re-noised (seed 1234) | none | **far** from reference |
| **C** perturbed + inject | re-noised (seed 1234) | source K,V | **close** if injection works |
| **D** self-inject (null) | original | source K,V | ≈ reference (plumbing sanity) |

**Headline metric:** PSNR/SSIM/LPIPS vs reference on the **free-middle pixel
frames** (latent 4–11 → pixels 25–88). Injection transports the transition iff
**C ≫ B** there (ΔPSNR > 0, ΔSSIM > 0, ΔLPIPS < 0), and **D ≈ reference**.

## How to run

```bash
source /workspace/cache/pod_init.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_041_ltx2_feature_inject/run.py
```

Requires the exp_040 recon cache to exist at
`outputs/videos/exp_040_ltx2_feature_cache/run_0001/shadow_smoke_4/feature_cache/`.

## Outputs

`outputs/videos/exp_041_ltx2_feature_inject/run_NNNN/`:

* `reference_recon.mp4` — source recon (decode of `z0_recon.pt`).
* `perturbed_baseline.mp4` — pass B.
* `perturbed_inject.mp4` — pass C.
* `self_inject.mp4` — pass D.
* `metrics.yaml` — full-clip + free-middle PSNR/SSIM/LPIPS of B/C/D vs
  reference, plus the headline (C − B on the free middle).
* `config_snapshot.yaml`, `run.log`.

## Reading the result

* **C − B strongly positive (free middle):** self-attn K,V injection
  transports the transition. Green light to widen — layer/step sweeps, then
  cross-clip transplant (good transition → hard clip).
* **C ≈ B:** injection at these sites/steps doesn't bind. Next probes: add
  `block_out` injection (PnP), inject earlier steps, or raise the layer count.
* **D far from reference:** plumbing bug (self-into-self should be near
  identity) — investigate before trusting C.

---

## Results (2026-05-23, shadow_smoke_4, A100)

Two configs, identical except injection sites. Metrics vs source recon on the
free-middle pixel frames (25–88). Headline = **C − B** (perturbed+inject minus
perturbed-baseline); injection helps iff ΔPSNR/ΔSSIM > 0 and ΔLPIPS < 0.

| run | sites | free-mid PSNR B→C | ΔPSNR | ΔSSIM | ΔLPIPS | verdict |
|---|---|---|---|---|---|---|
| run_0001 | `attn1_k, attn1_v` | 9.16 → 9.16 | −0.001 | −0.00004 | +0.0003 | **negligible (noise)** |
| run_0002 | `block_out, attn1_q/k/v` | 9.16 → 9.25 | **+0.083** | **+0.0028** | **−0.0024** | **small, correctly signed** |

**Null test (both runs):** self-injection into the *unperturbed* z1 = **exact
identity** (PSNR 120, SSIM 1.0, LPIPS 0.0). A standalone unit test confirmed
forward-hook write-replacement propagates. So the injection pathway is
**mechanically sound** — the result is real, not a plumbing artifact.

### Interpretation

The lever is **too sparse**. 4 mid-layers spread across a 48-layer stack
cannot hold a transplanted representation: between each injected layer there
are ~12 layers that recompute freely from the perturbed latent and wash the
graft out, and 31 of 40 steps have no injection at all. K,V-only at 4 layers
doesn't register; the full-residual (`block_out`) graft gives a real but tiny
(<0.1 dB) nudge in the right direction. This matches the PnP literature:
structure injection works when applied at a **contiguous block of early–mid
layers across most steps**, not at a sparse handful.

### It-1 next-experiment plan (tested in It-2 below — outcome: K,V still null)

1. Re-cache a contiguous dense layer block (done: layers 10–21).
2. Inject across that block at most/all steps (done: K,V, 24 steps).
3. Re-confirm with the A/B/C/D protocol.

---

## Results — It-2 dense K,V (2026-05-23, run_0004)

Re-cached the recon pass of ss4/ss8/ss0 at a **contiguous dense block** —
layers 10–21 (12), recon steps 0–23 (24), **K,V only**, **free-middle-token-
scoped** (570 MB/step, 13 GB/sample; via the new `token_scope` in
`feature_cache.py`). Injected dense K,V (strength 1.0, all 12 layers × 24
steps) on **ss4** across three variants (`config_dense.yaml`, multi-variant
loop). User stopped after ss4 to regroup, so ss8/ss0 were not run.

| variant | free-mid B→C PSNR | C−B ΔPSNR | D null (free-mid PSNR) | verdict |
|---|---|---|---|---|
| `cfg1` (CFG=1) | 9.16 → 9.13 | **−0.03** | **120.0 (exact identity)** | **null** |
| `cfg32` (CFG=3.2) | 9.20 → 9.14 | **−0.06** | 11.3 (CFG≠cache) | **null** |
| `null_cfg1` (CFG=1, empty prompt) | 9.14 → 9.15 | **+0.01** | 11.0 (prompt≠cache) | **null** |

(`run_0003` was a CFG=3.2 plumbing smoke test on the *sparse* 4-layer cache:
C−B = −0.08, consistent.)

### Interpretation — the "too sparse" hypothesis is FALSIFIED

Dense K,V (12 contiguous layers × 24 steps) is **just as null as the sparse
4-layer K,V** (It-1). The cfg1 **D null-test = exact identity** proves the
dense scoped-cache injection plumbing is flawless — so the null C−B is a real
property of K,V injection, not a bug. CFG=3.2 and a null prompt don't change
it. (cfg32/null_cfg1 D ≠ identity is *expected*: D is exact-identity only when
the edit pass matches the cache's CFG **and** prompt; both variants diverge
the trajectory from the CFG=1/real-prompt cache.)

**Why K,V can't transport here.** MasaCtrl-style K,V injection only binds when
the **queries** are shared between source and edit. We **fully re-noise** the
free middle, so its queries are random; attention `softmax(Q·Kᵀ)·V` with the
wrong Q cannot reconstruct source content out of source K,V. K,V injection
*preserves* structure across a small edit; it cannot *rebuild* structure into
a re-noised region — at any layer/step density.

### Open levers (each needs a re-cache — this cache holds K,V only)

1. **`block_out` (PnP)** — overwrites the whole block residual, so it is
   query-independent. It was the only positive It-1 signal (+0.08 at 4 layers).
   Re-cache `block_out` densely and inject it.
2. **Q+K,V** — also inject Q to restore query alignment, then K,V can bind.
3. **Reconsider the test.** Full re-noise is *harder* than the real goal
   (transplant a *good* transition's features into a *hard* clip ss5/ss6/ss9,
   where the edit clip already has a structured trajectory → queries aren't
   random). A milder/partial perturbation, or going straight to the cross-clip
   transplant with `block_out`, may be the better next step.

A future re-cache should grab `block_out + q + k + v` (free-middle-scoped) so
both (1) and (2) are testable without re-inverting.

---

## Results — It-3 Q+K,V with CFG cond-only fix (2026-05-25, run_0005)

Two changes vs run_0004:

1. **CFG cond-only fix.** `cond_only_at_cfg=True` on the injector writes
   the cached tensor only into row 1 (cond) of the batched
   `[uncond ; cond]` CFG>1 edit pass. Previously injection into both rows
   forced `v_uncond ≈ v_cond` in the injection mask and collapsed the CFG
   mix there.
2. **Q injection alongside K,V.** New cache at
   `sites=[attn1_q, attn1_k, attn1_v]` on the same dense grid (cache
   `outputs/videos/exp_040_ltx2_feature_cache/run_0003`, ss4 only,
   ~20 GB). Tests whether forcing source Q in addition to K,V binds
   when free-middle z is re-noised.

Headlines on ss4 free-middle pixel frames:

| variant                  | B_fm PSNR | C_fm PSNR | ΔPSNR  | ΔSSIM    | ΔLPIPS  | D_fm (null)         | verdict |
|--------------------------|-----------|-----------|--------|----------|---------|---------------------|---------|
| `cfg1_kv`                | 9.165     | 9.134     | −0.031 | +0.0015  | +0.0021 | **120.00 identity** | null    |
| `cfg32_kv_condonly`      | 9.204     | 9.186     | −0.018 | −0.00005 | −0.0009 | 11.45 (CFG≠cache)   | null    |
| `cfg1_qkv`               | 9.165     | 9.144     | −0.020 | +0.0008  | +0.0006 | **120.00 identity** | null    |
| `cfg32_qkv_condonly`     | 9.204     | 9.219     | **+0.015** | −0.0027 | +0.0008 | 11.56 (CFG≠cache)   | null    |

All four inside noise. `cfg32_qkv_condonly`'s +0.015 dB is the only
positive ΔPSNR but its ΔSSIM goes negative — no coherent signal.

### Interpretation

- **CFG cond-only was correct but insufficient.** It eliminates the
  CFG-collapse artefact, but the residual C−B at cfg32 is the same noise
  floor as cfg1. So the CFG mix collapse was real but not the bottleneck.
- **Q misalignment is not the only obstruction.** Forcing source Q in
  addition to K,V didn't bind. Even with the right Q + K + V, injecting
  only at the predictor substep cannot transport structure into a fully
  re-noised free middle.
- **Likely remaining limiter: corrector-substep dilution.** The 2nd-order
  midpoint integrator updates z via `z + dtau · v_mid`, where v_mid comes
  from the un-injected corrector at σ_mid. Each predictor write is
  filtered through one un-injected transformer call before z advances.
  The current cache has `substeps_midpoint: predictor` only.
- **Companion baseline `exp_042`** was run alongside (straight LTX-2 C2V
  production: Euler + CFG=3.2 + drop1, NO inversion, NO injection) on
  ss4/ss8/ss0. Output: `outputs/videos/exp_042_ltx2_c2v_drop1_baseline/run_0001/`.

### Status — three iterations all converge on null

- It-1 (sparse 4 layers): K,V null, full QKV+block_out small positive.
- It-2 (dense K,V): null at any density.
- It-3 (dense Q+K,V, CFG cond-only): null at every variant.

Open levers, in priority order: (a) cache `corrector` substep too and
inject at both substeps; (b) cache `block_out` densely and test PnP-style
transplant; (c) sweep strength below 1.0; (d) shift to milder
perturbation (partial re-noise α ∈ (0, 1)) or the real cross-clip
transplant setting.
