# Smoke-transition injection arc (exp_044+) — reconstructing the free-middle

Goal (set 2026-06-02): disentangle the shadow_smoke *transition* from
clip-specific scene features and inject it so the **free-middle** of the
exp_033 inverted clips reconstructs above **PSNR 18**. The transition = the
dark billowing smoke burst in free latent frames **4–12** (mask=0), pixels
~[25:97]. See `project_smoke_transition_regen_collapse` memory for the
grounding (per-frame PSNR decomposition, the recon vs regen split).

Success metric: **production-regen free-middle PSNR** (Euler, CFG=gen). exp_033
baseline median ≈ 14.6 (full-frame regen 17.92). recon (RF-midpoint, CFG=1)
free-mid median ≈ 22 but is **solver self-consistency** (the matching inverse
solver retraces the inversion), not a production result.

---

## exp_044 — regen CFG sweep (REFUTED the CFG hypothesis)

**Hypothesis:** the free-middle collapse (recon 27–38 dB → regen 11–18 dB on
z1-rich clips) is caused by CFG=4 + neg-prompt washing out z1's encoded smoke
in the unclamped free tokens. Lowering CFG should recover it.

**Design:** regen-only (load cached exp_033 z1, skip inversion), sweep global
CFG {4,1,2} + token-localized CFG (free_cfg on mask=0, anchor_cfg on mask>0).
Pilot ss0/ss7/ss5.

**Mechanistic note (verified):** anchor tokens are hard re-clamped to
`clean_latents` every step, so anchor CFG has ZERO effect → token-localized
CFG with free_cfg=X is provably equivalent to global CFG=X on the free tokens.
Confirmed: `loc_f1_a4` (16.36) ≡ `cfg1_global` (16.44).

**Results (free-mid PSNR):**

| clip | cfg4 | cfg1 | cfg2 | loc_f1_a4 | loc_f1.5_a4 | recon(midpoint cfg1) |
|------|-----:|-----:|-----:|----------:|------------:|---------------------:|
| ss0  | 16.43 | 16.44 | 17.79 | 16.36 | 17.46 | 37.6 |
| ss7  | 18.71 | 12.82 | — | — | — | 30.1 |

**REFUTED.** CFG is not the lever:
- ss0: cfg1 ≈ cfg4 (16.4); cfg2 a mild +1.4 dB optimum (17.79) — small,
  nowhere near recon's 37.6.
- ss7: cfg1 (12.82) is *worse* than cfg4 (18.71) — opposite sign. CFG effect
  is clip-dependent and inconsistent.

**Mechanism (the real finding):** the recon→regen gap is **solver
self-consistency**, NOT CFG. recon uses the same RF-midpoint discretization as
the inversion, so it retraces the inversion path and recovers z0's middle. The
production Euler sampler uses a different discretization and diverges in the
free middle *at any CFG* — z1 is a noise tuned to the midpoint inverse, only
the matching solver retraces it. The production sampler produces only
**generic prompt-smoke** in the free middle, not the source's specific smoke
turbulence. (Verified earlier: z1's free-middle is gaussian-identical across
all clips — std~1.02, per-token norm ~11.5≈√128 — and whether the solver
recovers z0 depends entirely on solver/clip, recon_mid_rel_z0 0.07–0.25 rich
vs 0.63–1.12 poor.)

**Implication:** to get production-regen middle > 18 we must INJECT a
source-specific smoke signal — recovering it from z1 is impossible for the
production sampler. This is the postmortem's "bootstrap-middle" idea
(exp_035/036, FAILED with a generic prior ~orthogonal to z0) but with a
**smoke-family prior** (resid 0.43–0.84, far closer). exp_032 already proved
pinning a *good* middle gives production regen ~31, so the question is purely
prior quality.

ss7/ss5 not completed — refutation unambiguous from ss0 + ss7-partial + the
all-10 cfg4 baseline. Pod repurposed to exp_045.

---

## exp_045 — middle-prior decode feasibility (ceiling for pin/inject) — WALL

Upper bound of any latent-pin/inject method: assemble perfect z0-anchors + a
candidate free-middle prior, decode, free-mid PSNR vs source. Deployment frame
(per user): inject smoke EXTRACTED FROM DONOR SAMPLE TRANSITIONS into a target
for which we only have endpoints. Leave-one-out = 9 donors → held-out target.

**Median free-mid PSNR across 10 clips:**

| prior | median | mean | what it is |
|-------|-------:|-----:|------------|
| `src` (target's own z0 middle) | **120.0** | 120.0 | sanity — decode is faithful |
| `gauss` (rms-matched noise) | 10.05 | 9.91 | noise floor |
| `endpoint_hold` | 8.57 | 8.44 | freeze start anchor |
| `endpoint_interp` | 10.54 | 10.64 | morph start→end (endpoints only) |
| `smoke_bcast_loo` | 10.30 | 10.20 | DONOR smoke, broadcast channel-state |
| `smoke_bcast_all` | 10.37 | 10.34 | donor smoke incl. self, broadcast |
| `smoke_spatial_loo` | 10.10 | 9.75 | DONOR smoke, spatially-resolved |
| `keepspatial` | 10.48 | 10.59 | endpoint spatial + donor smoke channel-state |

**WALL.** Every deployable prior sits **at the noise floor (~10.5 dB)**. Donor
smoke adds ≤ +0.4 dB over gauss. The marginal leaders (`endpoint_interp`,
`keepspatial` ~10.5) gain from ENDPOINT structure, not donor smoke. Meanwhile
the model's own production generation (prompt-smoke) gives free-mid ~14–18 —
*strictly better than any pin* (pinning removes the model's generative smoke).

**Mechanism / why the 68%-shared-latent signal vanishes in PSNR:** the shared
cross-clip smoke is **low-frequency channel-mean (darkening)**; pixel PSNR is
dominated by **high-frequency turbulence**, which is clip-specific and
uncorrelated across clips. A latent that's 0.7-cosine-aligned to the target
decodes to a perceptually-smoke-but-pixel-wrong frame ≈ noise PSNR.

**Conclusion (information limit):** the target's *specific* smoke turbulence is
NOT derivable from {endpoints + donor samples}. So **free-middle PSNR > 18 vs a
specific held-out clip is information-limited and not reachable by injection of
donor-extracted smoke** (latent substrate). `src`=120 vs deployable ≈10.5 is the
whole clip-specific-information gap; the target's own z1 via the matching
midpoint solver (recon, ~22) is the only thing that bridges it, and that needs
the target's full clip (unavailable at generation time).

**What this does NOT rule out:** (a) full-frame regen PSNR > 18 (currently
17.92 median — anchors carry it; modest free-middle gains suffice); (b) a
PERCEPTUAL/distributional metric where donor smoke clearly helps (the injected
middles ARE smoke-colored); (c) small gains from velocity/attention injection
making the *generated* (not pinned) smoke donor-informed — predicted to stay
below free-mid 18 due to the same texture wall, untested.

**User pivot (2026-06-02):** success redefined to PERCEPTUAL smoke quality
(not pixel-PSNR). Deployment = {target endpoints} + {donor sample transition(s)}.

## exp_046 — perceptual donor-smoke injection (latent pinning) — WIN

Pin a SINGLE donor's REAL free-middle latent (real texture + billowing
dynamics) into a target, decode, judge perceptually (free-middle signals +
visual) vs REAL source and the BASELINE (exp_033 prompt-only regen). Same-grid
donor/target. Run in `outputs/videos/exp_046_smoke_perceptual_inject/run_0002`.

**Free-middle signals (the prompt baseline lacks billowing DYNAMICS; donor
injection restores them):**

| clip | metric | REAL | BASELINE prompt | DONOR-injected |
|------|--------|-----:|----------------:|---------------:|
| ss6 | tdiff | 0.077 | 0.041 | 0.064 (ss0) |
| ss6 | lum   | 0.402 | 0.292 (too dark) | 0.374 |
| ss1 | tdiff | 0.078 | **0.024** (static) | **0.074** (ss5) |
| ss1 | lum   | 0.407 | 0.384 | 0.403 |
| ss9 | tdiff | 0.047 | 0.042 | 0.077 (ss5, overshoot) |

**Visual (montages, hard clips ss6/ss1):** the prompt baseline keeps the scene
largely visible with mild darkening — a near-static pass-through, not a smoke
event. Donor-injected shows a coherent dark billowing smoke mass with real
dynamics — clearly reads as smoke and beats the baseline.

**LIMITATIONS:** pinning SPLICES the donor's *specific* smoke (not adapted to
the target scene) → onset/offset blending with the target's anchors is
imperfect and saturation overshoots (donor carries its own scene colors).
`donorblend` (0.7·donor + 0.3·endpoint-interp) softens but dilutes the smoke.
LPIPS-to-source stays high (~0.73–0.83) — expected (it's a *different* smoke;
perceptual, not pixel-match).

### exp_046 run_0003 — smoke-DELTA disentanglement (latent)

`smokedelta:<donor>` = `target_endpoint_interp + (donor_middle −
donor_endpoint_interp)` — extract the donor's smoke as a delta from its OWN
scene-baseline, add onto the target's scene-baseline. Removes the donor's scene.

| clip | variant | LPIPS↓ | sat (real) | tdiff (real) | lum (real) |
|------|---------|-------:|-----------:|-------------:|-----------:|
| ss6 | donor:ss0 | 0.733 | 0.448 (0.222) | 0.064 (0.077) | 0.374 (0.402) |
| ss6 | smokedelta:ss0 | **0.648** | **0.289** | 0.031 (static) | 0.354 |
| ss1 | donor:ss5 | 0.777 | 0.333 (0.309) | 0.074 (0.078) | 0.403 (0.407) |
| ss1 | smokedelta:ss5 | **0.709** | **0.298** ✓ | **0.089** ✓ | 0.509 (bright) |

**smoke-delta FIXES saturation + lowers LPIPS-to-source** (disentanglement works
— donor scene removed). But introduces a **pin-vs-delta TRADEOFF** (visual,
montages): donor-pin = dramatic coherent billow but donor's content/identity;
smoke-delta = faithful target scene + correct color but UNDER-OCCLUDED (smoke
reads as a dark wash, not a full billow; ss1 lum 0.509 too bright). Over-drive
(:1.3) darkens only modestly. **Neither pure-latent method achieves "full
dramatic smoke billow that is also adapted to this clip's scene."**

**Conclusion of the latent substrate:** donor injection perceptually beats the
prompt baseline (more smoke-like, more dynamic), but latent pin/delta cannot
simultaneously give full occlusion AND target-scene adaptation. That requires
GENERATIVE injection — the model synthesizing a smoke billow conditioned on
donor smoke features while adapting to target anchors.

### exp_046 run_0004 — TEMPORALLY-WINDOWED donor injection (WINNING latent recipe)

`tempblend:<donor>` = `(1−w(t))·target_endpoint_interp + w(t)·donor_real_middle`,
where `w(t)` is a Gaussian bump over the free frames peaked at the darkest
latent frame (~8). So: target scene at the smoke onset/offset (continuity),
donor's full real smoke at the occlusion PEAK (where the scene is occluded so
donor identity is hidden). Resolves the pin-vs-delta tradeoff. Deployable:
target endpoints + a donor sample's real middle; no target middle used.

Signals (ss6 tempblend:1.4): lum 0.388 (real 0.402), sat 0.252 (real 0.222) —
both close to real, vs baseline lum 0.292/sat 0.124. ss1 tempblend: tdiff 0.067
(real 0.078) vs baseline 0.024. **Visual (montages): the target's own scene
transitions into a dark smoke billow at the peak and back to the scene** —
"emerges from this clip → full smoke → returns to this clip". Beats: baseline
(static darkened scene), donor-pin (donor's content throughout), smoke-delta
(under-occluded wash). NB LPIPS-to-source is a poor smoke metric — it rewards
the scene-preserving baseline; judge by darkening/dynamics/occlusion + visual.

**STATUS: perceptual goal met via latent injection.** Deployable recipe =
temporally-windowed injection of a donor sample's real smoke onto the target's
endpoint-interpolated scene. Validated on the two hardest clips (ss6, ss1).
Cheap (decode-only, no solver).

## exp_047 — VELOCITY-guided smoke generation (2nd working recipe)

Run the production sampler (Euler) but in the x0 (clean) domain pull the free
tokens toward the tempblend smoke target by `guide_weight g` each step
(`_x0_clamp_velocity`), so the MODEL synthesizes the smoke (seamless, coherent)
following the donor's darkening/dynamics — vs exp_046's static latent splice.
Deployable: target endpoints + one donor sample (same grid).

**Results (g sweep, hard clips):** guidance moves signals toward real —
ss6 lum 0.294(g0)→0.374(g0.8) [real 0.402], sat 0.157→0.283 [0.221];
ss1 tdiff 0.031(g0)→0.067(g0.8) [real 0.078], sat 0.131→0.222 [0.310].
**CFG-agnostic once guidance dominates: g0.8_cfg1 ≡ g0.8_cfg4** (same as the
exp_044 finding that the free tokens are what matter). lowsigma schedule ≈
const. Signals converge to ~tempblend (soft-pin compounding) BUT the output is a
single coherent model generation (no splice seam) with better mid-frame motion.
Visual (montages): guided-gen smoke is integrated into the target scene and more
seamless than the tempblend splice; both beat the static baseline.

## CONCLUSION — perceptual goal met, two deployable recipes

From {target endpoints + a donor sample transition}, two methods produce a
coherent smoke transition that emerges from and returns to the target's scene,
beating the prompt-only baseline (static darkened scene):
1. **tempblend** (latent, decode-only, cheap) — windowed donor-smoke splice.
2. **velocity-guided generation** (g≈0.8) — model synthesizes smoke steered to
   the donor target; seamless, CFG-agnostic; the genuine "inject at gen time".

Pixel-PSNR>18 vs the specific clip remains information-limited (turbulence is
clip-specific) — that was the wrong metric; perceptual quality is the right one
and is achieved. Open follow-ups: quantify perceptually (CLIP smoke-score /
FVD to the smoke family), validate on all 10 clips, and the block_out feature
substrate (exp_041 machinery) for finer turbulence control.

---

## exp_049 — σ-matched recon x̂₀-trajectory injection (self + donor, window sweep)

**Question.** exp_048 injected a clip's *static* final `z0_recon` into the
production Euler regen free-middle at every step → self free-mid only **13.05**
@ g=1 (off-manifold: a fully-sharp clean prediction forced at high σ). Does
injecting the **σ-MATCHED** step of the recon's own coarse→fine trajectory
`x̂₀(σ_i) = z_in − v_pred·σ_i` instead recover the reconstruction, and where in
the σ-schedule is the transition carried?

**Setup.** Producer = exp_040 `config_recon_traj17.yaml` (velocity-only cache,
recon all 40 steps, 10 clips → `run_0002`; 2.88 MB/step, 112 MB/clip). Consumer
derives `x̂₀(σ_i)` per recon step (no library change). During production Euler
regen (CFG=4), for steps in an injection window, blend the free-token clean
prediction toward `x̂₀(σ_i)` by weight g; recon & regen share `_build_sigma_grid`
→ step i is σ-matched (asserted at load). Windows: early[0,13) mid[13,26)
late[26,40) all[0,40). self g=1 (mechanism), donor g=0.8 (deployable). 3 clips ×
9 variants (`run_0006`).

**FREE-MID PSNR (vs source):**

| clip | base | self_all | self_early | self_mid | self_late | donor_all | donor_early | donor_mid | donor_late |
|------|------|----------|-----------|----------|-----------|-----------|-------------|-----------|------------|
| ss0  | 14.70| **33.26**| 17.75     | 24.72    | **33.26** | 8.30      | 14.56       | 14.01     | 8.35       |
| ss6  | 12.78| 15.18    | 14.90     | 14.97    | 15.18     | 8.45      | 12.12       | 12.16     | 8.47       |
| ss1  | 10.61| 11.88    | 12.01     | 11.87    | 11.88     | 7.88      | 11.19       | 11.11     | 7.91       |

**Findings.**

1. **σ-matched self injection RECOVERS the reconstruction — but only for
   z1-rich clips.** ss0: base 14.70 → self_all **33.26** (lpips_fm 0.475→0.032),
   i.e. injecting the recon's own σ-matched trajectory makes the production
   Euler+CFG=4 sampler reproduce the faithful reconstruction. This closes the
   recon→regen solver-mismatch gap (exp_044) *when the information exists*.
   The σ-MATCHED trajectory is essential: static-target (exp_048) gave 13.05;
   σ-matched gives 33.26 on the same kind of clip.

2. **Recovery is gated by z1-richness (exp_044 dichotomy holds).** Only ss0
   recovers; ss6 (15.18) and ss1 (11.88) are z1-poor — their inversions never
   encoded the middle, so there is nothing to recover (any injection just nudges
   to the sampler floor, +1–2.4 over base). ss0 rich / ss1+ss6 poor is
   consistent with exp_044's ss1/5/6/9-poor grouping.

3. **For z1-rich recovery, late-σ carries it (It-4 confirmed, x0-domain).**
   ss0 self windows are monotone: early 17.75 < mid 24.72 < **late 33.26 = all
   33.26**. The late window (steps 26–39) ALONE achieves the full recovery;
   early/mid are partial. Independent replication of the feature-injection It-4
   result (late-σ steps 27–39 carry the transition), now for x0-domain
   trajectory injection.

4. **z1-poor clips saturate across all windows** (any window ≈ all; no
   localization, no stacking) — nothing to recover.

5. **Donor (deployable cross-clip transfer): inject EARLY, never all/late.**
   donor_all/late tank free-mid to ~8 and over-saturate (sat→0.45–0.48) by
   forcing the OOD donor's fine-detail into the late steps; donor_early/mid
   preserve structure (≈ base PSNR) and nudge saturation/dynamics toward real.
   So for donor transfer use the EARLY (high-σ) window where the sampler
   re-integrates the OOD content. PSNR-vs-specific-clip stays information-limited
   (exp_045 wall) — donor value is perceptual, not PSNR.

**Conclusion.** σ-matched late-σ self-trajectory injection reproduces the
reconstruction (33 dB) for z1-rich clips — the mechanism works and is
late-σ-localized — but it is not deployable (needs the target's own recon
trajectory = the full target clip), and on z1-poor clips there is nothing to
recover. The deployable donor path remains information-limited on PSNR; its one
actionable refinement is **early-window** injection (gentle, structure-preserving)
over all/late (corrupting). No PSNR>18 deployable win; same wall as exp_045.

**Infra note.** Repeated ~78 GB CUDA OOM during this exp was a **missing
`@torch.inference_mode()` on `main()`** (lost when forking from exp_048 and
inserting `encode_prompt_bundle`): without it the full-video VAE encode builds an
autograd graph that `z0_packed` keeps alive (~75 GB live activations) → OOM at
text-encoder load. Neither `enable_model_cpu_offload` nor manual `.to("cpu")`
placement can evict live autograd tensors — chased both as red herrings. Fix is
the one-line decorator; see memory `ltx2_inference_mode_oom_footgun`. Results in
`outputs/videos/exp_049_smoke_signature_inject/run_0006/`. A100 PCIe torn down.
