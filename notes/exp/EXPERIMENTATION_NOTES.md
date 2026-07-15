# Fully Manual Notes
Written by me, don't add or modify!

# Exp 20, LTX2 C2V Diffusers
* Baseline implementation of C2V task (conditioned generation on 2 video endpoints) with diffusers library.

# Exp 21, Trajectory Logging on top of Exp20
* Baseline implementation of Trajectory Logging of Exp 20.
* VAE Latents, Velocity Fields, Transformer Hidden Features are hooked and saved.
* Analyzed within notebooks.

**TODO:** Observe and State the Results:

# Exp 23, VAE Latent LERP
* Question: Is the default behavior that LTX-2 collapses, the dissolve effect, when conditioned on semantically & contextually distant endpoints, is actually collapsing to a straight path at the VAE Latent Space between endpoints.
* Does LTX-2 being a Rectified Flow Matching model have an impact on that? Do that type of models have a tendency to collapse to the shortest-paths when the path with curvature does not have enough force on the manifold.   

* Results:
    - VAE Lerp between endpoints seem like the dissolve transitions produced by LTX-2.
    - Will check with different transition region lenghts, and state the result better.

# Exp 27, Flow Inversion Consistency Check
* Flow inversion is working. 
    Results:
        __main__  ──────────────────────────────────────────────────────────────────────
        __main__  RF inversion summary: 3/3 samples passed gate (LPIPS < 0.050)
        __main__    [PASS] class2__mallard-fly__mallard-water             LPIPS mean=0.0044  max=0.0154  steps=30
        __main__    [PASS] class5__car-roundabout__bus                    LPIPS mean=0.0020  max=0.0057  steps=30
        __main__    [PASS] class8__blackswan__boat                        LPIPS mean=0.0128  max=0.0513  steps=30
        __main__  ──────────────────────────────────────────────────────────────────────


* Need to match the steps, trajectory changes! 30 - 40
* Doing the inversion & reconstruction at CFG=1 is okay, only positive prompt effect is occuring.


# Exp 29, Improved Flow Inversion for Better & Matching Settings
* Fixed 
    - Generation step: 40
    - CFG=1 is fixed for inversion-reconstruction round-trip.
    - Added a regeneration path, in generation settings starting from inverted noise, cfg=3.2
    - Audio was randn, made zeros!
    - Added more granular logging.

* Still for generated videos, not out-of-distribution inversion & reconstruction test.

## Results:  
        Results are nice! Inv + Recon works perfectly for LTX2-Generated C2V Videos.

    --------------------------------------------------

    ### Recon (`[inv_recon]` vs Stage‑1 source)

        sample_id                          |   PSNR   |  SSIM   |  LPIPS 
        -----------------------------------|----------|---------|---------
        class2__mallard-fly__mallard-water |  45.53   | 0.9959  | 0.0031
        class5__car-roundabout__bus        |  45.48   | 0.9966  | 0.0019
        class8__blackswan__boat            |  44.23   | 0.9915  | 0.0050
        **mean (n=3)**                     |**45.08** |**0.9947**|**0.0033**

    --------------------------------------------------

    ### Regen (`[inv_regen]` vs Stage‑1 source)

        sample_id                          |   PSNR   |  SSIM   |  LPIPS 
        -----------------------------------|----------|---------|---------
        class2__mallard-fly__mallard-water |  31.19   | 0.8122  | 0.1609
        class5__car-roundabout__bus        |  32.48   | 0.8734  | 0.0797
        class8__blackswan__boat            |  35.26   | 0.8897  | 0.0647
        **mean (n=3)**                     |**32.98** |**0.8584**|**0.1018**

    --------------------------------------------------

        60 invert+recon steps (retry; regen still 40 Euler in log):

        Recon

            sample_id                          |   PSNR   |  SSIM   |  LPIPS 
            -----------------------------------|----------|---------|---------
            class2__mallard-fly__mallard-water |  46.03   | 0.9963  | 0.0026
            class5__car-roundabout__bus        |  45.82   | 0.9967  | 0.0018
            class8__blackswan__boat            |  46.25   | 0.9957  | 0.0026
            **mean (n=3)**                     |**46.03** |**0.9962**|**0.0023**

        Regen

            sample_id                          |   PSNR   |  SSIM   |  LPIPS 
            -----------------------------------|----------|---------|---------
            class2__mallard-fly__mallard-water |  31.15   | 0.8127  | 0.1620
            class5__car-roundabout__bus        |  32.32   | 0.8693  | 0.0801
            class8__blackswan__boat            |  35.31   | 0.8859  | 0.0705
            **mean (n=3)**                     |**32.93** |**0.8560**|**0.1042**





# Exp 30, Flow Inversion with Real Clips (Shadow Smoke Transition)
* Applied the same methodology with exp 29, but with existing videos (so not self-generated)
    - Generation step: 40
    - CFG=1 is fixed for inversion-reconstruction round-trip.
    - Made the first and last 1 sec of the videos the endpoint conditions! And made the inversion and recon with them fixed as conditions!

## Results
* Results are bad. A lot of vids are broken! 


# Until Exp 39, Inv+Recon improvements

# Exp 40, Feature Cache based on Inv + Recon at Exp 33
Inserted hooks to cache Q, K, V, Velocity Fields, Residual outputs (block_out), z0 (clean latent), z1 (embedded noise), ff_out, Audio to Video Attention

For each block index `l` in `layer_indices`, forward hooks are registered on:

| Site name | Module hooked | Output shape | Notes |
|---|---|---|---|
| `block_out` | `transformer_blocks[l]` | `[B, N_video, 4096]` | Block returns `(video, audio)` tuple; we keep `[0]`. |
| `attn1_q` | `transformer_blocks[l].attn1.to_q` | `[B, N_video, 4096]` | **Pre-RMSNorm, pre-RoPE.** |
| `attn1_k` | `transformer_blocks[l].attn1.to_k` | `[B, N_video, 4096]` | **Pre-RMSNorm, pre-RoPE.** |
| `attn1_v` | `transformer_blocks[l].attn1.to_v` | `[B, N_video, 4096]` | V is not norm'd / RoPE'd. |
| `attn2_q` | `transformer_blocks[l].attn2.to_q` | `[B, N_video, 4096]` | Video → text cross-attn Q. |
| `attn2_k` | `transformer_blocks[l].attn2.to_k` | `[B, N_text≈128, 4096]` | Text-side K. |
| `attn2_v` | `transformer_blocks[l].attn2.to_v` | `[B, N_text≈128, 4096]` | Text-side V. |
| `ff_out` | `transformer_blocks[l].ff` | `[B, N_video, 4096]` | Video FFN output. |
| `audio_attn1_q/k/v` | `transformer_blocks[l].audio_attn1.*` | `[B, N_audio, 2048]` | Audio self-attn. |
| `a2v_q/k/v` | `transformer_blocks[l].audio_to_video_attn.*` | Q: `[B, N_video, ·]`, K/V: `[B, N_audio, ·]` | Audio → video cross-attn. |


# Exp 41, C2V Attention Feature Injection 
* Ran generation for 4 cases
    * Perturbed (random noise, scaled to inverted z1 channelwise rms) baseline
    * Perturbed Injected (K, V) injection at steps written in the config, to the layers written in the config
    * Target Recon (The source of features)
    * Self-Injection (inject the features extracted into the inverted z1, sanity check because Flow Matching is deterministic, they are already the features that should occur in the generation, so we expect the output to be the same as the target recon even after injection)

    ## Current Setup
        Config knobs

        |   Bucket   |           Field           |                       Value                        |                   Role                    |
        |------------|--------------------------|----------------------------------------------------|-------------------------------------------|
        | model      | model_id                 | Lightricks/LTX-2                                   | Base C2V pipeline                         |
        | source     | cache_run_dir            | outputs/videos/exp_040_ltx2_feature_cache/run_0003  | exp_040 dense Q+K+V cache run             |
        | source     | samples                  | ["shadow_smoke_4"]                                 | Single sample                             |
        | inference  | num_frames / frame_rate  | 121 / 24.0                                         | Output shape                              |
        | inference  | num_inference_steps      | 40                                                 | Midpoint recon steps                      |
        | inference  | guidance_scale (default) | 1.0                                                | Per-variant overridable                   |
        | perturb    | free_middle_latent_frames| [4..11]                                            | Token region reseeded in z1               |
        | perturb    | seed                     | 1234                                               | Reseed seed (≠ gen seed 42)               |
        | perturb    | match_rms                | true                                               | Preserve noise level                      |
        | injection  | phase                    | recon                                              | Cache phase (matches exp_040 recon pass)   |
        | injection  | substep                  | predictor                                          | Inject on σ_curr only; corrector free     |
        | injection  | layers                   | [10..21] (12 layers)                               | Dense mid block                           |
        | injection  | steps                    | [0..23] of 40                                      | Early/content-forming half                |
        | injection  | sites (default)          | [attn1_k, attn1_v]                                 | Per-variant overridable                   |
        | injection  | strength                 | 1.0                                                | Hard replace                              |
        | injection  | region                   | free_middle                                        | Token scope                               |
        | runtime    | seed                     | 42                                                 | Generation seed                           |
        | outputs    | dir                      | outputs/videos/exp_041_ltx2_feature_inject         | Run dir parent                            |

        Variants in config_qkv_condonly.yaml

        |    Variant name     |  gs  |  sites   |     cond_only_at_cfg      |
        |---------------------|------|----------|---------------------------|
        | cfg1_kv             | 1.0  | K,V      | — (CFG=1, batch flag off) |
        | cfg32_kv_condonly   | 3.2  | K,V      | true                      |
        | cfg1_qkv            | 1.0  | Q,K,V    | —                         |
        | cfg32_qkv_condonly  | 3.2  | Q,K,V    | true                      |

        Run-time mechanics (run.py)

        |       Item        |                                             Detail                                             |
        |-------------------|-----------------------------------------------------------------------------------------------|
        | reference         | z0_recon (from cache)                                                                         |
        | B pass            | reconstruct(z1_pert), no injection                                                            |
        | C pass            | reconstruct(z1_pert), injector attached                                                       |
        | D pass            | reconstruct(z1), injector attached (self-inject null)                                         |
        | Hard anchor clamp | z = z*(1-mask) + clean_latents*mask — applied at entry and after every step (both substeps)   |
        | Metric scope      | full clip + free-middle pixel slice [(4−1)·8+1 .. 11·8+1) = [25, 89) (clamped to video length)|
        | Headline          | free_middle: C − B for PSNR / SSIM / LPIPS                                                   |
        | Prompt resolution | variant.prompt → static["prompt"] → final                                                     |
        | Negative prompt   | always static["negative_prompt"] — no variant override path right now                         |
        | CFG batch         | At gs>1, transformer batches [uncond; cond]; injector writes both rows unless cond_only_at_cfg=true |

    ## Notes:
    * Remove the prompt, (because of the result of exp42, only normal generation with the basic smoke transition yields nearly the same results, isolate the effects of the injection) 

    ## Versions
    1. No-Prompt
        * No-Prompt: Remove both positive and negative prompts. Or only negative? Removing both seems plausable because all the Q, K, and V of the free middle are dependent on the text tokens too, or is it so for the T2V Cross-Attention Transformer Blocks? If so injecting the RECON features in the Self-Attention Transformer Blocks will affect the Block Output? And make the CFG=3.2 case not collapse into CFG=1 case? Actually yes because on every step if CFG is bigger than 1, cond and uncond passes will diverge because of the Delta introduced in the Self-Attention Transformer Blocks.

        * So, remove both prompts. 

        ### Result:
        * No visible difference at cfg=1

    
    ## IDEAS:
    * Use Endpoint Clip Length of 16
        * To isolate the existing smoke effects within the clips. 
        * The current injecting is calculating the Endpoint Clip Length itself, fix that!
    * Check the conditioning Encoding: "Do the conditions' encodings capture the ground-truth transition information? Or encoded in isolation?"
    * Inject block_out early?
        - Block Out: 
            * What's inside one `LTX2TransformerBlock` (forward, in order):

            | Line  | Code                                                         | Description                    |
            |-------|--------------------------------------------------------------|--------------------------------|
            | 466   | `attn_hidden = attn1(norm1(video))`                          | VIDEO self-attn                |
            | 471   | `video += attn_hidden * gate_msa`                            | Apply gated VIDEO self-attn    |
            | 484   | `attn_audio = audio_attn1(audio_norm1(audio))`               | AUDIO self-attn                |
            | 489   | `audio += attn_audio * audio_gate_msa`                       | Apply gated AUDIO self-attn    |
            | 493   | `attn_hidden = attn2(norm2(video), text)`                    | VIDEO ↔ TEXT cross-attn        |
            | 499   | `video += attn_hidden`                                       | Apply VIDEO cross-attn         |
            | 502   | `attn_audio = audio_attn2(audio_norm2, text)`                | AUDIO ↔ TEXT cross-attn        |
            | 508   | `audio += attn_audio`                                        | Apply AUDIO cross-attn         |
            | 555   | `a2v_attn = audio_to_video_attn(video, audio)`               | AUDIO → VIDEO cross-attn       |
            | 563   | `video += a2v_gate * a2v_attn`                               | Apply AUDIO→VIDEO cross-attn   |
            | 573   | `v2a_attn = video_to_audio_attn(audio, video)`               | VIDEO → AUDIO cross-attn       |
            | 581   | `audio += v2a_gate * v2a_attn`                               | Apply VIDEO→AUDIO cross-attn   |
            | 585   | `ff_output = ff(norm3(video))`                               | VIDEO feedforward (FFN)        |
            | 586   | `video += ff_output * gate_mlp`                              | Apply gated VIDEO FFN          |
            | 589   | `audio_ff_output = audio_ff(audio_norm3(audio))`             | AUDIO feedforward (FFN)        |
            | 590   | `audio += audio_ff_output * audio_gate_mlp`                  | Apply gated AUDIO FFN          |
            | 592   | `return (hidden_states, audio_hidden_states)`                | Tuple output                   |
       
        - So Block_Out in our context gets the final tuple after ALL the Attentions and FFNs! And it only gets the video residual stream!
        - block_out is the heaviest possible intervention site. At strength=1.0 you are overwriting the entire
          post-block video residual — every sublayer's contribution (self-attn output + text-cross-attn output +
          audio-cross-attn output + FFN output, all summed with the pre-block input) gets replaced by the source's
          residual at that depth.
        - I think as the saturation test, injecting the block-out is the best place to start to see if injecting actually does anything at all? And doing this in 4 different settings, one for each layers all timesteps, one early timesteps, mid timesteps, and last timesteps? Would that make sense? First start with hard injection? 

        * !!! Block_out blocks the residual stream, so the only important thing is the biggest injected layer? Because it overwrites the residual stream too!
        - Check the caching? Effect of prompt? Effect of Conditionings? Effect of the Audio? 


    * A paper (QK-Edit Revisiting Attention-based Injection in MM-DiT) says that in MM-DiT Architectures do not directly inject into the self-attention thingies! will check!


    ###### DETERMINE WHAT SHOULD BE DISENTANGLED

    ### VAE Latent Space Injection
    1. Can we place an attractor (Smoke Transitioning Manifold?) & Repulsor (VAE Latent LERP between condition endpoints), change the forces of attraction & repulsion along the process?
        * LERP Between Endpoints
        * Determine the cluster of smoke transition
            * Attract based on Video and Flow Time

    ### Velocity Field Injection
    2. To point at the specific shadow transition, don't directly inject the velocities but do test-time-optimization based on the predicted z0 at each step, globalizing the target.


# Exp 48, Velocity Injection
* Current version is injecting the z0 in each step, instead of that, make predicted z0 injection at the matching step cached from the reconstruction step!
* Inject only in specific timesteps (early-mid)
* Adjust the strength.


* RUN THE Caching independently.
* Then injection on generation time! Run nicely!
* Run with 25 Frames Condition Video Length 

* Which noise do we start from? Inverted z1? If so problematic? Which z1?







# Exp 50, LoRA Baseline
* Pivoted to training solutions.
* With exp 50 & 51, did LoRA Training in different settings.
    1. T2V LoRA training. 
        - Given 10 shadow smoke samples, one prompt for each. Trained LoRA
    2. I2V LoRA training.   
        - Given 10 shadow smoke samples, one prompt for each, first-frame conditioning with p=0.5. Trained LoRA
    3. C2V LoRA training.
        - Given 10 shadow smoke samples, one prompt for each, first-last clip conditioning (C2V setting) with p=1. Trained LoRA

* Q: Is the baseline LoRA training on LTX-2 adequate to transfer Shadow Smoke Transition between unseen endpoints seamlessly using 10 reference samples.
- Answer: Yes!

***State:***
* Base model fails to generate the described transition. Wrong place, wrond mechanics, no movement etc.

***Problems:*** 
    1- LTX-2's conditioning training is incredibly robust. 
    2- Trained using 10 training samples, can be decreased.
    3- Evaluation problem: What does "seamless" mean? How do we measure success.
    4- Is it specific to Shadow Smoke Transition samples? Does it generalize to other types of transitions?
    5- Was the endpoints genuinely OOD? Or similar to Shadow Smoke Transition Samples? And what was N (pairs x seeds)?
        - Add endpoint pairs from DAVIS dataset? 
    6- What does base C2V + descriptive prompt produce on the same grid?
        - Base model can not produce nicely! 
    7- Does training-time conditioning exactly match the inference time conditioning?
        - seems like it, but need to check! 

