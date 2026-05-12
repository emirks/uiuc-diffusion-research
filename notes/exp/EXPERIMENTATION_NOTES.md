# Fully Manual Notes

### Exp 20, LTX2 C2V Diffusers
* Baseline implementation of C2V task (conditioned generation on 2 video endpoints) with diffusers library.

### Exp 21, Trajectory Logging on top of Exp20
* Baseline implementation of Trajectory Logging of Exp 20.
* VAE Latents, Velocity Fields, Transformer Hidden Features are hooked and saved.
* Analyzed within notebooks.

**TODO:** Observe and State the Results:

### Exp 23, VAE Latent LERP
* Question: Is the default behavior that LTX-2 collapses, the dissolve effect, when conditioned on semantically & contextually distant endpoints, is actually collapsing to a straight path at the VAE Latent Space between endpoints.
* Does LTX-2 being a Rectified Flow Matching model have an impact on that? Do that type of models have a tendency to collapse to the shortest-paths when the path with curvature does not have enough force on the manifold.   

* Results:
    - VAE Lerp between endpoints seem like the dissolve transitions produced by LTX-2.
    - Will check with different transition region lenghts, and state the result better.

### Exp 27, Flow Inversion
* Flow inversion is working. 
    Results:
        __main__  ──────────────────────────────────────────────────────────────────────
        __main__  RF inversion summary: 3/3 samples passed gate (LPIPS < 0.050)
        __main__    [PASS] class2__mallard-fly__mallard-water             LPIPS mean=0.0044  max=0.0154  steps=30
        __main__    [PASS] class5__car-roundabout__bus                    LPIPS mean=0.0020  max=0.0057  steps=30
        __main__    [PASS] class8__blackswan__boat                        LPIPS mean=0.0128  max=0.0513  steps=30
        __main__  ──────────────────────────────────────────────────────────────────────

