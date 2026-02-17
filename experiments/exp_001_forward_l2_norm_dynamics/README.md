# exp_001_forward_l2_norm_dynamics

## Question
In the DDPM forward process `x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε`, does the trajectory push samples toward the origin, or toward the high-dimensional Gaussian shell?

## Setup
- **Dataset**: CIFAR-10 (train)
- **Data location**: `data/raw/cifar10`
- **Preprocessing**: images normalized to `[-1, 1]`
- **Forward process**: `diffusers.DDPMScheduler.add_noise`
- **Timesteps**: `T = 1000` (linear β schedule)
- **Metric**:
  - `E[||x_t||_2]`
  - `E[||x_t||_2^2]`

For CIFAR-10 tensors shaped `(3, 32, 32)`, the dimensionality is:
- `d = 3 * 32 * 32 = 3072`
- `sqrt(d) ≈ 55.43`

## How to run
From repo root:

```bash
python experiments/exp_001_forward_l2_norm_dynamics/run.py
```

Config lives in:
- `experiments/exp_001_forward_l2_norm_dynamics/config.yaml`

## Outputs
- **Norm logs**:
  - `outputs/logs/exp_001_forward_l2_norm_dynamics.pt` (tensor of `E[||x_t||]` over `t`)
  - `outputs/logs/exp_001_forward_l2_norm_dynamics_sq.pt` (tensor of `E[||x_t||^2]` over `t`)
- **Plot** (if enabled):
  - `outputs/figures/exp_001_forward_l2_norm_dynamics.png`
- **Example images** (if enabled):
  - `outputs/images/exp_001_forward_l2_norm_dynamics/x0/*.png`
  - `outputs/images/exp_001_forward_l2_norm_dynamics/t_XXXX/*.png`

## Conclusion
Forward diffusion does **not** collapse toward the origin. Instead, in high dimensions it approaches an approximately isotropic Gaussian where the mass concentrates on a thin shell.

Empirically in this experiment:
- `E[||x_t||_2]` increases with `t` and approaches `≈ sqrt(d)`.
- `E[||x_t||_2^2]` approaches `≈ d`.

Intuition:
- As `t` grows, the `x_0` contribution decays and `x_t` becomes dominated by `ε ~ N(0, I)`.
- For `ε ~ N(0, I_d)`, `E[||ε||^2] = d`, and typical `||ε|| ≈ sqrt(d)`.

This explains why the forward process “moves toward a Gaussian shell” rather than toward zero.
