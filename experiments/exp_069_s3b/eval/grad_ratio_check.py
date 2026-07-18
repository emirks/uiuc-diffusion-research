"""Stage-3b pre-launch gate: aux(margin):FM gradient-norm ratio (Round-3 Decision 2).

Rule: ‖∇_LoRA(λ_margin·L_margin)‖ / ‖∇_LoRA L_FM‖ at σ∈{0.1,0.3,0.6} must fall in
[0.05, 0.5]. If above 0.5 at low σ, halve λ_margin once and record. Below 0.05 = margin
inert (acceptable; the online activation band is the real-time guard).

Loads the base model + ic3 adapter (the 3b starting point), one real pair batch from
exp_064's precompute, and measures grads via two forwards on identical inputs (margin
off vs on) — the difference isolates the margin gradient. LoRA params only.
"""
import glob
import pathlib
import random

import torch

LAB = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
ADAPTER = LAB / "diffusion-research/outputs/training/exp_064_ic3_aligned_retrain/ic3/checkpoints/lora_weights_step_05000.safetensors"
PRE = LAB / "diffusion-research/experiments/exp_064_ic3_aligned_retrain/dataset/.precomputed"
LAMBDA_MARGIN = 0.3

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.training_strategies.transition import TransitionStrategy, TransitionStrategyConfig
from ltx_trainer.training_strategies.flexible import ModalityConfig, MaskConditionConfig, ReferenceConditionConfig

TARGET_MODULES = ["attn1.to_k","attn1.to_q","attn1.to_v","attn1.to_out.0",
                  "attn2.to_k","attn2.to_q","attn2.to_v","attn2.to_out.0",
                  "ff.net.0.proj","ff.net.2"]


class FixedSampler:
    def __init__(self, s): self.s = s
    def sample_for(self, latents):
        return torch.full((latents.shape[0],), self.s, dtype=latents.dtype, device=latents.device)


def video_cfg(**kw):
    return TransitionStrategyConfig(
        name="transition",
        video=ModalityConfig(is_generated=True, latents_dir="latents", conditions=[
            MaskConditionConfig(type="mask", mask_dir="masks", probability=1.0),
            ReferenceConditionConfig(type="reference", latents_dir="reference_latents", probability=1.0)]),
        residual_reference=True, gamma=1.0922, **kw)


def load_one_batch():
    # find a two-sided pair (mask has 3 conditioned latent frames)
    lat_files = glob.glob(str(PRE / "latents/**/*.pt"), recursive=True)
    random.seed(0); random.shuffle(lat_files)
    for lf in lat_files:
        rel = pathlib.Path(lf).relative_to(PRE / "latents")
        mk = PRE / "masks" / rel; rf = PRE / "reference_latents" / rel; cf = PRE / "conditions" / rel
        if not (mk.exists() and rf.exists() and cf.exists()):
            continue
        lat = torch.load(lf, weights_only=True); ref = torch.load(rf, weights_only=True)
        msk = torch.load(mk, weights_only=True); con = torch.load(cf, weights_only=True)
        b = lambda d: {k: (v.unsqueeze(0) if torch.is_tensor(v) else torch.tensor([v])) for k, v in d.items()}
        batch = {
            "video_latents": b(lat), "reference_latents": b(ref),
            "masks": {"mask": msk["mask"].unsqueeze(0)},
            "conditions": {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in con.items()},
        }
        return batch
    raise RuntimeError("no complete pair found")


@torch.enable_grad()
def grads_for(strategy, model, batch, sigma):
    inputs = strategy.prepare_training_inputs({k: dict(v) if isinstance(v, dict) else v for k, v in batch.items()},
                                              FixedSampler(sigma))
    vp, _ = model(video=inputs.video, audio=None, perturbations=None)
    loss = strategy.compute_loss(vp, None, inputs)
    model.zero_grad(set_to_none=True)
    loss.mean().backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None and "lora" in n.lower()}


def main():
    device = torch.device("cuda")
    model = load_transformer(str(MODEL), device="cpu", dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, target_modules=TARGET_MODULES,
                                             lora_dropout=0.0, init_lora_weights=True))
    sd = load_file(str(ADAPTER)); sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
    set_peft_model_state_dict(model.get_base_model(), sd)
    model = model.to(device)
    for _, p in model.named_parameters():
        p.requires_grad_("lora" in _.lower())

    batch = load_one_batch()
    batch = {k: ({kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                 if isinstance(v, dict) else v) for k, v in batch.items()}

    s_fm = TransitionStrategy(video_cfg(lambda_par=0.25, margin_enabled=False))
    s_full = TransitionStrategy(video_cfg(lambda_par=0.25, margin_enabled=True,
                                          lambda_margin=LAMBDA_MARGIN, margin_start_step=0))
    print(f"aux(margin):FM grad-norm ratio  (rule: [0.05, 0.5])   λ_margin={LAMBDA_MARGIN}")
    worst_low = 0.0
    for sig in (0.1, 0.3, 0.6):
        g_fm = grads_for(s_fm, model, batch, sig)
        g_full = grads_for(s_full, model, batch, sig)
        num = 0.0; den = 0.0
        for n in g_fm:
            gm = (g_full[n].float() - g_fm[n].float())
            num += gm.pow(2).sum().item(); den += g_fm[n].float().pow(2).sum().item()
        ratio = (num ** 0.5) / (den ** 0.5 + 1e-12)
        flag = "OK" if 0.05 <= ratio <= 0.5 else ("HIGH" if ratio > 0.5 else "low(inert)")
        print(f"  σ={sig}:  ratio={ratio:.4f}  [{flag}]")
        if sig <= 0.3:
            worst_low = max(worst_low, ratio)
    print(f"\nverdict: worst low-σ ratio={worst_low:.4f} -> "
          + ("HALVE λ_margin to 0.15 and record" if worst_low > 0.5 else "keep λ_margin=0.3"))


if __name__ == "__main__":
    main()
