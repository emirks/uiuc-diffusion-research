"""exp_058 — CPU unit test: per-sample suffix masks == exp_056 prefix+suffix.

Verifies against the REAL FlexibleStrategy that
  (a) MaskCondition(mask = first 2 latent frames + last 1) reproduces
      Prefix(tb=2,p=1)+Suffix(tb=1,p=1) EXACTLY (latents/timesteps/loss_mask),
  (b) MaskCondition(mask = first 2 only) == Prefix(tb=2) alone (one-sided),
  (c) both compose with the reference condition (concat before target,
      loss sliced to target region only).
Run inside the trainer venv: cd $LAB/LTX-2-official && uv run python <this file>
"""
import sys, types, tempfile, pathlib
import torch

from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategy, FlexibleStrategyConfig, ModalityConfig,
    PrefixConditionConfig, SuffixConditionConfig, MaskConditionConfig,
    ReferenceConditionConfig,
)
from ltx_trainer.timestep_samplers import UniformTimestepSampler

F, H, W, C = 16, 15, 20, 128   # 121f @ 480x640 -> 16x15x20 latent grid
SEQ = F * H * W

def make_batch(tmp, mask_frames):
    torch.manual_seed(0)
    lat = torch.randn(1, C, F, H, W)
    ref = torch.randn(1, C, F, H, W)
    mask = torch.zeros(F, H, W)
    for f in mask_frames:
        mask[f] = 1.0
    return {
        "video_latents": {"latents": lat, "num_frames": torch.tensor([F]),
                    "height": torch.tensor([H]), "width": torch.tensor([W]),
                    "fps": torch.tensor([24.0])},
        "reference_latents": {"latents": ref, "num_frames": torch.tensor([F]),
                              "height": torch.tensor([H]), "width": torch.tensor([W]),
                              "fps": torch.tensor([24.0])},
        "masks": {"mask": mask.unsqueeze(0)},   # dataloader adds batch dim
        "conditions": {"video_prompt_embeds": torch.randn(1, 8, 4096),
                       "prompt_attention_mask": torch.ones(1, 8, dtype=torch.bool)},
    }

def build(conds, tmp):
    cfg = FlexibleStrategyConfig(
        name="flexible",
        video=ModalityConfig(is_generated=True, latents_dir=str(tmp), conditions=conds),
    )
    return FlexibleStrategy(cfg)

def run(strategy, batch, seed=1234):
    torch.manual_seed(seed)
    sampler = UniformTimestepSampler(min_value=0.4, max_value=0.4)  # deterministic sigma
    return strategy.prepare_training_inputs(batch, sampler)

with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    ref_cond = ReferenceConditionConfig(type="reference", latents_dir="reference_latents", probability=1.0)

    # --- (a) two-sided: mask [0,1,15] vs prefix(2)+suffix(1) ---
    s_ps = build([PrefixConditionConfig(type="prefix", temporal_boundary=2, probability=1.0),
                  SuffixConditionConfig(type="suffix", temporal_boundary=1, probability=1.0),
                  ref_cond], tmp)
    s_mask = build([MaskConditionConfig(type="mask", mask_dir="masks", probability=1.0),
                    ref_cond], tmp)

    b1 = make_batch(tmp, mask_frames=[0, 1, F - 1])
    out_ps = run(s_ps, make_batch(tmp, []))          # masks ignored by prefix/suffix path
    out_mask = run(s_mask, b1)

    assert torch.equal(out_ps.video.latent, out_mask.video.latent), "latents differ (two-sided)"
    assert torch.equal(out_ps.video.timesteps, out_mask.video.timesteps), "timesteps differ"
    assert torch.equal(out_ps.video_loss_mask, out_mask.video_loss_mask), "loss_mask differs"
    print("[a] PASS  mask[0,1,15] == prefix(2)+suffix(1), with reference concat")

    # --- (b) one-sided: mask [0,1] vs prefix(2) only ---
    s_p = build([PrefixConditionConfig(type="prefix", temporal_boundary=2, probability=1.0), ref_cond], tmp)
    out_p = run(s_p, make_batch(tmp, []))
    out_os = run(s_mask, make_batch(tmp, mask_frames=[0, 1]))
    assert torch.equal(out_p.video.latent, out_os.video.latent), "latents differ (one-sided)"
    assert torch.equal(out_p.video.timesteps, out_os.video.timesteps), "timesteps differ (one-sided)"
    assert torch.equal(out_p.video_loss_mask, out_os.video_loss_mask), "loss_mask differs (one-sided)"
    print("[b] PASS  mask[0,1] == prefix(2) only")

    # --- (c) structural checks on the mask path ---
    m = out_mask
    assert m.video.latent.shape[1] == 2 * SEQ, "reference not concatenated"
    tpf = H * W
    ts_target = m.video.timesteps[:, SEQ:]           # target region after ref concat
    assert (ts_target[:, :2 * tpf] == 0).all(), "start latent frames not clean"
    assert (ts_target[:, -tpf:] == 0).all(), "end latent frame not clean"
    assert (ts_target[:, 2 * tpf:-tpf] > 0).all(), "middle frames not noised"
    lm = m.video_loss_mask
    assert (~lm[:, :SEQ]).all(), "reference tokens leak into loss"
    lm_t = lm[:, SEQ:]
    assert (~lm_t[:, :2 * tpf]).all() and (~lm_t[:, -tpf:]).all() and lm_t[:, 2 * tpf:-tpf].all(), \
        "loss mask wrong on target"
    one = out_os.video_loss_mask[:, SEQ:]
    assert (~one[:, :2 * tpf]).all() and one[:, 2 * tpf:].all(), "one-sided loss mask wrong (suffix should be trainable)"
    print("[c] PASS  structure: ref loss-excluded, endpoints clean, middle trained; one-sided suffix trainable")

print("ALL PASS")
