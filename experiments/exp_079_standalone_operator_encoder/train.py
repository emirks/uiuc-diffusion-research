"""exp_079 SupCon-T — train the standalone operator encoder (no generator in the loop).

One arm x one seed per invocation:

    ARM=E1 SEED=42 python experiments/exp_079_standalone_operator_encoder/train.py
    ARM=ablation SEED=42 ...      # the owner's exact plain class-SupCon proposal

E1 uses composite (class, manipulation) labels, so a clip's own time-reverse/warp is a NEGATIVE
with byte-identical content. The `ablation` arm uses class-only labels (the owner's proposal) and
is expected to score ~0 on the temporal probes — that contrast is the point.

Writes outputs/exp_079/<arm>_seed<k>/{encoder.pt, train_log.jsonl, config_snapshot.yaml}.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
sys.path.insert(0, str(LAB / "LTX-2-bneck/packages/ltx-trainer/src"))

from supcon_data import (  # noqa: E402
    ClassBalancedBatchSampler, ManipLatentDataset, ProjectionHead, supcon_loss,
)


def build_encoder(cfg_enc: dict):
    """OperatorTokenEncoder with NORMAL init (the class zero-inits output_proj for the
    generator-coupled campaign; standalone we need a live code from step 0)."""
    from ltx_trainer.operator_encoder import OperatorTokenEncoder  # noqa: PLC0415

    enc = OperatorTokenEncoder(
        token_shape=tuple(cfg_enc["token_shape"]),
        latent_channels=cfg_enc["latent_channels"],
        width=cfg_enc["width"],
        depth=cfg_enc["depth"],
        num_heads=cfg_enc["num_heads"],
        prefix_latent_frames=cfg_enc["prefix_latent_frames"],
        suffix_latent_frames=cfg_enc["suffix_latent_frames"],
        skip_scale=cfg_enc.get("skip_scale", 0.0),
    )
    if not cfg_enc.get("zero_init_output", False):
        torch.nn.init.xavier_uniform_(enc.output_proj.weight)
        torch.nn.init.zeros_(enc.output_proj.bias)
    return enc


def main() -> None:
    arm = os.environ.get("ARM", "E1")
    seed = int(os.environ.get("SEED", "42"))
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    if arm not in cfg["arms"]:
        raise SystemExit(f"ARM must be one of {list(cfg['arms'])}, got {arm!r}")
    label_mode = cfg["arms"][arm]["label_mode"]

    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / f"{arm}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    device = cfg["runtime"]["device"]
    opt_cfg = cfg["optimization"]

    ds = ManipLatentDataset(
        split_json=REPO_ROOT / cfg["data"]["split"],
        latent_root=REPO_ROOT / cfg["data"]["manip_latents"],
        split_names=["train"],
        manips=cfg["data"]["train_manips"],
        label_mode=label_mode,
        augment=opt_cfg.get("augment", True),
    )
    sampler = ClassBalancedBatchSampler(
        ds, opt_cfg["sampler_classes_per_batch"], opt_cfg["sampler_samples_per_class"],
        steps=opt_cfg["steps"], seed=seed,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg["runtime"]["num_workers"],
                        pin_memory=True)
    print(f"[e079/{arm}/s{seed}] {len(ds)} samples, {ds.num_labels} labels "
          f"(label_mode={label_mode}), batch={sampler.p}x{sampler.k}, steps={opt_cfg['steps']}",
          flush=True)

    enc = build_encoder(cfg["encoder"]).to(device)
    head = ProjectionHead(cfg["encoder"]["latent_channels"], cfg["projection"]["hidden"],
                          cfg["projection"]["out_dim"]).to(device)
    n_par = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"[e079/{arm}/s{seed}] params: {n_par / 1e6:.2f}M", flush=True)

    optim = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                              lr=opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"])
    warm, total = opt_cfg["warmup_steps"], opt_cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + torch.cos(torch.tensor((s - warm) / max(total - warm, 1) * 3.14159265)).item()))

    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    t0 = time.time()
    enc.train(); head.train()
    for step, (lat, labels, _) in enumerate(loader):
        lat = lat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = enc(lat)
            z = head(tokens.float())
        loss = supcon_loss(z, labels, cfg["supcon"]["temperature"])

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
        optim.step(); sched.step()

        if step % 50 == 0 or step == total - 1:
            # cross-sample token sensitivity: the anti-collapse canary, live during training
            with torch.no_grad():
                t = tokens.float().flatten(1)
                d = torch.cdist(t, t)
                sens = (d.sum() / (d.numel() - d.shape[0])) / t.norm(dim=1).mean().clamp_min(1e-9)
            rec = {"step": step, "loss": round(float(loss), 4), "lr": round(sched.get_last_lr()[0], 6),
                   "token_sens": round(float(sens), 4), "sec": round(time.time() - t0, 1)}
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[e079/{arm}/s{seed}] {rec}", flush=True)

    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "arm": arm, "seed": seed, "label_mode": label_mode,
                "encoder_cfg": cfg["encoder"], "projection_cfg": cfg["projection"],
                "num_labels": ds.num_labels}, out_dir / "encoder.pt")
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    print(f"[e079/{arm}/s{seed}] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
