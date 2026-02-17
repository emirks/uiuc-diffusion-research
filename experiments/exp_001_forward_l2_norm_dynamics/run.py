#!/usr/bin/env python3
"""
experiments/exp_001_forward_l2_norm_dynamics/run.py

Forward diffusion ℓ2-norm dynamics inspection using diffusers (DDPMScheduler).

Question:
Does forward diffusion move samples toward the origin, or toward a Gaussian shell?

Outputs (repo-root relative):
- outputs/logs/exp_001_forward_l2_norm_dynamics.pt          (E[||x_t||_2] over t)
- outputs/logs/exp_001_forward_l2_norm_dynamics_sq.pt       (E[||x_t||_2^2] over t)
- outputs/figures/exp_001_forward_l2_norm_dynamics.png      (plot, if enabled)
- outputs/images/exp_001_forward_l2_norm_dynamics/t_XXXX/*.png  (optional)

Run from repo root:
  python experiments/exp_001_forward_l2_norm_dynamics/run.py

Dependencies:
  pip install torch torchvision diffusers pyyaml pillow matplotlib
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any, List

import torch
from diffusers import DDPMScheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e


# -------------------------
# Utilities (kept local to experiment; does NOT use src/)
# -------------------------

def repo_root() -> Path:
    # This file: diffusion-research/experiments/exp_001_forward_l2_norm_dynamics/run.py
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping/dict, got {type(cfg)} from: {path}")
    return cfg


def pick_device(device_cfg: str) -> torch.device:
    device_cfg = device_cfg.lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cifar10_is_present(root: Path) -> bool:
    # torchvision CIFAR10 expects a "cifar-10-batches-py" directory after extraction
    return (root / "cifar-10-batches-py").exists()


def to_uint8_pil(x: torch.Tensor) -> Image.Image:
    """
    x: (C,H,W) in [-1,1] -> uint8 PIL
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0  # [0,1]
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(x)


def format_t_dir(t: int) -> str:
    return f"t_{t:04d}"


def take_n_images(dl: DataLoader, n: int, device: torch.device) -> torch.Tensor:
    """
    Collect exactly n images (x0) from dataloader, normalized in [-1,1].
    """
    xs = []
    total = 0
    for batch in dl:
        x0 = batch[0]  # (img, label)
        if total + x0.size(0) > n:
            x0 = x0[: (n - total)]
        xs.append(x0)
        total += x0.size(0)
        if total >= n:
            break
    x = torch.cat(xs, dim=0).to(device)
    return x


# -------------------------
# Main
# -------------------------

def main() -> None:
    root = repo_root()
    cfg_path = Path(__file__).with_name("config.yaml").resolve()
    print(f"[info] loading config: {cfg_path}")
    cfg = load_config(cfg_path)

    required_top = ["experiment_name", "dataset", "diffusion", "logging", "save_images"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise KeyError(
            "Config is missing required keys: "
            + ", ".join(missing)
            + f". Loaded top-level keys: {sorted(cfg.keys())}. Config path: {cfg_path}"
        )

    torch.manual_seed(int(cfg["logging"]["seed"]))

    device = pick_device(cfg["logging"]["device"])
    print(f"[info] device: {device}")

    # Outputs
    out_logs = root / "outputs" / "logs"
    out_figs = root / "outputs" / "figures"
    out_imgs = root / "outputs" / "images" / cfg["experiment_name"]

    ensure_dir(out_logs)
    ensure_dir(out_figs)
    if bool(cfg["save_images"]["enabled"]):
        ensure_dir(out_imgs)

    # Dataset
    data_root = (root / cfg["dataset"]["data_root"]).resolve()
    ensure_dir(data_root)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),                 # [0,1]
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # [-1,1]
        ]
    )

    if cfg["dataset"]["name"].lower() != "cifar10":
        raise ValueError("This experiment currently supports dataset.name: cifar10")

    train = (cfg["dataset"]["split"].lower() == "train")
    download = not cifar10_is_present(data_root)
    if download:
        print(f"[info] CIFAR-10 not found under {data_root}; downloading...")
    ds = datasets.CIFAR10(root=str(data_root), train=train, download=download, transform=tfm)

    dl = DataLoader(
        ds,
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["dataset"]["num_workers"]),
        drop_last=False,
    )

    # Forward process scheduler (diffusers)
    T = int(cfg["diffusion"]["timesteps"])
    scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        clip_sample=False,
    )

    # Data subset
    num_images = int(cfg["dataset"]["num_images"])
    x0_all = take_n_images(dl, num_images, device=device)
    bsz = x0_all.size(0)
    d = x0_all[0].numel()

    print(f"[info] using num_images={bsz}, dimensionality d={d}, sqrt(d)={math.sqrt(d):.3f}")

    mean_norm = torch.zeros(T, device="cpu")
    mean_norm_sq = torch.zeros(T, device="cpu")

    per_image_cfg = cfg.get("per_image_norms", {})
    per_image_enabled = bool(per_image_cfg.get("enabled", True))
    per_image_count = int(per_image_cfg.get("num_images", 9))
    per_image_count = max(0, min(per_image_count, bsz))
    per_image_norms = torch.zeros((T, per_image_count), device="cpu") if (per_image_enabled and per_image_count > 0) else None

    # Iterate timesteps; for each t sample fresh noise and compute stats.
    with torch.no_grad():
        for t in range(T):
            t_tensor = torch.full((bsz,), t, device=device, dtype=torch.long)
            eps = torch.randn_like(x0_all)
            xt = scheduler.add_noise(original_samples=x0_all, noise=eps, timesteps=t_tensor)

            flat_xt = xt.flatten(1)
            norms = torch.linalg.vector_norm(flat_xt, ord=2, dim=1)  # ||x_t||
            norms_sq = (flat_xt * flat_xt).sum(dim=1)                # ||x_t||^2

            mean_norm[t] = norms.mean().detach().cpu()
            mean_norm_sq[t] = norms_sq.mean().detach().cpu()

            if per_image_norms is not None:
                per_image_norms[t] = norms[:per_image_count].detach().cpu()

            if t % 100 == 0 or t == T - 1:
                print(f"[t={t:4d}] E||x_t||={mean_norm[t].item():.3f}, E||x_t||^2={mean_norm_sq[t].item():.3f}")

    # Save logs
    norms_path = out_logs / "exp_001_forward_l2_norm_dynamics.pt"
    norms_sq_path = out_logs / "exp_001_forward_l2_norm_dynamics_sq.pt"
    torch.save(mean_norm, norms_path)
    torch.save(mean_norm_sq, norms_sq_path)
    print(f"[saved] {norms_path}")
    print(f"[saved] {norms_sq_path}")

    if per_image_norms is not None:
        per_image_path = out_logs / "exp_001_forward_l2_norm_dynamics_per_image.pt"
        torch.save(per_image_norms, per_image_path)
        print(f"[saved] {per_image_path}")

    # Optional plotting
    if bool(cfg["logging"]["save_plot"]):
        try:
            import matplotlib.pyplot as plt

            fig_path = out_figs / "exp_001_forward_l2_norm_dynamics.png"

            xs = torch.arange(T).numpy()
            plt.figure()
            plt.plot(xs, mean_norm.numpy(), label="E[||x_t||]")
            plt.axhline(y=math.sqrt(d), linestyle="--", label="sqrt(d)")
            plt.xlabel("t")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[saved] {fig_path}")

            if per_image_norms is not None:
                per_image_dir = out_figs / "exp_001_forward_l2_norm_dynamics_per_image"
                ensure_dir(per_image_dir)

                grid_path = out_figs / "exp_001_forward_l2_norm_dynamics_per_image_grid.png"
                cols = 3
                rows = int(math.ceil(per_image_count / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
                for i in range(rows * cols):
                    ax = axes[i // cols][i % cols]
                    if i < per_image_count:
                        ax.plot(xs, per_image_norms[:, i].numpy())
                        ax.set_title(f"image {i}")
                        ax.set_xlabel("t")
                        ax.set_ylabel("||x_t||")
                    else:
                        ax.axis("off")
                fig.tight_layout()
                fig.savefig(grid_path, dpi=200)
                plt.close(fig)
                print(f"[saved] {grid_path}")

                for i in range(per_image_count):
                    one_path = per_image_dir / f"image_{i:04d}_l2_norm.png"
                    plt.figure()
                    plt.plot(xs, per_image_norms[:, i].numpy())
                    plt.xlabel("t")
                    plt.ylabel("||x_t||")
                    plt.tight_layout()
                    plt.savefig(one_path, dpi=200)
                    plt.close()
        except ImportError:
            print("[warn] matplotlib not installed; skipping plot. Install with `pip install matplotlib`.")

    # Optional: save example images at selected timesteps
    if bool(cfg["save_images"]["enabled"]):
        ts: List[int] = list(cfg["save_images"]["timesteps"])
        ts = sorted({int(t) for t in ts if 0 <= int(t) < T})
        n_save = int(cfg["save_images"]["num_images"])
        n_save = min(n_save, bsz)

        x0_subset = x0_all[:n_save].detach()

        # Save x0 for reference
        x0_dir = out_imgs / "x0"
        ensure_dir(x0_dir)
        for i in range(n_save):
            to_uint8_pil(x0_subset[i]).save(x0_dir / f"img_{i:04d}.png")

        with torch.no_grad():
            for t in ts:
                t_dir = out_imgs / format_t_dir(t)
                ensure_dir(t_dir)
                t_tensor = torch.full((n_save,), t, device=device, dtype=torch.long)
                eps = torch.randn_like(x0_subset)
                xt = scheduler.add_noise(x0_subset, eps, t_tensor)
                for i in range(n_save):
                    to_uint8_pil(xt[i]).save(t_dir / f"img_{i:04d}.png")

        print(f"[saved] images under: {out_imgs}")

    # Interpretation hint
    print("\n[interpretation]")
    print("- Norm does NOT converge to 0; it converges to the Gaussian shell (radius ~ sqrt(d)).")
    print("- If E[||x_0||^2] > d, norms tend to decrease early; if < d, they increase.")
    print("- As t grows, E[||x_t||^2] approaches d (since ε ~ N(0,I) has E||ε||^2 = d).")


if __name__ == "__main__":
    main()
