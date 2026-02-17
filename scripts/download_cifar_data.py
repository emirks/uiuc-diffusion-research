#!/usr/bin/env python3
"""
scripts/download_data.py

Downloads small image datasets into the repo's data/ directory.

Current support:
- CIFAR-10 (via torchvision)

Directory convention:
- data/raw/<dataset_name>/            : raw downloaded dataset
- data/processed/<dataset_name>_png/  : optional exported PNG subset for inspection

Run from repo root:
  python scripts/download_data.py --dataset cifar10

Examples:
  # Download CIFAR-10 only
  python scripts/download_data.py --dataset cifar10

  # Download + export 200 images to PNG for quick look
  python scripts/download_data.py --dataset cifar10 --export_png --num_export 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from torchvision import datasets


def repo_root() -> Path:
    # scripts/download_data.py -> parents[1] is repo root
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_data_readme(data_dir: Path) -> None:
    readme = data_dir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "# Data Directory\n\n"
        "This directory is not meant to be committed to git.\n\n"
        "Conventions:\n"
        "- `raw/` holds downloaded datasets in their original format.\n"
        "- `processed/` holds derived artifacts (e.g., exported PNG subsets).\n\n"
        "If you need to recreate data, use scripts under `scripts/`.\n",
        encoding="utf-8",
    )


def export_cifar_png(ds, out_dir: Path, num_export: int) -> None:
    """
    Exports a subset of CIFAR-10 images as PNG for easy manual inspection.
    Uses the dataset in its PIL form (no normalization).
    """
    ensure_dir(out_dir)
    n = min(num_export, len(ds))
    for i in range(n):
        img, label = ds[i]  # img is PIL.Image, label is int
        assert isinstance(img, Image.Image)
        img.save(out_dir / f"img_{i:06d}_label_{label}.png")
    print(f"[saved] Exported {n} PNGs to: {out_dir}")


def download_cifar10(raw_root: Path) -> None:
    """
    Downloads CIFAR-10 train+test into raw_root/cifar10
    """
    cifar_dir = raw_root / "cifar10"
    ensure_dir(cifar_dir)

    # Download both splits so you're never blocked later
    datasets.CIFAR10(root=str(cifar_dir), train=True, download=True)
    datasets.CIFAR10(root=str(cifar_dir), train=False, download=True)

    print(f"[ok] CIFAR-10 downloaded to: {cifar_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    p.add_argument("--export_png", action="store_true", help="Export a PNG subset to data/processed for inspection.")
    p.add_argument("--num_export", type=int, default=200, help="How many images to export as PNG (if enabled).")
    args = p.parse_args()

    root = repo_root()
    data_dir = root / "data"
    raw_root = data_dir / "raw"
    processed_root = data_dir / "processed"

    ensure_dir(raw_root)
    ensure_dir(processed_root)
    write_data_readme(data_dir)

    if args.dataset == "cifar10":
        download_cifar10(raw_root)

        if args.export_png:
            # For export, load the dataset in PIL form (no transform)
            cifar_dir = raw_root / "cifar10"
            ds = datasets.CIFAR10(root=str(cifar_dir), train=True, download=False)
            out_dir = processed_root / "cifar10_png"
            export_cifar_png(ds, out_dir, args.num_export)

    print("[done]")


if __name__ == "__main__":
    main()
