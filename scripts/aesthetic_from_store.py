#!/usr/bin/env python
"""aesthetic_from_store — LAION aesthetic quality merged into the lens rows.

The LAION "improved aesthetic predictor" head is a plain MLP 768->1024->128->64->16->1
(``$LAB/cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth``; state-dict keys ``layers.{0,2,4,6,7}.*``,
the Dropouts at 1/3/5 carry no params). Its input is the L2-normalized CLIP ViT-L/14 image embedding
— EXACTLY the ``clip_l14@r224`` per-frame feature the store already holds (``feats [T,768]``,
L2-normalized). Per gen: score every frame through the head and take the mean -> ``aesthetic``.

Reads each arm's ``rows.jsonl`` in a lenses eval (written by ``scripts/lens_pass_gridv3.py --collect``)
and merges an ``aesthetic`` column in place (idempotent — reruns overwrite it). When ``clip_l14@r224``
is absent for a gen, ``aesthetic`` is NaN and a ``missing clip_l14@r224 (aesthetic)`` note is added to
the row's ``warnings``; the script never crashes. CPU torch (the head is tiny; features are precomputed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.feature_store import FeatureStore  # noqa: E402

LAB = Path(os.environ.get("LAB", "/taiga/illinois/eng/cs/jrehg/users/emirkisa"))
DEFAULT_HEAD = LAB / "cache" / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth"
CLIP_L14_NS = "clip_l14@r224"
# The canonical LAION head layer shapes (verified at load).
EXPECTED_SHAPES = {
    "layers.0.weight": (1024, 768), "layers.0.bias": (1024,),
    "layers.2.weight": (128, 1024), "layers.2.bias": (128,),
    "layers.4.weight": (64, 128), "layers.4.bias": (64,),
    "layers.6.weight": (16, 64), "layers.6.bias": (16,),
    "layers.7.weight": (1, 16), "layers.7.bias": (1,),
}


def build_head():
    """The LAION predictor MLP. The state dict prefixes every param with ``layers.``, so the
    Sequential lives under a ``.layers`` attribute (matching the original ``MLP`` module)."""
    import torch.nn as nn

    class AestheticMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),   # layers.0
                nn.Dropout(0.2),        # layers.1 (no params)
                nn.Linear(1024, 128),   # layers.2
                nn.Dropout(0.2),        # layers.3
                nn.Linear(128, 64),     # layers.4
                nn.Dropout(0.1),        # layers.5
                nn.Linear(64, 16),      # layers.6
                nn.Linear(16, 1),       # layers.7
            )

        def forward(self, x):
            return self.layers(x)

    return AestheticMLP()


def load_aesthetic_head(path: Path = DEFAULT_HEAD, device: str = "cpu"):
    """Load + verify the LAION head; returns an eval()-mode module on ``device``."""
    import torch
    sd = torch.load(str(path), map_location="cpu")
    for k, shp in EXPECTED_SHAPES.items():
        if k not in sd:
            raise ValueError(f"aesthetic head missing key {k}")
        got = tuple(sd[k].shape)
        if got != shp:
            raise ValueError(f"aesthetic head {k}: shape {got} != expected {shp}")
    head = build_head()
    head.load_state_dict(sd)
    head.eval().to(device)
    return head


def aesthetic_from_feats(feats: np.ndarray, head, device: str = "cpu") -> float:
    """Mean over frames of head(L2-normalized clip_l14 feats). feats: [T,768] (already normalized)."""
    import torch
    x = np.asarray(feats, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.size == 0:
        return float("nan")
    with torch.no_grad():
        t = torch.from_numpy(x).to(device).float()
        # re-normalize defensively (a no-op on already-normalized feats; matches the LAION path)
        t = torch.nn.functional.normalize(t, dim=-1)
        y = head(t).squeeze(-1)              # [T]
        return float(y.mean().item())


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.parent / f"{path.name}.tmp-{os.getpid()}"
    tmp.write_text(text)
    os.replace(tmp, path)


def merge_arm(fs: FeatureStore, head, rows_path: Path, device: str) -> dict:
    """Merge the ``aesthetic`` column into one arm's rows.jsonl (idempotent)."""
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
    n_ok = n_missing = 0
    for r in rows:
        gen = REPO_ROOT / r["gen"] if r.get("gen") else None
        warnings = [w for w in (r.get("warnings") or []) if "clip_l14@r224 (aesthetic)" not in w]
        if gen is not None and fs.has(gen, CLIP_L14_NS):
            feats = fs.get(gen, CLIP_L14_NS)["feats"]
            r["aesthetic"] = aesthetic_from_feats(feats, head, device)
            n_ok += 1
        else:
            r["aesthetic"] = float("nan")
            warnings = warnings + ["missing clip_l14@r224 (aesthetic)"]
            n_missing += 1
        r["warnings"] = warnings
    _atomic_write(rows_path, "".join(json.dumps(r) + "\n" for r in rows))
    return {"n": len(rows), "aesthetic_ok": n_ok, "aesthetic_missing": n_missing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="merge LAION aesthetic into a lenses eval's rows.jsonl")
    ap.add_argument("--eval-id", default="039_lenses_gridv3__dai__2026-09-18")
    ap.add_argument("--head", default=str(DEFAULT_HEAD))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--store-root", default=None)
    a = ap.parse_args(argv)

    eval_dir = REPO_ROOT / "store" / "evals" / a.eval_id
    if not eval_dir.is_dir():
        raise SystemExit(f"no eval dir {eval_dir} (run lens_pass_gridv3.py --collect first)")
    fs = FeatureStore(Path(a.store_root).resolve() if a.store_root else REPO_ROOT)
    head = load_aesthetic_head(Path(a.head), a.device)
    rows_files = sorted(eval_dir.glob("*/rows.jsonl"))
    if not rows_files:
        raise SystemExit(f"no <arm>/rows.jsonl under {eval_dir}")
    tot_ok = tot_missing = tot_n = 0
    for rp in rows_files:
        st = merge_arm(fs, head, rp, a.device)
        tot_ok += st["aesthetic_ok"]; tot_missing += st["aesthetic_missing"]; tot_n += st["n"]
        print(f"  {rp.parent.name:34s} n={st['n']:4d} aesthetic_ok={st['aesthetic_ok']:4d} "
              f"missing={st['aesthetic_missing']:4d}", flush=True)
    print(f"[aesthetic] arms={len(rows_files)} rows={tot_n} ok={tot_ok} missing={tot_missing} "
          f"head={a.head}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
