"""exp_079 SupCon-T — dataset, class-balanced sampler, augmentation, and the SupCon loss.

Composite-label supervised contrastive learning over frozen LTX-VAE latents:

  * A sample is one (clip, manipulation) pair -> latent [128,16,20,15].
  * Its label under arm E1 is the COMPOSITE (class, manipulation); under the `ablation` arm
    (the owner's exact proposal) it is (class,) alone.
  * Positives = same label, different sample. Under E1 a clip's own reverse/warp therefore lands
    in a DIFFERENT class from itself -> it is a NEGATIVE with byte-identical content. That pair is
    the only content-identical / operator-different signal we have pre-dataset.

Augmentation is spatial/appearance only (nuisance invariance) and is applied IN LATENT space:
horizontal flip of the latent grid, plus small per-channel gain/offset jitter. Temporal structure
is never augmented — it is the signal.
"""

import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]


class ManipLatentDataset(Dataset):
    """(clip, manipulation) -> latent [C,F,H,W], with composite or class-only labels."""

    def __init__(self, split_json: Path, latent_root: Path, split_names: list[str],
                 manips: list[str], label_mode: str = "class_manip", augment: bool = False):
        if label_mode not in ("class_manip", "class"):
            raise ValueError(f"label_mode must be class_manip|class, got {label_mode!r}")
        self.label_mode = label_mode
        self.augment = augment
        split = json.loads(Path(split_json).read_text())

        self.items: list[dict] = []
        for sp in split_names:
            for c in split[sp]:
                for m in manips:
                    p = Path(latent_root) / sp / c["cls"] / f"{c['clip']}__{m}.pt"
                    if p.exists():
                        self.items.append({"path": p, "clip": c["clip"], "cls": c["cls"], "manip": m})
        if not self.items:
            raise FileNotFoundError(f"no latents under {latent_root} for splits={split_names}")

        keys = sorted({self._label_key(i) for i in self.items})
        self.label_to_idx = {k: i for i, k in enumerate(keys)}
        for it in self.items:
            it["label"] = self.label_to_idx[self._label_key(it)]

        # index by label for the class-balanced sampler
        self.by_label: dict[int, list[int]] = {}
        for i, it in enumerate(self.items):
            self.by_label.setdefault(it["label"], []).append(i)

    def _label_key(self, it: dict) -> str:
        return f"{it['cls']}|{it['manip']}" if self.label_mode == "class_manip" else it["cls"]

    @property
    def num_labels(self) -> int:
        return len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, lat: Tensor) -> Tensor:
        # spatial/appearance nuisance aug only — NEVER temporal (that is the signal).
        if random.random() < 0.5:
            lat = torch.flip(lat, dims=[-1])                    # horizontal flip of the latent grid
        gain = 1.0 + 0.05 * torch.randn(lat.shape[0], 1, 1, 1, dtype=lat.dtype)
        offset = 0.02 * torch.randn(lat.shape[0], 1, 1, 1, dtype=lat.dtype)
        return lat * gain + offset

    def __getitem__(self, i: int):
        it = self.items[i]
        lat = torch.load(it["path"], map_location="cpu", weights_only=False)["latents"].float()
        if self.augment:
            lat = self._augment(lat)
        return lat, it["label"], i


class ClassBalancedBatchSampler(torch.utils.data.Sampler):
    """P labels x K samples per batch — guarantees positives exist for every anchor."""

    def __init__(self, dataset: ManipLatentDataset, classes_per_batch: int,
                 samples_per_class: int, steps: int, seed: int = 0):
        self.by_label = {k: v for k, v in dataset.by_label.items() if len(v) >= 2}
        if not self.by_label:
            raise ValueError("no label has >=2 samples; SupCon needs positives")
        self.p = min(classes_per_batch, len(self.by_label))
        self.k = samples_per_class
        self.steps = steps
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        labels = list(self.by_label)
        for _ in range(self.steps):
            batch: list[int] = []
            for lab in self.rng.sample(labels, self.p):
                pool = self.by_label[lab]
                batch += (self.rng.sample(pool, self.k) if len(pool) >= self.k
                          else [self.rng.choice(pool) for _ in range(self.k)])
            yield batch


def supcon_loss(z: Tensor, labels: Tensor, temperature: float = 0.1) -> Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), L_out form. z is L2-normalized [B,D]."""
    device = z.device
    sim = z @ z.T / temperature
    # numerical stability: subtract per-row max (detached)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask

    exp_sim = sim.exp().masked_fill(self_mask, 0.0)
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12).log()

    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return z.sum() * 0.0
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


class ProjectionHead(nn.Module):
    """mean-pool the K operator tokens -> 2-layer MLP -> L2-normalized embedding."""

    def __init__(self, in_dim: int = 128, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens [B,C,F,H,W] (tiny latent video) -> z [B,out_dim], L2-normalized."""
        pooled = tokens.flatten(2).mean(dim=2)          # [B, C]
        return torch.nn.functional.normalize(self.net(pooled), dim=-1)


if __name__ == "__main__":
    # self-checks that need no data on disk
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    lab = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    l_rand = supcon_loss(z, lab).item()
    # perfectly clustered embeddings must score far lower than random ones
    zc = torch.nn.functional.normalize(
        torch.stack([torch.eye(4)[i // 2] for i in range(8)]) + 1e-3 * torch.randn(8, 4), dim=-1)
    l_clust = supcon_loss(zc, lab).item()
    assert l_clust < l_rand, f"clustered {l_clust} should beat random {l_rand}"
    # no-positive case is finite
    assert torch.isfinite(supcon_loss(z, torch.arange(8)))
    p = ProjectionHead()
    out = p(torch.randn(4, 128, 6, 4, 3))
    assert out.shape == (4, 128) and torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)
    print(f"[supcon_data] self-checks PASS (loss random={l_rand:.3f} clustered={l_clust:.3f}, "
          f"proj {tuple(out.shape)} unit-norm)")
