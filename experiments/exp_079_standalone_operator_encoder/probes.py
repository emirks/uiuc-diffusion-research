"""exp_079 SupCon-T — the pre-registered probe battery (bars frozen in config.yaml BEFORE training).

Five metrics, each labelled with what it may certify:

  1. NON-COLLAPSE (gate >=0.2)  cross-demo token sensitivity, B1's metric verbatim
                                mean pairwise ||E(di)-E(dj)|| / mean ||E(d)||.
                                Sanity only — SupCon's repulsion nearly guarantees it.
                                Reference: B1 0.0018, b1r residual 0.0075 (both dead).
  2. CLASS SEPARATION (report)  nearest-centroid accuracy on held-out INSTANCES.
                                Discriminativeness only. Explicitly NOT operator evidence.
  3. TEMPORAL GENERALIZATION (GATE, load-bearing, confound-valid) — on held-out ZS CLASSES:
                                margin(m) = d(z(V), z(m(V))) / median_V' d(z(V), z(V'))
                                over same-class other instances V'.
                                Bars: reverse margin >=1.0; held-out-gamma margin >=0.5;
                                gamma-monotonicity Spearman rho >=0.7 vs |log gamma|.
                                This is the ONE probe the class-confound cannot void: m(V) has
                                byte-identical content to V and differs only in the operator.
  4. CONTENT-LEAK GUARDS (report, one-sided) instance-ID decode within held-out classes;
                                endpoint-appearance ridge R^2 from z vs the raw pooled-latent
                                baseline (ratio). Low = content-light. Never claimed as
                                disentanglement (that needs the factorial dataset).
  5. CORPSE CONTROLS            the same battery on the B1 / b1r encoders — both MUST fail
                                metrics 1 and 3, which is what certifies the harness has
                                discriminative power against known-dead encoders.

    python experiments/exp_079_standalone_operator_encoder/probes.py --all
    python experiments/exp_079_standalone_operator_encoder/probes.py --arm E1 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(LAB / "LTX-2-bneck/packages/ltx-trainer/src"))

from supcon_data import ProjectionHead  # noqa: E402

MANIP_GAMMA = {"ease_in_g2": 2.0, "ease_out_g05": 0.5,
               "warp_g3": 3.0, "warp_g033": 1.0 / 3.0, "warp_g15": 1.5, "warp_g067": 2.0 / 3.0}


# ---------------------------------------------------------------- feature extraction

def load_arm(ckpt: Path, cfg: dict, device: str):
    """Load a trained SupCon-T encoder + head."""
    from ltx_trainer.operator_encoder import OperatorTokenEncoder  # noqa: PLC0415

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    ec = sd["encoder_cfg"]
    enc = OperatorTokenEncoder(
        token_shape=tuple(ec["token_shape"]), latent_channels=ec["latent_channels"],
        width=ec["width"], depth=ec["depth"], num_heads=ec["num_heads"],
        prefix_latent_frames=ec["prefix_latent_frames"],
        suffix_latent_frames=ec["suffix_latent_frames"], skip_scale=ec.get("skip_scale", 0.0))
    enc.load_state_dict(sd["encoder"]); enc.to(device).eval()
    pc = sd["projection_cfg"]
    head = ProjectionHead(ec["latent_channels"], pc["hidden"], pc["out_dim"])
    head.load_state_dict(sd["head"]); head.to(device).eval()
    return enc, head


def load_corpse(train_dir: Path, cfg: dict, device: str):
    """Load a B1/b1r encoder from its LoRA checkpoint (operator_encoder.* tensors) — metric 5."""
    from ltx_trainer.operator_encoder import OperatorTokenEncoder  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    cks = sorted((train_dir / "checkpoints").glob("*"), key=lambda p: p.stat().st_mtime)
    if not cks:
        return None, None
    tensors = {}
    for f in list(cks[-1].rglob("*.safetensors")) + list(cks[-1].glob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            if "operator_encoder" in k:
                tensors[k.split("operator_encoder.", 1)[-1]] = v
    if not tensors:
        return None, None
    tcfg = yaml.safe_load((train_dir / "training_config.yaml").read_text())
    bn = tcfg["training_strategy"]["video"]["conditions"][0].get("bottleneck", {})
    enc = OperatorTokenEncoder(
        token_shape=tuple(bn.get("token_shape", [6, 4, 3])),
        width=bn.get("width", 512), depth=bn.get("depth", 2), num_heads=bn.get("num_heads", 8),
        prefix_latent_frames=bn.get("prefix_latent_frames", 2),
        suffix_latent_frames=bn.get("suffix_latent_frames", 1),
        skip_scale=bn.get("skip_scale", 0.0))
    missing, unexpected = enc.load_state_dict(tensors, strict=False)
    enc.to(device).eval()
    return enc, {"ckpt": str(cks[-1]), "missing": len(missing), "unexpected": len(unexpected)}


@torch.no_grad()
def embed(enc, head, latent_root: Path, records: list[dict], device: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (z [N,D] projection embeddings or pooled tokens if head is None, tokens_flat [N,K*C])."""
    zs, toks = [], []
    for r in records:
        lat = torch.load(r["path"], map_location="cpu", weights_only=False)["latents"].float()
        t = enc(lat.unsqueeze(0).to(device))
        toks.append(t.float().flatten().cpu().numpy())
        zs.append((head(t.float()) if head is not None
                   else torch.nn.functional.normalize(t.float().flatten(2).mean(2), dim=-1)
                   ).squeeze(0).cpu().numpy())
    return np.stack(zs), np.stack(toks)


def collect(latent_root: Path, split: dict, split_name: str, manips: list[str]) -> list[dict]:
    out = []
    for c in split[split_name]:
        for m in manips:
            p = latent_root / split_name / c["cls"] / f"{c['clip']}__{m}.pt"
            if p.exists():
                out.append({"path": p, "clip": c["clip"], "cls": c["cls"], "manip": m})
    return out


# ---------------------------------------------------------------- metrics

def m1_sensitivity(tokens: np.ndarray) -> float:
    """Cross-demo token sensitivity — B1's metric verbatim."""
    d = np.linalg.norm(tokens[:, None, :] - tokens[None, :, :], axis=-1)
    n = tokens.shape[0]
    mean_pair = d.sum() / max(n * n - n, 1)
    return float(mean_pair / max(np.linalg.norm(tokens, axis=1).mean(), 1e-12))


def m2_class_separation(z: np.ndarray, cls: list[str]) -> float:
    """Leave-one-out nearest-centroid accuracy."""
    classes = sorted(set(cls))
    y = np.array([classes.index(c) for c in cls])
    correct = 0
    for i in range(len(y)):
        cents = []
        for k in range(len(classes)):
            mask = (y == k); mask[i] = False
            cents.append(z[mask].mean(0) if mask.any() else np.full(z.shape[1], np.inf))
        correct += int(np.argmin([np.linalg.norm(z[i] - c) for c in cents]) == y[i])
    return correct / max(len(y), 1)


def m3_temporal(z: np.ndarray, recs: list[dict], heldout_manips: list[str],
                train_manips: list[str], perm_manips: list[str] | None = None) -> dict:
    """Margin of each manipulation relative to same-class instance spread. LOAD-BEARING.

    GATES (content-controlled — the manipulation is a PERMUTATION of the frame multiset, so an
    order-invariant code is exactly invariant to it and any nonzero margin is order information):
      * reverse_margin              (trained manipulation)
      * heldout_permutation_margin  (median over probe-only permutations)
    DIAGNOSTICS (report-only — gamma warps resample with repetition and move the multiset, so a
    content-only code responds; measured rho=0.946 on the synthetic order-blind encoder):
      * heldout_gamma_margin, gamma_monotonicity_rho
    """
    perm_manips = perm_manips or []
    idx = {(r["clip"], r["manip"]): i for i, r in enumerate(recs)}
    by_cls: dict[str, list[str]] = {}
    for r in recs:
        by_cls.setdefault(r["cls"], []).append(r["clip"])
    for k in by_cls:
        by_cls[k] = sorted(set(by_cls[k]))

    per_manip: dict[str, list[float]] = {}
    mono_rhos: list[float] = []
    for cls, clips in by_cls.items():
        for v in clips:
            if (v, "identity") not in idx:
                continue
            zv = z[idx[(v, "identity")]]
            # denominator: median distance to OTHER same-class instances (identity manip)
            others = [z[idx[(o, "identity")]] for o in clips
                      if o != v and (o, "identity") in idx]
            if not others:
                continue
            denom = float(np.median([np.linalg.norm(zv - zo) for zo in others]))
            if denom <= 1e-9:
                continue
            gam_d = []
            for m in set(train_manips + heldout_manips):
                if m == "identity" or (v, m) not in idx:
                    continue
                d = float(np.linalg.norm(zv - z[idx[(v, m)]]))
                per_manip.setdefault(m, []).append(d / denom)
                if m in MANIP_GAMMA:
                    gam_d.append((abs(np.log(MANIP_GAMMA[m])), d))
            if len(gam_d) >= 3:
                from scipy.stats import spearmanr  # noqa: PLC0415
                # include the identity anchor (|log g| = 0, distance 0)
                xs = [0.0] + [g for g, _ in gam_d]
                ys = [0.0] + [d for _, d in gam_d]
                rho = spearmanr(xs, ys).statistic
                if np.isfinite(rho):
                    mono_rhos.append(float(rho))

    med = {m: float(np.median(v)) for m, v in per_manip.items()}
    gam = [med[m] for m in heldout_manips if m in med and m in MANIP_GAMMA]
    perm = [med[m] for m in perm_manips if m in med]
    return {
        "margin_per_manip": med,
        # --- GATES (content-controlled)
        "reverse_margin": med.get("reverse", float("nan")),
        "heldout_permutation_margin": float(np.median(perm)) if perm else float("nan"),
        "permutation_margins": {m: med[m] for m in perm_manips if m in med},
        # --- DIAGNOSTICS (report-only; NOT content-controlled)
        "heldout_gamma_margin_REPORT_ONLY": float(np.median(gam)) if gam else float("nan"),
        "gamma_monotonicity_rho_REPORT_ONLY": float(np.median(mono_rhos)) if mono_rhos else float("nan"),
        "gamma_caveat": "gamma warps change the frame multiset; an order-blind code responds "
                        "(synthetic CONTENT encoder scored rho=0.946) — diagnostics only",
        "n_clips_scored": len(mono_rhos),
    }


def _ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float = 1.0, folds: int = 5) -> float:
    """K-fold ridge R^2 (numpy closed form; no sklearn in this env)."""
    n = X.shape[0]
    if n < folds * 2:
        return float("nan")
    X = np.c_[X, np.ones(n)]
    order = np.arange(n)
    preds = np.zeros_like(Y)
    for f in range(folds):
        te = order[f::folds]; tr = np.setdiff1d(order, te)
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        W = np.linalg.solve(A, X[tr].T @ Y[tr])
        preds[te] = X[te] @ W
    ss_res = ((Y - preds) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def m4_leak(z: np.ndarray, tokens: np.ndarray, recs: list[dict], latent_root: Path) -> dict:
    """Instance-ID decode + endpoint-appearance R^2 from z vs a raw pooled-latent baseline."""
    # instance-ID within class (identity manip only), leave-one-out nearest centroid over clips
    ids = [i for i, r in enumerate(recs) if r["manip"] == "identity"]
    per_cls_acc, chance = [], []
    by_cls: dict[str, list[int]] = {}
    for i in ids:
        by_cls.setdefault(recs[i]["cls"], []).append(i)
    for cls, members in by_cls.items():
        clips = sorted({recs[i]["clip"] for i in members})
        if len(clips) < 2:
            continue
        # each clip has 1 identity sample -> use all manips for a within-class instance probe
        sub = [i for i, r in enumerate(recs) if r["cls"] == cls]
        y = np.array([clips.index(recs[i]["clip"]) for i in sub])
        zz = z[sub]
        correct = 0
        for j in range(len(sub)):
            cents = []
            for k in range(len(clips)):
                mask = (y == k); mask[j] = False
                cents.append(zz[mask].mean(0) if mask.any() else np.full(zz.shape[1], np.inf))
            correct += int(np.argmin([np.linalg.norm(zz[j] - c) for c in cents]) == y[j])
        per_cls_acc.append(correct / len(sub)); chance.append(1.0 / len(clips))

    # endpoint appearance = pooled first + last latent frames of the SOURCE latent
    Y, Zi, Base = [], [], []
    for i, r in enumerate(recs):
        lat = torch.load(r["path"], map_location="cpu", weights_only=False)["latents"].float()
        Y.append(torch.cat([lat[:, 0].mean((-2, -1)), lat[:, -1].mean((-2, -1))]).numpy())
        Zi.append(z[i]); Base.append(lat.mean((1, 2, 3)).numpy())
    Y = np.array(Y); Zi = np.array(Zi); Base = np.array(Base)
    # remove class means: within-class leakage is the question
    for arr in (Y, Zi, Base):
        for cls in {r["cls"] for r in recs}:
            m = np.array([r["cls"] == cls for r in recs])
            arr[m] -= arr[m].mean(0)
    r2_z, r2_base = _ridge_r2(Zi, Y), _ridge_r2(Base, Y)
    return {
        "instance_id_acc": float(np.mean(per_cls_acc)) if per_cls_acc else float("nan"),
        "instance_id_chance": float(np.mean(chance)) if chance else float("nan"),
        "endpoint_r2_from_z": r2_z,
        "endpoint_r2_baseline": r2_base,
        "endpoint_r2_ratio": float(r2_z / r2_base) if r2_base and np.isfinite(r2_base) and r2_base > 1e-6 else float("nan"),
    }


# ---------------------------------------------------------------- driver

def run_one(name: str, enc, head, cfg: dict, split: dict, latent_root: Path, device: str) -> dict:
    train_m, ho_m = cfg["data"]["train_manips"], cfg["data"]["heldout_manips"]
    # metric 3/4 live on the held-out ZS CLASSES (all 8 manips encoded there)
    zs_recs = collect(latent_root, split, "heldout_class", train_m + ho_m)
    hi_recs = collect(latent_root, split, "heldout_instance", ["identity"])
    if not zs_recs:
        return {"arm": name, "error": "no heldout_class latents — run encode_manipulations first"}

    z_zs, tok_zs = embed(enc, head, latent_root, zs_recs, device)
    out = {"arm": name, "n_zs_samples": len(zs_recs)}
    out["m1_sensitivity"] = m1_sensitivity(tok_zs)
    if hi_recs:
        z_hi, _ = embed(enc, head, latent_root, hi_recs, device)
        out["m2_class_sep_acc"] = m2_class_separation(z_hi, [r["cls"] for r in hi_recs])
        out["m2_chance"] = 1.0 / len({r["cls"] for r in hi_recs})
    out["m3"] = m3_temporal(z_zs, zs_recs, ho_m, train_m, cfg["data"].get("perm_manips", []))
    out["m4"] = m4_leak(z_zs, tok_zs, zs_recs, latent_root)

    p = cfg["probes"]
    m3 = out["m3"]
    # Only content-controlled (multiset-preserving) manipulations gate. Gamma is reported, never gating.
    out["bars"] = {
        "m1_noncollapse": bool(out["m1_sensitivity"] >= p["sensitivity_min"]),
        "m3_reverse_margin": bool(m3["reverse_margin"] >= p["temporal_reverse_margin_min"]),
        "m3_heldout_permutation": bool(
            m3["heldout_permutation_margin"] >= p["heldout_permutation_margin_min"]),
    }
    out["PASS"] = all(out["bars"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every trained arm/seed + corpse controls")
    ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--corpses", action="store_true", help="only the B1/b1r corpse controls")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    split = json.loads((REPO_ROOT / cfg["data"]["split"]).read_text())
    latent_root = REPO_ROOT / cfg["data"]["manip_latents"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = REPO_ROOT / cfg["outputs"]["dir"]
    results = []

    if not args.corpses:
        targets = []
        if args.arm:
            targets = [(args.arm, args.seed or cfg["optimization"]["seeds"][0])]
        elif args.all:
            targets = [(a, s) for a in cfg["arms"] for s in cfg["optimization"]["seeds"]]
        for arm, seed in targets:
            ck = out_root / f"{arm}_seed{seed}" / "encoder.pt"
            if not ck.exists():
                print(f"[probe] skip {arm}_seed{seed} (no checkpoint)"); continue
            enc, head = load_arm(ck, cfg, device)
            r = run_one(f"{arm}_seed{seed}", enc, head, cfg, split, latent_root, device)
            results.append(r); print(json.dumps(r, indent=1), flush=True)

    if args.all or args.corpses:
        for name, rel in cfg["corpse_encoders"].items():
            enc, info = load_corpse(REPO_ROOT / rel, cfg, device)
            if enc is None:
                print(f"[probe] corpse {name}: no operator_encoder tensors found"); continue
            r = run_one(f"corpse_{name}", enc, None, cfg, split, latent_root, device)
            r["corpse_info"] = info
            r["EXPECTED"] = "must FAIL m1 and m3 (harness calibration on a known-dead encoder)"
            results.append(r); print(json.dumps(r, indent=1), flush=True)

    if results:
        out_root.mkdir(parents=True, exist_ok=True)
        dst = out_root / "probe_table.json"
        prev = json.loads(dst.read_text()) if dst.exists() else []
        keep = [x for x in prev if x.get("arm") not in {r["arm"] for r in results}]
        dst.write_text(json.dumps(keep + results, indent=1))
        print(f"[probe] wrote {dst.relative_to(REPO_ROOT)} ({len(keep + results)} rows)")


if __name__ == "__main__":
    main()
