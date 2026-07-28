"""CTT v2 — THE MIXED-FORMAT SMOKE GATE, per-format instrument (A9 §3 item 3).

A9 makes this gate mandatory and pre-registers its consequence: *"if per-format loss is not
finite and comparable, S4 AUTO-DROPS under the pre-registered fallback ladder"* (§4:
`S4 misses the cutoff => mix = 15 / 6 / 79`).

WHAT THIS SCRIPT IS NOT
-----------------------
It does not modify the trainer.  Every model, strategy, dataset, loss and sampler object is
the certified trainer's own class, imported and called; the forward+loss path is a
line-for-line replay of `trainer.py:_training_step` (:340-393).  What the script adds is
*accounting* the trainer does not do: it attributes every measurement to a named sample, so
per-format numbers exist at all (the trainer logs one scalar `train/loss` per step and, with
wandb off, only every 20th step).

The one injected object is `_FixedSigmaSampler`, used ONLY in the sigma-matched diagnostic
below.  It is passed as the `timestep_sampler` argument of the trainer's own public
`prepare_training_inputs(batch, timestep_sampler)`; it is defined in this file, never
installed anywhere, and is not used for the native-schedule measurement or for any training
run.  A9 forbids *patching the sampler to force a uniform shift* for the run — this is the
opposite: it holds sigma fixed so the two formats can be compared WITHOUT the shift confound
that A9 orders us to keep, disclose and caveat.

THE PRE-REGISTERED GATE (fixed before the job was submitted; see GATE_BARS below)
--------------------------------------------------------------------------------
    G1 index      the trainer's own `PrecomputedDataset` indexes N of N samples, no silent
                  drop  (REF_mixed_length "Fast index: N valid samples from N total")
    G2 finite     every per-sample loss, in every arm, at every sigma, is finite and > 0
    G3 shifts     the set of realized shifts, computed by the trainer's own
                  `_get_shift_for_sequence_length` from each sample's ACTUAL patchified
                  seq_len, is EXACTLY the two expected values and nothing else
    G4 comparable at MATCHED sigma, the per-format mean loss ratio lies in [1/3, 3]
    G5 native     under the native (unmatched) schedule every arm's mean loss is finite and
                  the arms stay within [1/10, 10] — a looser band because the shift
                  difference is real and expected here, so this bar only catches a
                  MECHANICAL blow-up, not the schedule effect
    G6 geometry   mask numel == F*H*W per sample; reference geometry == target geometry;
                  loss-mask fraction equals the mask's complement

    python scripts/ctt_v2/smoke/mixed_format_probe.py --root <smoke root> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
TRAINER = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer"
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"

#: pre-registered bars — edited only by a ruling, never by a result
GATE_BARS = {
    "G2_min_loss": 0.0,
    "G4_matched_sigma_ratio_band": [1.0 / 3.0, 3.0],
    "G5_native_ratio_band": [0.1, 10.0],
    "sigma_grid": [0.1, 0.25, 0.5, 0.75, 0.9],
    "native_draws_per_sample": 8,
    "seed": 42,
}


def log(m: str) -> None:
    print(f"[probe] {m}", flush=True)


# --------------------------------------------------------------------------------------
class _FixedSigmaSampler:
    """DIAGNOSTIC ONLY (see module docstring). Satisfies the `TimestepSampler` interface."""

    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, batch_size, seq_length=None, device=None):
        import torch

        return torch.full((batch_size,), self.value, device=device)

    def sample_for(self, batch):
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        return self.sample(batch.shape[0], device=batch.device)


# --------------------------------------------------------------------------------------
def build(root: Path, model_path: Path):
    """Instantiate the trainer's own dataset / strategy / model. Nothing bespoke."""
    import torch
    import yaml
    from ltx_core.text_encoders.gemma import convert_to_additive_mask
    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.datasets import PrecomputedDataset
    from ltx_trainer.model_loader import load_embeddings_processor, load_transformer
    from ltx_trainer.timestep_samplers import SAMPLERS
    from ltx_trainer.training_strategies import get_training_strategy

    ic = yaml.safe_load((LAB / "diffusion-research/eval_ladder/train/configs/ic_gen.yaml")
                        .read_text())
    # the certified dataset/conditioning block, verbatim; only the root is repointed
    cfg_d = {
        "model": {"model_path": str(model_path), "training_mode": "lora",
                  "text_encoder_path": ic["model"]["text_encoder_path"]},
        "lora": ic["lora"],
        "training_strategy": ic["training_strategy"],
        "optimization": dict(ic["optimization"], steps=1),
        "acceleration": ic["acceleration"],
        "data": {"preprocessed_data_root": str(root), "num_dataloader_workers": 0},
        "validation": {"samples": [], "interval": None},
        "checkpoints": ic["checkpoints"],
        "flow_matching": ic["flow_matching"],
        "wandb": {"enabled": False},
        "seed": 42,
        "output_dir": str(root / "_probe_out"),
    }
    cfg = LtxTrainerConfig(**cfg_d)

    strategy = get_training_strategy(cfg.training_strategy)
    sources = cfg.training_strategy.get_data_sources()
    log(f"data_sources = {sources}")
    ds = PrecomputedDataset(str(root), data_sources=sources)
    log(f"PrecomputedDataset: {len(ds)} samples")

    sampler_cls = SAMPLERS[cfg.flow_matching.timestep_sampling_mode]
    native_sampler = sampler_cls(**cfg.flow_matching.timestep_sampling_params)
    log(f"native sampler = {sampler_cls.__name__}"
        f"({cfg.flow_matching.timestep_sampling_params})")

    dev = torch.device("cuda")
    log("loading transformer (19B) ...")
    t0 = time.time()
    tr = load_transformer(checkpoint_path=str(model_path), device="cpu",
                          dtype=torch.bfloat16).to(dtype=torch.bfloat16)
    tr.requires_grad_(False)
    tr = tr.to(dev).eval()
    log(f"transformer on {dev} in {time.time()-t0:.0f}s")
    emb = load_embeddings_processor(checkpoint_path=str(model_path), device=dev,
                                    dtype=torch.bfloat16)
    emb.feature_extractor = None
    return ds, strategy, tr, emb, native_sampler, convert_to_additive_mask, dev


def collate1(sample: dict) -> dict:
    """`torch.utils.data.default_collate` for a single sample: add the batch dim."""
    import torch

    out = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.unsqueeze(0) if isinstance(vv, torch.Tensor)
                           else torch.tensor([vv])) for kk, vv in v.items()}
        elif isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
    return out


def one_forward(batch, strategy, tr, emb, conv, sampler, dev):
    """Line-for-line replay of `trainer.py:_training_step` (:340-393)."""
    import torch

    b = {}
    for k, v in batch.items():
        b[k] = {kk: (vv.to(dev) if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()} if isinstance(v, dict) else v
    conditions = b["conditions"]
    vf = conditions["video_prompt_embeds"]
    af = conditions.get("audio_prompt_embeds")
    mask = conditions["prompt_attention_mask"]
    additive = conv(mask, vf.dtype)
    ve, ae, am = emb.create_embeddings(vf, af, additive)
    conditions["video_prompt_embeds"] = ve
    conditions["audio_prompt_embeds"] = ae
    conditions["prompt_attention_mask"] = am

    mi = strategy.prepare_training_inputs(b, sampler)
    # The real trainer runs this forward under accelerate's native-AMP wrapper:
    # `Accelerator(mixed_precision="bf16").prepare(model)` replaces `model.forward` with
    # `convert_outputs_to_fp32(autocast(bf16)(forward))`.  Reproduced literally, because
    # without it the intrinsic-mask multiply (`flexible.py:542`, a float32 mask times bf16
    # clean latents) promotes `noisy_latents` to float32 and `F.linear` raises
    # "mat1 and mat2 must have the same dtype".  Autocast is what makes the certified run
    # work; the probe must not differ from it.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vp, ap = tr(video=mi.video, audio=mi.audio, perturbations=None)
    vp = vp.float() if vp is not None else None
    ap = ap.float() if ap is not None else None
    loss = strategy.compute_loss(vp, ap, mi)
    return {
        "loss": float(loss.detach().float().mean().item()),
        "sigma": float(mi.video.sigma.detach().float().mean().item()),
        "seq_len_total": int(mi.video.latent.shape[1]),
        "target_seq_len": int(mi.video_targets.shape[1]),
        "loss_mask_frac": float(mi.video_loss_mask.float().mean().item()),
    }


# --------------------------------------------------------------------------------------
def run(root: Path, out: Path, model_path: Path, quick: bool) -> int:
    import torch

    man = json.loads((root / "SMOKE_ROOT_MANIFEST.json").read_text())
    by_rel = {r["rel"]: r for r in man["samples"]}
    grid = GATE_BARS["sigma_grid"][:2] if quick else GATE_BARS["sigma_grid"]
    ndraw = 2 if quick else GATE_BARS["native_draws_per_sample"]

    ds, strategy, tr, emb, native, conv, dev = build(root, model_path)

    # ---- G1 index ---------------------------------------------------------------------
    n_expected = man["n_samples"]
    g1 = len(ds) == n_expected
    log(f"G1 index: dataset {len(ds)} of manifest {n_expected} -> {'PASS' if g1 else 'FAIL'}")

    # map dataset index -> manifest rel, through the dataset's own bookkeeping
    first_key = next(iter(ds.sample_files))
    idx_rel = [str(p) for p in ds.sample_files[first_key]]

    rows = []
    from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler as SLN

    for i in range(len(ds)):
        rel = idx_rel[i]
        meta = by_rel.get(rel)
        if meta is None:
            raise SystemExit(f"dataset sample {rel!r} is not in the smoke manifest")
        raw = ds[i]
        batch = collate1(raw)

        # geometry, straight from the loaded payload
        lat = raw["video_latents"]["latents"]
        f, h, w = (int(raw["video_latents"]["num_frames"]),
                   int(raw["video_latents"]["height"]), int(raw["video_latents"]["width"]))
        ref = raw["reference_latents"]
        mk = raw["masks"]["mask"]
        tok = f * h * w
        realized_shift = SLN._get_shift_for_sequence_length(tok)

        geo_bad = []
        if mk.numel() != tok:
            geo_bad.append(f"mask numel {mk.numel()} != {tok}")
        if (int(ref["num_frames"]), int(ref["height"]), int(ref["width"])) != (f, h, w):
            geo_bad.append("reference geometry != target geometry")
        if tuple(lat.shape[1:]) != (f, h, w):
            geo_bad.append(f"latent shape {tuple(lat.shape)} vs metadata ({f},{h},{w})")
        if abs(realized_shift - meta["expected_shift"]) > 1e-12:
            geo_bad.append(f"realized shift {realized_shift!r} != manifest "
                           f"{meta['expected_shift']!r}")

        # native schedule, ndraw independent draws
        torch.manual_seed(GATE_BARS["seed"] + i)
        nat = [one_forward(batch, strategy, tr, emb, conv, native, dev) for _ in range(ndraw)]

        # the reference is prepended AFTER the sigma draw and shares this sample's geometry,
        # so the combined sequence must be exactly twice the target and the loss mask must be
        # the mask's complement diluted by the (loss-free) reference half.
        cond_frac = float(mk.float().mean().item())
        if nat[0]["seq_len_total"] != 2 * tok:
            geo_bad.append(f"seq_len_with_reference {nat[0]['seq_len_total']} != 2*{tok}")
        if nat[0]["target_seq_len"] != tok:
            geo_bad.append(f"target_seq_len {nat[0]['target_seq_len']} != {tok}")
        want_lm = (1.0 - cond_frac) * tok / nat[0]["seq_len_total"]
        if abs(nat[0]["loss_mask_frac"] - want_lm) > 1e-6:
            geo_bad.append(f"loss_mask_frac {nat[0]['loss_mask_frac']:.6f} != "
                           f"(1-{cond_frac:.4f})*{tok}/{nat[0]['seq_len_total']} "
                           f"= {want_lm:.6f}")
        # sigma-matched diagnostic
        matched = {}
        for sv in grid:
            torch.manual_seed(GATE_BARS["seed"] + 10_000 + i)
            matched[f"{sv:.2f}"] = one_forward(batch, strategy, tr, emb, conv,
                                              _FixedSigmaSampler(sv), dev)

        row = {
            "rel": rel, "arm": meta["arm"], "format": meta["format"],
            "sided": meta["sided"], "latent_fhw": [f, h, w], "tokens": tok,
            "fps": float(raw["video_latents"]["fps"]),
            "realized_shift": realized_shift,
            "expected_shift": meta["expected_shift"],
            "seq_len_with_reference": nat[0]["seq_len_total"],
            "target_seq_len": nat[0]["target_seq_len"],
            "loss_mask_frac": nat[0]["loss_mask_frac"],
            "mask_cond_frac": float(mk.float().mean().item()),
            "native": nat,
            "native_mean_loss": sum(x["loss"] for x in nat) / len(nat),
            "matched": matched,
            "geometry_problems": geo_bad,
        }
        rows.append(row)
        log(f"  [{i+1}/{len(ds)}] {meta['arm']:14s} tok={tok:5d} shift={realized_shift:.6f} "
            f"seq(+ref)={nat[0]['seq_len_total']:5d} lossmask={nat[0]['loss_mask_frac']:.4f} "
            f"native<loss>={row['native_mean_loss']:.5f} "
            f"matched={[round(v['loss'],4) for v in matched.values()]}"
            + (f"  GEOM_BAD={geo_bad}" if geo_bad else ""))
        del raw, batch
        torch.cuda.empty_cache()

    # ---- aggregate --------------------------------------------------------------------
    fmts = sorted({r["format"] for r in rows})
    arms = sorted({r["arm"] for r in rows})

    def agg(keyfn, pick):
        o = {}
        for r in rows:
            o.setdefault(keyfn(r), []).append(pick(r))
        return {k: {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v)}
                for k, v in sorted(o.items())}

    per_format_native = agg(lambda r: r["format"], lambda r: r["native_mean_loss"])
    per_arm_native = agg(lambda r: r["arm"], lambda r: r["native_mean_loss"])
    per_format_matched = {}
    for sv in grid:
        k = f"{sv:.2f}"
        per_format_matched[k] = agg(lambda r: r["format"], lambda r, k=k: r["matched"][k]["loss"])

    all_losses = ([x["loss"] for r in rows for x in r["native"]]
                  + [v["loss"] for r in rows for v in r["matched"].values()])
    import math

    g2_bad = [x for x in all_losses if not math.isfinite(x) or x <= GATE_BARS["G2_min_loss"]]
    g2 = not g2_bad

    realized = sorted({r["realized_shift"] for r in rows})
    expected = sorted(set(man["distinct_expected_shifts"]))
    g3 = (len(realized) == len(expected)
          and all(abs(a - b) < 1e-12 for a, b in zip(realized, expected)))

    lo, hi = GATE_BARS["G4_matched_sigma_ratio_band"]
    g4_rows, g4 = [], True
    if len(fmts) >= 2:
        for k, d in per_format_matched.items():
            means = [d[f]["mean"] for f in fmts if f in d]
            ratio = max(means) / min(means)
            ok = lo <= ratio <= hi
            g4 &= ok
            g4_rows.append({"sigma": k, "means": {f: d[f]["mean"] for f in fmts if f in d},
                            "max_over_min": ratio, "ok": ok})

    nlo, nhi = GATE_BARS["G5_native_ratio_band"]
    nmeans = [v["mean"] for v in per_format_native.values()]
    native_ratio = max(nmeans) / min(nmeans) if len(nmeans) > 1 else 1.0
    g5 = nlo <= native_ratio <= nhi and all(math.isfinite(m) for m in nmeans)

    geo_bad = [f"{r['rel']}: {r['geometry_problems']}" for r in rows if r["geometry_problems"]]
    g6 = not geo_bad

    gates = {
        "G1_fast_index_N_of_N": {"ok": g1, "dataset_n": len(ds), "expected_n": n_expected},
        "G2_all_losses_finite_positive": {"ok": g2, "n_measurements": len(all_losses),
                                          "offenders": g2_bad[:10]},
        "G3_realized_shifts_exact": {"ok": g3, "realized": realized, "expected": expected},
        "G4_matched_sigma_comparable": {"ok": g4, "band": [lo, hi], "rows": g4_rows},
        "G5_native_loss_finite_comparable": {"ok": g5, "band": [nlo, nhi],
                                            "max_over_min": native_ratio,
                                            "per_format": per_format_native},
        "G6_geometry_consistent": {"ok": g6, "offenders": geo_bad},
    }
    verdict = "PASS" if all(v["ok"] for v in gates.values()) else "FAIL"

    rec = {
        "schema": "ctt_v2_mixed_format_smoke_gate/1",
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "authority": "A9 §3 item 3 — mandatory mixed-format smoke gate; AUTO-DROP consequence "
                     "pre-registered in A9 §4 (S4 misses cutoff => mix 15/6/79)",
        "root": str(root), "n_samples": len(rows),
        "captions": "PLACEHOLDER — one shared Gemma embedding for every sample "
                    "(Gemini credit-blocked; A9 permits placeholders for this gate)",
        "gate_bars": GATE_BARS,
        "formats": fmts, "arms": arms,
        "per_format_native": per_format_native,
        "per_arm_native": per_arm_native,
        "per_format_matched_sigma": per_format_matched,
        "gates": gates,
        "VERDICT": verdict,
        "samples": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "SMOKE_GATE.json").write_text(json.dumps(rec, indent=1) + "\n")

    log("")
    log("=" * 78)
    log("PER-FORMAT LOSS, native schedule (each format at ITS OWN shift — the confound A9")
    log("orders us to keep, disclose and caveat)")
    for f_, d in per_format_native.items():
        log(f"   {f_:6s} n={d['n']}  mean={d['mean']:.6f}  min={d['min']:.6f}  max={d['max']:.6f}")
    log(f"   max/min across formats = {native_ratio:.3f}  (bar {nlo}-{nhi})")
    log("PER-ARM LOSS, native schedule")
    for a_, d in per_arm_native.items():
        log(f"   {a_:14s} n={d['n']}  mean={d['mean']:.6f}")
    log("")
    log("PER-FORMAT LOSS at MATCHED sigma (shift confound removed — the comparability test)")
    for k, d in per_format_matched.items():
        s = "  ".join(f"{f_}={d[f_]['mean']:.6f}" for f_ in fmts if f_ in d)
        r = [x for x in g4_rows if x["sigma"] == k]
        log(f"   sigma={k}  {s}   max/min={r[0]['max_over_min']:.3f}" if r else f"   sigma={k}  {s}")
    log("")
    for k, v in gates.items():
        log(f"   [{'PASS' if v['ok'] else 'FAIL'}] {k}")
    log(f"VERDICT: {verdict}")
    log(f"-> {out/'SMOKE_GATE.json'}")
    log("=" * 78)
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(LAB / "diffusion-research/outputs/ctt_v2/smoke/root_mixed"))
    ap.add_argument("--out", default=str(LAB / "misc/ctt_v2_final/artefacts/smoke_gate"))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--quick", action="store_true", help="2 sigmas, 2 native draws")
    args = ap.parse_args()
    try:
        return run(Path(args.root), Path(args.out), Path(args.model), args.quick)
    except Exception:
        traceback.print_exc()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "SMOKE_GATE_CRASH.txt").write_text(traceback.format_exc())
        return 2


if __name__ == "__main__":
    sys.exit(main())
