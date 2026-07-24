"""Synthetic control test for the exp_079 probe battery — runs on CPU, needs no real latents.

The advisor requires corpse controls (B1/b1r) to prove the battery can detect a dead encoder.
Those need the real manipulation latents. This test proves the same property immediately, using
three synthetic encoders whose behaviour is known by construction:

  DEAD      constant output, ignores its input      -> must FAIL m1 (non-collapse) and m3 (temporal)
  CONTENT   encodes only per-clip appearance, is    -> must PASS m1, FAIL m3 (this is the b1r failure
            blind to frame ORDER (mean over time)      mode and the plain-class-SupCon prediction)
  ORACLE    encodes appearance AND a direction/     -> must PASS m1 and m3
            timing signature read from frame order

If the battery cannot separate ORACLE from CONTENT, metric 3 is not measuring what it claims and the
bars are meaningless — better to learn that here than after six training runs.

    python experiments/exp_079_standalone_operator_encoder/test_probe_harness.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from manip_utils import GAMMA_MANIPS, HELDOUT_MANIPS, PERM_MANIPS, TRAIN_MANIPS, manipulate  # noqa: E402
from probes import m1_sensitivity, m2_class_separation, m3_temporal  # noqa: E402

N_CLASSES, N_CLIPS_PER_CLASS, F, H, W, C = 4, 3, 16, 5, 4, 8
ALL_MANIPS = TRAIN_MANIPS + HELDOUT_MANIPS


def synth_clip(rng: np.random.Generator, cls: int) -> torch.Tensor:
    """A fake 'video' [F,C,H,W]: a class-typical appearance SHARED by every clip of the class
    (seeded by cls, so within-class distance < between-class — the real corpus's structure), plus a
    clip-specific appearance, plus a monotone temporal ramp so frame ORDER carries real information."""
    cls_rng = np.random.default_rng(1000 + cls)               # shared across the class's clips
    cls_app = cls_rng.normal(size=(1, C, H, W)) * 2.0
    cls_motion = cls_rng.normal(size=(1, C, H, W)) * 1.5      # class-typical manner
    clip_app = rng.normal(size=(1, C, H, W)) * 0.5
    t = np.linspace(0, 1, F).reshape(F, 1, 1, 1)
    return torch.tensor(cls_app + clip_app + t * cls_motion, dtype=torch.float32)


def encode(kind: str, frames: torch.Tensor) -> np.ndarray:
    """Synthetic 'encoders' over a manipulated pixel clip [F,C,H,W] -> flat code."""
    x = frames.numpy()
    if kind == "DEAD":
        return np.ones(32)
    appearance = x.mean(axis=0).ravel()                       # order-invariant
    if kind == "CONTENT":
        return appearance
    # ORACLE: appearance + an order-sensitive signature (first-half minus second-half, and the
    # per-frame trajectory's signed curvature) — exactly the kind of thing a temporal code carries.
    half = x.shape[0] // 2
    direction = (x[half:].mean(axis=0) - x[:half].mean(axis=0)).ravel()
    per_frame = x.reshape(x.shape[0], -1).mean(axis=1)
    timing = np.diff(per_frame, n=1)
    return np.concatenate([appearance, direction * 3.0, timing * 3.0])


def build(kind: str, rng: np.random.Generator):
    recs, codes = [], []
    for c in range(N_CLASSES):
        for k in range(N_CLIPS_PER_CLASS):
            base = synth_clip(rng, c)
            for m in ALL_MANIPS:
                recs.append({"clip": f"c{c}_k{k}", "cls": f"cls{c}", "manip": m})
                codes.append(encode(kind, manipulate(base, m)))
    z = np.stack(codes)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z, recs


def main() -> None:
    print(f"{'encoder':9s} {'m1_sens':>9s} {'m2_class':>9s} | GATES {'m3_rev':>8s} {'m3_perm':>8s}"
          f" | REPORT-ONLY {'m3_hoγ':>8s} {'m3_ρ':>7s}")
    got = {}
    for kind in ("DEAD", "CONTENT", "ORACLE"):
        rng = np.random.default_rng(0)                        # same clips for every encoder
        z, recs = build(kind, rng)
        sens = m1_sensitivity(z)
        ident = [i for i, r in enumerate(recs) if r["manip"] == "identity"]
        csep = m2_class_separation(z[ident], [recs[i]["cls"] for i in ident])
        m3 = m3_temporal(z, recs, HELDOUT_MANIPS, TRAIN_MANIPS, PERM_MANIPS)
        got[kind] = (sens, m3)
        print(f"{kind:9s} {sens:9.4f} {csep:9.2f} |       {m3['reverse_margin']:8.3f} "
              f"{m3['heldout_permutation_margin']:8.3f} |             "
              f"{m3['heldout_gamma_margin_REPORT_ONLY']:8.3f} "
              f"{m3['gamma_monotonicity_rho_REPORT_ONLY']:7.3f}")

    # ---- the properties the battery MUST have for its bars to mean anything
    dead_s, dead_m3 = got["DEAD"]
    cont_s, cont_m3 = got["CONTENT"]
    orac_s, orac_m3 = got["ORACLE"]

    assert dead_s < 0.2, f"DEAD must fail non-collapse, got sensitivity {dead_s}"
    assert cont_s >= 0.2 and orac_s >= 0.2, "live encoders must pass non-collapse"
    assert cont_m3["reverse_margin"] < 1.0, \
        f"CONTENT (order-blind) must fail the reverse bar, got {cont_m3['reverse_margin']}"
    assert orac_m3["reverse_margin"] >= 1.0, \
        f"ORACLE must pass the reverse bar, got {orac_m3['reverse_margin']}"
    assert orac_m3["reverse_margin"] > 5 * max(cont_m3["reverse_margin"], 1e-9), \
        "metric 3 must separate an order-sensitive code from an order-blind one by a wide margin"
    # (no assertion on gamma monotonicity: it is a report-only diagnostic, not a gate — an
    # order-blind code passes it, which is exactly why it was demoted.)

    print("\nHARNESS CONTROLS PASS:")
    print("  - DEAD    fails non-collapse (m1) as a collapsed encoder must")
    print("  - CONTENT passes m1 but FAILS the temporal bar -> metric 3 is not satisfiable by a")
    print("            content-only code (this is the b1r failure mode and the ablation-arm prediction)")
    print("  - ORACLE  passes both -> metric 3 is reachable when real temporal structure is encoded")

    # ---- WHICH temporal bars are actually content-controlled (structural, not synthetic)
    # reverse is a PERMUTATION of the frame multiset: any order-invariant encoder is EXACTLY
    # invariant to it, so a nonzero reverse margin cannot be produced by a content-only code.
    # A gamma warp RESAMPLES WITH REPETITION: it changes the frame multiset, so appearance
    # statistics move and an order-blind encoder responds. The gamma bars are therefore NOT
    # content-controlled -- verified here, and true by construction rather than by luck.
    assert abs(cont_m3["reverse_margin"]) < 1e-5, (
        "an order-invariant encoder must be invariant to reverse (frame-multiset permutation) "
        f"up to float noise, got {cont_m3['reverse_margin']}")
    # GATE 2 (revised): the held-out PERMUTATION margin must inherit reverse's content-control.
    assert abs(cont_m3["heldout_permutation_margin"]) < 1e-5, (
        "GATE 2 must be content-controlled: an order-invariant encoder must be invariant to the "
        f"probe permutations, got {cont_m3['heldout_permutation_margin']}")
    assert orac_m3["heldout_permutation_margin"] >= 0.5, (
        "a genuinely temporal code must clear the held-out permutation bar, got "
        f"{orac_m3['heldout_permutation_margin']}")

    gamma_contaminated = cont_m3["gamma_monotonicity_rho_REPORT_ONLY"] >= 0.7
    print("\nBAR-VALIDITY EVIDENCE (structural — why the bars were revised pre-read):")
    print(f"  GATE reverse margin        CONTENT {cont_m3['reverse_margin']:.3f}  ORACLE "
          f"{orac_m3['reverse_margin']:.3f}   -> CONTENT-CONTROLLED (permutation; exact invariance)")
    print(f"  GATE heldout-perm margin   CONTENT {cont_m3['heldout_permutation_margin']:.3f}  ORACLE "
          f"{orac_m3['heldout_permutation_margin']:.3f}   -> CONTENT-CONTROLLED (probe-only "
          "permutations; multiset preserved)")
    print(f"  report gamma monotonicity  CONTENT {cont_m3['gamma_monotonicity_rho_REPORT_ONLY']:.3f}  "
          f"ORACLE {orac_m3['gamma_monotonicity_rho_REPORT_ONLY']:.3f}   -> "
          f"{'NOT content-controlled: an order-blind code passes rho>=0.7' if gamma_contaminated else 'ok'}")
    print(f"  report heldout-gamma marg  CONTENT {cont_m3['heldout_gamma_margin_REPORT_ONLY']:.3f}  "
          f"ORACLE {orac_m3['heldout_gamma_margin_REPORT_ONLY']:.3f}   -> weakly discriminative "
          "(warps move the frame multiset)")
    print("  => GATES = reverse + held-out permutation (both multiset-preserving). Gamma is "
          "diagnostics only.\n     Advisor-ruled 2026-07-24, pre-read.")


if __name__ == "__main__":
    main()
