"""IV1 / IV2 — the instrument-validity preconditions (E1PRIME_DIRECTIVE §2.3).

    "The kill verdict binds the hypothesis ONLY IF both hold; otherwise the recorded
     verdict is INSTRUMENT-INVALID.
       IV1 (effect vs nothing): binary LOO 1-NN over pooled signatures {223 real clips,
         223 rendered nulls} >= 0.90 accuracy.
       IV2 (snap vs nothing): binary 1-NN {hard-cut splice, rendered lerp}, reusing the
         existing Bar-6 splice construction, >= 0.90.
     Rationale: a signature that cannot distinguish an effect from its absence, or a cut
     from a crossfade — the objects it was designed to make identifiable — cannot issue
     findings about class identity."

THE m~ == 0 TAUTOLOGY (PREREG §P5a — DISCLOSED BEFORE ANY IV NUMBER EXISTED).
controls.make_lerp returns concat([prefix, mid, suffix]), so a rendered null's own
first-9/last-8 frames ARE the prefix/suffix it was built from: the rendered null OF a
rendered null is that null, bit-for-bit. Every "nothing" object therefore has
m~ = m_lerp - m_lerp == 0 EXACTLY, on every frame. The "nothing" class is a CONSTANT on
the m~ channel, so any real clip with non-zero m~ separates from it there for purely
structural reasons.

Consequences, all stated before the run:
  - IV1 and IV2 certify LESS than their names suggest.
  - They are computed EXACTLY AS REGISTERED anyway (the executor does not redesign the
    owner's precondition), and their registered verdicts are the ones that bind.
  - A NON-GATING (a_hat, b_hat)-only column is reported beside each, on the two channels
    that are NOT structurally constant for the "nothing" class. If the full signature
    passes while that column sits at chance, the pass rests on the tautological channel.
  - The tautology does NOT make the IV vacuous: if real clips ALSO have m~ ~ 0 (real
    effects do not depart from their own rendered null), the IV FAILS.

The 1-NN is the frozen exam kernel (report.retrieval_eval) with binary labels — the same
function that judged the incumbent, never a reimplementation.
"""

from __future__ import annotations

import numpy as np

from ..report import retrieval_eval
from . import e1prime


def binary_1nn(sigs: list[np.ndarray | None], labels: list[str]) -> dict:
    """Binary LOO 1-NN over pooled signatures. Undefined signatures are dropped and
    COUNTED (never scored as wrong, never silently ignored) — §1.5."""
    D = e1prime.distance_matrix(sigs)
    r = retrieval_eval(D, labels)
    return {
        "accuracy": float(r["accuracy_1nn"]),
        "coverage": float(r["coverage"]),
        "n_pooled": len(labels),
        "n_defined": int(sum(s is not None for s in sigs)),
        "per_class_recall": r["per_class_recall"],
        "chance": float(r["chance"]),
    }


def ab_only(sigs: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """The NON-GATING disclosure column: the same signatures restricted to (a_hat,
    b_hat) — the two channels that are not structurally constant for a rendered null."""
    return [None if s is None else s[:, :2] for s in sigs]


def check(name: str, sigs: list[np.ndarray | None], labels: list[str],
          min_accuracy: float) -> dict:
    """One IV precondition, computed as a fact + its disclosure column."""
    full = binary_1nn(sigs, labels)
    disc = binary_1nn(ab_only(sigs), labels)
    return {
        "name": name,
        "min_accuracy": float(min_accuracy),
        "accuracy": full["accuracy"],
        "pass": bool(full["accuracy"] >= min_accuracy),
        "coverage": full["coverage"],
        "n_pooled": full["n_pooled"],
        "n_defined": full["n_defined"],
        "per_class_recall": full["per_class_recall"],
        "chance": full["chance"],
        "disclosure_a_hat_b_hat_only": {
            "accuracy": disc["accuracy"],
            "coverage": disc["coverage"],
            "gating": False,
            "why": "the m~ channel is EXACTLY 0 for every rendered null by construction "
                   "(make_lerp is idempotent on its own endpoints), so the full-signature "
                   "1-NN is partly tautological. This column uses only the two channels "
                   "that are not structurally constant for the 'nothing' class. It is "
                   "reported so owner review can see whether the IV verdict rests on the "
                   "tautological channel. IT DOES NOT ALTER THE VERDICT.",
        },
    }


def gate(iv1: dict, iv2: dict) -> dict:
    """§2.3, mechanically. Both must hold for the kill verdict to bind the HYPOTHESIS."""
    ok = bool(iv1["pass"] and iv2["pass"])
    return {
        "rule": "E1PRIME_DIRECTIVE §2.3: the kill verdict binds the hypothesis only if "
                "BOTH instrument-validity preconditions hold (>= 0.90 each); otherwise "
                "the recorded verdict is INSTRUMENT-INVALID and the workbench closes "
                "UNADJUDICATED, with no repair attempts (§2.6 case 2).",
        "iv1": {"accuracy": iv1["accuracy"], "min": iv1["min_accuracy"], "pass": iv1["pass"]},
        "iv2": {"accuracy": iv2["accuracy"], "min": iv2["min_accuracy"], "pass": iv2["pass"]},
        "instrument_valid": ok,
        "verdict": ("INSTRUMENT VALID — the kill rule binds the hypothesis"
                    if ok else
                    "INSTRUMENT-INVALID — the program closes UNADJUDICATED; the kill "
                    "rule's outcome does not bind the hypothesis. No repair attempts."),
    }
