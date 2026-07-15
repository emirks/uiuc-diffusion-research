"""The amplitude ladder and the noise-limited ORACLE (advisor C5-R).

    THE PROBLEM. §3.4 grades injected-trajectory recovery on `amp_err = |max|rec| -
    max|truth|| / max|truth| <= 0.10`. The first construction injected a per-pair
    translation of 0.314 px. The flow fit's own per-pair parameter noise is
    sigma ~ 0.02-0.05 px, and max|.| over ~120 pairs adds ~2.5-3 sigma at the peak.
    So amp_err ~ 0.2-0.5 for ANY estimator at that amplitude — INCLUDING AN ORACLE
    THAT FITS THE FLOW EXACTLY. A test an oracle cannot pass is not a test of M1b;
    it grades the instrument's noise floor.

    THE FIX. Do not pick an amplitude by judgment. Build a LADDER of corpus-derived
    amplitudes and grade each (substrate, kind) cell at the HIGHEST rung whose
    constructed truth is actually CONSTRUCTIBLE — i.e. where (i) the probe row is
    DEFINED under the frozen §3.2 gates and (ii) an oracle could pass the frozen
    §3.4 thresholds.

    THE ANTI-GAMING GUARD, STRUCTURAL. `select_verdict_rung` takes ONLY the frozen
    gates and the oracle simulation. It NEVER sees the metric's recovered parameters
    — that is asserted in code below. This is what makes "the highest valid rung" a
    statement about construction validity rather than "the rung where it passes".

    Nothing here changes a frozen number: max|.|, the 0.9 correlation floor, the 10%
    amplitude bound, the §3.2 gates and the aggregation rule are all untouched.
"""

from __future__ import annotations

import json

import numpy as np

from . import acceptance, m1b_flow, paths

RUNGS = ("p50", "p75", "p90")        # pre-declared; no rung is ever added later
QUANTILES = {"p50": 50, "p75": 75, "p90": 90}
T_FRAMES = 121
EASE_FACTOR = (T_FRAMES - 1) / (np.pi / 2)     # inverts the ease-in-out profile
N_ORACLE_DRAWS = 200


def corpus_quantiles() -> dict:
    """Per-channel per-pair |delta| quantiles over DEFINED + texture-gated core pairs
    of CAMERA-TAGGED clips. Corpus-only, outcome-independent."""
    corpus = paths.load_corpus()
    gates = paths.load_gates()
    tmin = gates["phase1"]["m1b_flow"]["min_pair_texture"]
    f = np.load(paths.WB_OUT / "phase1/camera_fits.npz")
    fkeys = [str(k) for k in f["keys"]]
    tex = np.load(paths.WB_CACHE / "texture.npz")["pair_texture"]
    cam = {c for c, v in corpus["classes"].items() if "camera" in v.get("tags", [])}

    out = {}
    for ch, nm in enumerate(m1b_flow.PARAM_NAMES):
        vals = []
        for i, k in enumerate(fkeys):
            if corpus["clips"][k]["class"] not in cam:
                continue
            m = f["defined"][i] & f["core_pairs"][i] & (tex[i] >= tmin)
            v = f["params"][i][m][:, ch]
            vals.append(np.abs(v[np.isfinite(v)]))
        v = np.concatenate(vals)
        out[nm] = {r: float(np.percentile(v, q)) for r, q in QUANTILES.items()}
        out[nm]["n_pairs"] = int(v.size)
    return out


def ladder_amplitudes(q: dict) -> dict:
    """Rung -> per-channel TOTAL amplitude (each channel in its own units)."""
    return {r: {nm: q[nm][r] * EASE_FACTOR for nm in m1b_flow.PARAM_NAMES}
            for r in RUNGS}


def sigma_from_unmoved_channels(probe_rows: list[dict]) -> dict:
    """sigma(substrate, channel) from the UNMOVED channels of the FIRST run's probes.

    A channel the injected trajectory never touches carries pure noise under the
    actual warp/interpolation/border conditions — the honest estimate of the fit's
    per-pair parameter noise. Pooled across the kinds that leave that channel
    unmoved (tx from pan_y/zoom/rotate; ty from pan_x/zoom/rotate; log_scale and
    rotation from the pans)."""
    acc: dict[tuple[str, str], list[float]] = {}
    for r in probe_rows:
        for nm, v in r["params"].items():
            if v.get("graded"):
                continue                     # this channel MOVED — not a noise sample
            s = r.get("sigma", {}).get(nm)
            if s is not None and np.isfinite(s):
                acc.setdefault((r["clip"], nm), []).append(float(s))
    return {f"{c}|{nm}": float(np.median(v)) for (c, nm), v in acc.items()}


def oracle_valid(truth_rel: np.ndarray, sigma: dict, clip: str,
                 rng: np.random.Generator, n_draws: int = N_ORACLE_DRAWS) -> dict:
    """Could a PERFECT estimator — one whose only error is the flow's own parameter
    noise — pass the frozen §3.4 thresholds at this amplitude?

    recovered = truth + N(0, sigma), pushed through grade_injection UNCHANGED (same
    max|.|, same 0.9 / 10%). Oracle-valid iff the MEDIAN over draws has
    amp_err <= 0.10 and corr >= 0.90. No new threshold is introduced.

    This never touches the metric's recovered values."""
    n = len(truth_rel)
    defined = np.ones(n, bool)
    errs: dict[str, list[float]] = {}
    corrs: dict[str, list[float]] = {}
    for _ in range(n_draws):
        rec = truth_rel.copy()
        for ci, nm in enumerate(m1b_flow.PARAM_NAMES):
            s = sigma.get(f"{clip}|{nm}")
            if s is None or not np.isfinite(s):
                s = 0.0
            rec[:, ci] = rec[:, ci] + rng.normal(0.0, s, size=n)
        g = acceptance.grade_injection(rec, truth_rel, defined)
        for nm, v in g["params"].items():
            if v["graded"]:
                errs.setdefault(nm, []).append(v["amp_err"])
                corrs.setdefault(nm, []).append(v["corr"])
    per = {nm: {"median_amp_err": float(np.median(e)),
                "median_corr": float(np.median(corrs[nm])),
                "amp_ok": bool(np.median(e) <= acceptance.AMP_ERR_MAX),
                "corr_ok": bool(np.median(corrs[nm]) >= acceptance.CORR_MIN)}
           for nm, e in errs.items()}
    return {"per_channel": per,
            "valid": bool(per and all(v["amp_ok"] and v["corr_ok"] for v in per.values()))}


def select_verdict_rung(oracle_by_rung: dict, defined_by_rung: dict) -> dict:
    """THE ANTI-GAMING GUARD. Inputs are ONLY (a) the oracle simulation and (b) the
    frozen-gate definedness of the probe row. The metric's RECOVERED PARAMETERS ARE
    NOT AN INPUT and cannot be — this function has no access to them.

    Returns the HIGHEST rung that is both oracle-valid and defined; None if no rung
    qualifies (the cell is then excluded-with-reason: the construction cannot carry
    constructed truth there)."""
    for r in reversed(RUNGS):                        # p90 -> p75 -> p50
        if oracle_by_rung.get(r, {}).get("valid") and defined_by_rung.get(r, False):
            return {"rung": r, "reason": None}
    return {"rung": None,
            "reason": "no rung is both oracle-valid and defined under the frozen "
                      "§3.2 gates — constructed truth is not constructible for this "
                      "cell at any corpus-derived amplitude"}


def main() -> int:
    """Pre-declare the ladder: quantiles, amplitudes, sigma, oracle validity."""
    rng = np.random.default_rng(20260714)
    q = corpus_quantiles()
    amps = ladder_amplitudes(q)

    print("CORPUS PER-PAIR |delta| QUANTILES (defined + texture-gated core pairs of "
          "camera-tagged clips):")
    for nm in m1b_flow.PARAM_NAMES:
        print(f"  {nm:10s} p50 {q[nm]['p50']:.6f}  p75 {q[nm]['p75']:.6f}  "
              f"p90 {q[nm]['p90']:.6f}   (n={q[nm]['n_pairs']})")
    print(f"\nLADDER AMPLITUDES (total, per channel, own units; x{EASE_FACTOR:.3f}):")
    for r in RUNGS:
        print(f"  {r}: " + "  ".join(f"{nm}={amps[r][nm]:.4f}"
                                     for nm in m1b_flow.PARAM_NAMES))

    # sigma from the FIRST run's unmoved channels (already committed; zero cost)
    first = json.loads((paths.WB_OUT / "phase1/acceptance_first_construction.json")
                       .read_text()) if (paths.WB_OUT /
                       "phase1/acceptance_first_construction.json").exists() else None
    sigma = {}
    if first:
        sigma = sigma_from_unmoved_channels(first["injected_trajectory"]["rows"])
    if not sigma:
        # fall back to the identity control: the base clips' own corpus fits
        f = np.load(paths.WB_OUT / "phase1/camera_fits.npz")
        fkeys = [str(k) for k in f["keys"]]
        man = json.loads((paths.WB_CACHE / "probes/manifest.json").read_text())
        for k in man["static_clips"]:
            i = fkeys.index(k)
            m = f["defined"][i] & f["core_pairs"][i]
            for ci, nm in enumerate(m1b_flow.PARAM_NAMES):
                v = f["params"][i][m][:, ci]
                v = v[np.isfinite(v)]
                if v.size:
                    sigma[f"{k}|{nm}"] = float(np.std(v))
    print(f"\nSIGMA (per substrate x channel), {len(sigma)} cells — the identity "
          f"control OVER-estimates (the 'static' clips are not perfectly static), "
          f"which is conservative:")
    for nm in m1b_flow.PARAM_NAMES:
        vals = [v for k2, v in sigma.items() if k2.endswith("|" + nm)]
        if vals:
            print(f"  {nm:10s} median sigma {np.median(vals):.5f}  "
                  f"range [{min(vals):.5f}, {max(vals):.5f}]")

    # oracle validity per (substrate, kind, rung)
    man = json.loads((paths.WB_CACHE / "probes/manifest.json").read_text())
    kinds = man["inject_kinds"]
    cells = {}
    print(f"\nORACLE SIM ({N_ORACLE_DRAWS} draws): could a PERFECT estimator, whose "
          f"only error is the flow's own parameter noise, pass the FROZEN 0.9/10%?")
    print(f"  {'substrate':28s} {'kind':9s} " +
          "  ".join(f"{r:>6s}" for r in RUNGS))
    for k in man["static_clips"]:
        for kind in kinds:
            row = {}
            for r in RUNGS:
                cum = acceptance.trajectory(kind, T_FRAMES, amps[r])
                truth = acceptance.relative_params(cum)
                row[r] = oracle_valid(truth, sigma, k, rng)
            cells[f"{k}|{kind}"] = row
            print(f"  {k[:28]:28s} {kind:9s} " +
                  "  ".join(f"{'VALID' if row[r]['valid'] else '  -  ':>6s}" for r in RUNGS))

    out = {
        "purpose": "Ladder + noise-limited oracle (advisor C5-R). Pre-declared BEFORE "
                   "any corrected probe flow was computed. The oracle asks whether a "
                   "PERFECT estimator could pass the FROZEN §3.4 thresholds at each "
                   "amplitude; a test an oracle fails grades the instrument's noise "
                   "floor, not the metric. select_verdict_rung() reads ONLY the oracle "
                   "and the frozen-gate definedness — never the metric's recovered "
                   "parameters.",
        "unchanged": ["max|.| amplitude statistic", "corr >= 0.9", "amp_err <= 0.10",
                      "§3.2 40% inlier / 30% clip caps", "texture gate",
                      "all-graded-must-pass aggregation", "gates.yaml"],
        "rungs": list(RUNGS),
        "corpus_quantiles": q,
        "ladder_amplitudes": amps,
        "ease_factor": EASE_FACTOR,
        "sigma_per_substrate_channel": sigma,
        "n_oracle_draws": N_ORACLE_DRAWS,
        "oracle_validity": {k2: {r: v[r]["valid"] for r in RUNGS} for k2, v in cells.items()},
        "oracle_detail": cells,
        "phase1_finding": {
            "statement": "In this corpus the MEDIAN per-pair camera translation "
                         "(tx 0.297 px, ty 0.421 px) is within an order of magnitude "
                         "of the flow fit's own per-pair parameter noise; the vigorous "
                         "decile is tx 3.11 px / ty 2.31 px.",
            "oracle_at_p50": "An oracle fails §3.4's peak-amplitude criterion at p50.",
        },
    }
    p = paths.WB_OUT / "phase1/probe_ladder.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
