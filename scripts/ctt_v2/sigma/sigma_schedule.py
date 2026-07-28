"""CTT v2 — the per-stratum sigma distribution, computed ANALYTICALLY (A9 §3 item 1).

A9's noise-schedule prescription forbids two things and orders one:

    FORBIDDEN  padding/resampling S4 to 121 frames (interpolation invents frames; retiming
               corrupts dynamics, half the frozen metric's definition of manner)
    FORBIDDEN  patching the timestep sampler to force a uniform shift (a recipe change
               smuggled inside a "pure dataset intervention" claim)
    ORDERED    "compute and archive the exact per-stratum sigma distributions analytically
               from the root manifest; stamp them into DATASET.md"

This module is the ORDERED item.  It imports nothing from the trainer at module level and
modifies nothing: it re-derives the trainer's sampler in closed form and then *checks itself*
against the trainer's own class by Monte Carlo.

THE SHIFT LAW (verified first-hand in
`LTX-2-cond-bleed-fix/packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py:121-134`)
--------------------------------------------------------------------------------------------
    m = (max_shift - min_shift) / (max_tokens - min_tokens) = (2.05 - 0.95) / (4096 - 1024)
      = 1.1 / 3072
    b = min_shift - m * min_tokens = 0.95 - (1.1/3072) * 1024 = 0.5833333...
    shift = m * seq_length + b                       # NOT clamped: extrapolates freely

`seq_length` is the TARGET token count and nothing else.  Verified by call chain:
`trainer.py:375` -> `flexible.py:400 _initialize_noisy_target(latents, ...)` -> `:482
timestep_sampler.sample_for(latents)`, and `_initialize_noisy_target` runs at **Step 3**,
BEFORE the reference latents are concatenated at Step 5 (`flexible.py:655-666` ->
`_apply_reference_condition`, `:689 torch.cat([cond_latents, noisy_latents], dim=1)`).
So the IC-LoRA reference does NOT double the token count the shift is computed from.
`seq_length = F_latent * H_latent * W_latent` (patch size 1; `_patchify_latent_data`).

THE SAMPLER, exactly (`ShiftedLogitNormalTimestepSampler`, defaults std=1.0, eps=1e-3,
uniform_prob=0.1 — `ic_gen.yaml: timestep_sampling_params: {}` means defaults)
--------------------------------------------------------------------------------------------
    mu    = shift
    z     ~ Normal(mu, std)
    x     = sigmoid(z)
    p005  = sigmoid(mu - 2.5758*std)      # 0.5th  percentile of the logit-normal
    p999  = sigmoid(mu + 3.0902*std)      # 99.9th percentile
    y     = (x - p005) / (p999 - p005)                       # stretch to ~[0,1]
    y     = y if y >= eps else 2*eps - y                     # REFLECT (not clip!)
    y     = clamp(y, 0, 1)
    sigma = y with prob (1 - uniform_prob), else Uniform(eps, 1)

Two branches matter and are easy to miss:
  * the REFLECTION sends the bottom ~0.5% of z (where the stretched value falls below eps)
    to `2*eps - y`, i.e. it maps the extreme LOW tail up to ~0.27 (S4) / ~0.77 (121f).
    It is a real, non-negligible feature of the low-sigma mass, so it is modelled exactly.
  * `clamp(...,1)` puts an exact POINT MASS of (1 - Phi(3.0902)) * (1 - uniform_prob)
    = 0.001 * 0.9 = 0.0009 at sigma == 1.0.

Usage
-----
    python scripts/ctt_v2/sigma/sigma_schedule.py                 # table + JSON, no GPU
    python scripts/ctt_v2/sigma/sigma_schedule.py --mc 4000000    # + Monte-Carlo self-check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
TRAINER_SRC = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer/src"

# --------------------------------------------------------------------------------------
# The shift law, re-derived from the sampler's OWN default arguments rather than copied.
# `read_shift_law_from_trainer()` below proves these four numbers against the source file.
# --------------------------------------------------------------------------------------
MIN_TOKENS, MAX_TOKENS = 1024, 4096
MIN_SHIFT, MAX_SHIFT = 0.95, 2.05
SHIFT_M = (MAX_SHIFT - MIN_SHIFT) / (MAX_TOKENS - MIN_TOKENS)      # 1.1 / 3072
SHIFT_B = MIN_SHIFT - SHIFT_M * MIN_TOKENS                          # 0.5833333...

#: sampler defaults (`ShiftedLogitNormalTimestepSampler.__init__`)
STD = 1.0
EPS = 1e-3
UNIFORM_PROB = 0.1
Z_999 = 3.0902
Z_005 = -2.5758

#: A9 §4 mix weights (DOSSIER §12), DERIVED from the single ruled source
#: `root_common.STRATUM_WEIGHTS_PCT` — S0 15 / S1 6 / S2 total 69 / S4 10 — with the
#: S2a:S2b split computed pro-rata from the FROZEN assembled base pair counts (A12,
#: `misc/ctt_v2_final/PREREG_mix_inputs.json`).  Never restated as a literal here: a
#: private copy of the mix is exactly how DATASET §11.1's stale-weights landmine arose.
#:
#: The split is IMMATERIAL to every number this module computes — S2a and S2b carry the
#: same geometry, so only the S2 TOTAL can enter a sigma law — and `_weights_pct()` asserts
#: that invariance rather than asking the reader to take it on trust.  It matters only for
#: the per-stratum display rows, which must not print a number the mix contract disowns.
sys.path.insert(0, str(HERE.parent))
import root_common as rc  # noqa: E402

PRORATA_GROUPS = {k: tuple(v) for k, v in rc.PRORATA_GROUPS.items()}


def _weights_pct() -> tuple[dict, str]:
    counts, src = None, None
    if rc.PREREG_MIX_INPUTS.exists():
        rec = rc.read_json(rc.PREREG_MIX_INPUTS)
        counts = rec.get("frozen_assembled_base_pair_counts") or None
        src = f"{rc.PREREG_MIX_INPUTS} (frozen counts {counts})"
    if not counts or not all(counts.get(m) for g in rc.PRORATA_GROUPS.values() for m in g):
        # No frozen counts yet.  The S2 TOTAL is ruled and is all that can affect a sigma
        # number, so fall back to an even display split and SAY SO — never silently.
        counts = {m: 1 for g in rc.PRORATA_GROUPS.values() for m in g}
        src = (f"{rc.PREREG_MIX_INPUTS} ABSENT — S2a:S2b shown as an even DISPLAY split; "
               f"the ruled S2 total is unaffected and no computed number changes")
        print(f"[sigma] ⚠ {src}", file=sys.stderr)
    weights, _split = rc.expand_prorata_weights(rc.STRATUM_WEIGHTS_PCT, counts)
    return weights, src


WEIGHTS_PCT, WEIGHTS_SOURCE = _weights_pct()

#: The four `sigma_tracker.SigmaBucketTracker` default buckets — the diagnostic A9 §3 item 2
#: asks to split by stratum, so the analytic table is reported in the SAME bins.
TRACKER_BUCKETS = [0.0, 0.25, 0.5, 0.75, 1.0]

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# --------------------------------------------------------------------------------------
# Latent geometry per stratum.
#
# 🔴 The (5,20,15) figure in A9 §3 / §50 IS NOT ACHIEVABLE and the ruling text is wrong on
# this point — recorded in DOSSIER §10.9 by the operator who built the encode.  refVFX
# I2V_LoRA is natively 832x464; 464 is not a multiple of the VAE spatial factor 32
# (464/32 = 14.5) and `process_videos.py:parse_resolution_buckets` rejects 832x464x33
# outright.  The delivered artifact is the minimal legal deviation `832x448x33` — for an
# 832x464 source a PURE 16-row centre crop with NO resampling — giving latent (5,14,26).
# (20,15) is the S2/corpus grid (480x640), which no bucket derived from 832x464 can yield.
#
# CONSEQUENCE FOR THE SHIFT, which the ruling did not follow through:
#     A9's premise   5*20*15 = 1,500 tokens -> shift 1.1204
#     the artifact   5*14*26 = 1,820 tokens -> shift 1.2350
# Both are computed and reported below so the delta is explicit rather than absorbed.
# --------------------------------------------------------------------------------------
STRATA_GEOMETRY = {
    "S0":  {"px": (480, 640, 121), "latent": (16, 20, 15), "fps": 24.0,
            "note": "certified corpus format (ic_gen root)"},
    "S1":  {"px": (480, 640, 121), "latent": (16, 20, 15), "fps": 24.0,
            "note": "specialist renders, corpus format"},
    "S2a": {"px": (480, 640, 121), "latent": (16, 20, 15), "fps": 24.0,
            "note": "gl-transitions over corpus content"},
    "S2b": {"px": (480, 640, 121), "latent": (16, 20, 15), "fps": 24.0,
            "note": "gl-transitions over HumanVid content"},
    "S4":  {"px": (832, 448, 33), "latent": (5, 14, 26), "fps": 16.0,
            "note": "refVFX I2V_LoRA, 832x448x33 (16-row centre crop of native 832x464), "
                    "33f @ 16fps native — NOT the (5,20,15) A9 §3 asserts"},
}

#: kept only to quantify how wrong A9's premise was; never used for any assertion
S4_A9_PREMISE_LATENT = (5, 20, 15)


# --------------------------------------------------------------------------------------
# closed-form pieces
# --------------------------------------------------------------------------------------
def tokens(latent: tuple[int, int, int]) -> int:
    f, h, w = latent
    return f * h * w


def shift_for_tokens(n_tokens: int) -> float:
    return SHIFT_M * n_tokens + SHIFT_B


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _phi(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class SigmaLaw:
    """The exact law of one `sample()` draw at a given shift. No sampling anywhere."""

    def __init__(self, mu: float, std: float = STD, eps: float = EPS,
                 uniform_prob: float = UNIFORM_PROB):
        self.mu, self.std, self.eps, self.uniform_prob = mu, std, eps, uniform_prob
        self.p005 = _sigmoid(mu + Z_005 * std)
        self.p999 = _sigmoid(mu + Z_999 * std)
        self.delta = self.p999 - self.p005
        #: y at z -> -inf, i.e. the most negative pre-reflection value
        self.y_min = -self.p005 / self.delta
        #: the z at which the stretched value equals eps (below it, reflection fires)
        self.z_eps = self._z_for_y(eps)
        #: the z at which the stretched value equals 1 (above it, clamp fires)
        self.z_one = mu + Z_999 * std
        self.p_reflected = _phi((self.z_eps - mu) / std)
        self.p_clamped_at_one = (1.0 - _phi(Z_999)) * (1.0 - uniform_prob)

    # -- geometry -----------------------------------------------------------------------
    def _z_for_y(self, y: float) -> float:
        """z such that (sigmoid(z) - p005)/delta == y.  Requires p005 + y*delta in (0,1)."""
        x = self.p005 + y * self.delta
        if x <= 0.0:
            return -math.inf
        if x >= 1.0:
            return math.inf
        return _logit(x)

    # -- CDF ----------------------------------------------------------------------------
    def cdf_stretched(self, v: float) -> float:
        """P(g(z) <= v) for the non-uniform branch, reflection and clamp handled exactly."""
        if v < 0.0:
            return 0.0
        if v >= 1.0:
            return 1.0
        # main (unreflected) branch: z in [z_eps, z_one], y increasing in z
        z_v = self._z_for_y(v)
        z_hi = min(max(z_v, self.z_eps), self.z_one)
        main = _phi((z_hi - self.mu) / self.std) - _phi((self.z_eps - self.mu) / self.std)
        main = max(0.0, main)
        # reflected branch: z < z_eps, sigma = 2*eps - y(z), DEcreasing in z.
        #   sigma <= v  <=>  y(z) >= 2*eps - v  <=>  z >= z_{2eps-v}
        refl = 0.0
        if v >= self.eps:
            y_thr = 2.0 * self.eps - v
            if y_thr <= self.y_min:
                refl = self.p_reflected                     # whole branch is <= v
            else:
                z_thr = self._z_for_y(y_thr)
                refl = max(0.0, self.p_reflected - _phi((z_thr - self.mu) / self.std))
        return min(1.0, main + refl)

    def cdf_uniform(self, v: float) -> float:
        if v <= self.eps:
            return 0.0
        if v >= 1.0:
            return 1.0
        return (v - self.eps) / (1.0 - self.eps)

    def cdf(self, v: float) -> float:
        return (self.uniform_prob * self.cdf_uniform(v)
                + (1.0 - self.uniform_prob) * self.cdf_stretched(v))

    # -- moments and quantiles ----------------------------------------------------------
    def quantile(self, q: float, tol: float = 1e-12) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    def moments(self, n: int = 200_001) -> tuple[float, float]:
        """E[sigma], sd[sigma] by Simpson quadrature of 1 - CDF; exact to ~1e-9.

        E[X] = int_0^1 (1 - F(v)) dv ;  E[X^2] = int_0^1 2v (1 - F(v)) dv.
        """
        if n % 2 == 0:
            n += 1
        h = 1.0 / (n - 1)
        s1 = s2 = 0.0
        for i in range(n):
            v = i * h
            w = 1.0 if i in (0, n - 1) else (4.0 if i % 2 else 2.0)
            sf = 1.0 - self.cdf(v)
            s1 += w * sf
            s2 += w * 2.0 * v * sf
        m1 = s1 * h / 3.0
        m2 = s2 * h / 3.0
        return m1, math.sqrt(max(0.0, m2 - m1 * m1))

    def bucket_mass(self, edges: list[float]) -> dict[str, float]:
        out = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            # top bucket is closed on the right so the sigma==1 point mass lands in it
            m = self.cdf(hi) - self.cdf(lo) if hi < 1.0 else 1.0 - self.cdf(lo)
            out[f"{lo:.2f}-{hi:.2f}"] = m
        return out


class MixtureLaw:
    """Weighted mixture of per-stratum laws — the pooled sigma the run actually trains on."""

    def __init__(self, parts: list[tuple[float, SigmaLaw]]):
        tot = sum(w for w, _ in parts)
        self.parts = [(w / tot, law) for w, law in parts]

    def cdf(self, v: float) -> float:
        return sum(w * law.cdf(v) for w, law in self.parts)

    quantile = SigmaLaw.quantile
    moments = SigmaLaw.moments
    bucket_mass = SigmaLaw.bucket_mass


# --------------------------------------------------------------------------------------
# self-checks
# --------------------------------------------------------------------------------------
def _load_trainer_samplers():
    """Load `ltx_trainer/timestep_samplers.py` DIRECTLY from the certified trainer source.

    Deliberately bypasses `import ltx_trainer` — the package `__init__` pulls in `rich`,
    which the research env does not have, and this module must stay runnable on a login
    node with no GPU and no trainer venv.  `timestep_samplers.py` imports only `torch`, so
    loading the file by path gives the REAL class with no substitutions.
    """
    import importlib.util  # noqa: PLC0415

    path = TRAINER_SRC / "ltx_trainer/timestep_samplers.py"
    spec = importlib.util.spec_from_file_location("_ctt_timestep_samplers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__ctt_source__ = str(path)
    return mod


def read_shift_law_from_trainer() -> dict:
    """Prove the four constants against the trainer source instead of trusting this file."""
    S = _load_trainer_samplers().ShiftedLogitNormalTimestepSampler

    sig = __import__("inspect").signature(S._get_shift_for_sequence_length)
    d = {k: v.default for k, v in sig.parameters.items() if v.default is not v.empty}
    m = (d["max_shift"] - d["min_shift"]) / (d["max_tokens"] - d["min_tokens"])
    b = d["min_shift"] - m * d["min_tokens"]
    init = __import__("inspect").signature(S.__init__)
    idef = {k: v.default for k, v in init.parameters.items() if v.default is not v.empty}
    rec = {
        "source": "ltx_trainer/timestep_samplers.py:_get_shift_for_sequence_length",
        "defaults": d, "m": m, "b": b,
        "sampler_init_defaults": idef,
        "spot_checks": {str(t): S._get_shift_for_sequence_length(t) for t in (1024, 1500, 1820, 4096, 4800)},
    }
    bad = []
    if abs(m - SHIFT_M) > 1e-15:
        bad.append(f"m: trainer {m!r} vs module {SHIFT_M!r}")
    if abs(b - SHIFT_B) > 1e-12:
        bad.append(f"b: trainer {b!r} vs module {SHIFT_B!r}")
    for k, want in (("std", STD), ("eps", EPS), ("uniform_prob", UNIFORM_PROB)):
        if abs(float(idef[k]) - want) > 1e-15:
            bad.append(f"{k}: trainer {idef[k]!r} vs module {want!r}")
    for t in (1024, 1500, 1820, 4096, 4800):
        got = S._get_shift_for_sequence_length(t)
        if abs(got - shift_for_tokens(t)) > 1e-12:
            bad.append(f"shift({t}): trainer {got!r} vs module {shift_for_tokens(t)!r}")
    rec["agrees_with_module"] = not bad
    rec["disagreements"] = bad
    return rec


def mc_check(laws: dict[str, SigmaLaw], n: int, seed: int = 42) -> dict:
    """Draw from the trainer's OWN sampler and compare its empirical CDF to the closed form."""
    import torch  # noqa: PLC0415

    sampler = _load_trainer_samplers().ShiftedLogitNormalTimestepSampler()
    grid = [i / 200.0 for i in range(201)]
    out = {}
    for name, law in laws.items():
        torch.manual_seed(seed)
        s = sampler.sample(batch_size=n, seq_length=law.n_tokens).double()
        s_sorted, _ = torch.sort(s)
        ks = 0.0
        worst = None
        for v in grid:
            emp = float(torch.searchsorted(s_sorted, torch.tensor(v, dtype=torch.float64),
                                           right=True)) / n
            d = abs(emp - law.cdf(v))
            if d > ks:
                ks, worst = d, v
        out[name] = {
            "n_draws": n, "seed": seed,
            "ks_sup_deviation": ks, "at_sigma": worst,
            "empirical_mean": float(s.mean()), "analytic_mean": law.mean,
            "mean_abs_error": abs(float(s.mean()) - law.mean),
            "empirical_frac_at_one": float((s >= 1.0).double().mean()),
            "analytic_point_mass_at_one": law.p_clamped_at_one,
        }
    return out


# --------------------------------------------------------------------------------------
def build(weights: dict[str, float] | None = None) -> dict:
    weights = dict(weights or WEIGHTS_PCT)
    # A12 — the DERIVED S2a:S2b split cannot move any number computed here, because the
    # members of a pro-rata group share a geometry and a sigma law is a function of the
    # geometry alone.  Asserted rather than assumed: if a future stratum joins S2 with a
    # different grid, this stops the module from silently reporting a split-dependent
    # pooled sigma while the split itself is derived from counts.
    for g, members in PRORATA_GROUPS.items():
        geos = {tuple(STRATA_GEOMETRY[m]["latent"]) for m in members if m in STRATA_GEOMETRY}
        assert len(geos) <= 1, (
            f"pro-rata group {g} spans geometries {sorted(geos)}; the derived split would "
            f"then change the pooled sigma, and this module's weights would need the real "
            f"assembled counts rather than the frozen pre-registration")
    law_by_stratum: dict[str, SigmaLaw] = {}
    rows = []
    for s, geo in STRATA_GEOMETRY.items():
        n_tok = tokens(geo["latent"])
        mu = shift_for_tokens(n_tok)
        law = SigmaLaw(mu)
        law.n_tokens = n_tok
        law.mean, law.sd = law.moments()
        law_by_stratum[s] = law
        rows.append({
            "stratum": s, "weight_pct": weights.get(s, 0.0),
            "px_whf": list(geo["px"]), "latent_fhw": list(geo["latent"]), "fps": geo["fps"],
            "tokens": n_tok, "shift": mu,
            "mean": law.mean, "sd": law.sd,
            "quantiles": {f"p{int(q*100)}": law.quantile(q) for q in QUANTILES},
            "tracker_buckets": law.bucket_mass(TRACKER_BUCKETS),
            "p_reflected_branch": law.p_reflected,
            "point_mass_at_sigma_1": law.p_clamped_at_one,
            "note": geo["note"],
        })

    mix = MixtureLaw([(weights[s], law_by_stratum[s]) for s in weights if weights.get(s, 0) > 0])
    mix.eps = EPS
    mix_mean, mix_sd = MixtureLaw.moments(mix)
    pooled = {
        "weights_pct": weights,
        "mean": mix_mean, "sd": mix_sd,
        "quantiles": {f"p{int(q*100)}": MixtureLaw.quantile(mix, q) for q in QUANTILES},
        "tracker_buckets": MixtureLaw.bucket_mass(mix, TRACKER_BUCKETS),
    }

    # what a 121f-only corpus would have looked like — the counterfactual the caveat needs
    only121 = MixtureLaw([(1.0, law_by_stratum["S0"])])
    only121.eps = EPS
    o_mean, o_sd = MixtureLaw.moments(only121)

    a9_tok = tokens(S4_A9_PREMISE_LATENT)
    a9_law = SigmaLaw(shift_for_tokens(a9_tok))
    a9_law.n_tokens = a9_tok
    a9_law.mean, a9_law.sd = a9_law.moments()

    return {
        "schema": "ctt_v2_sigma_schedule/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "authority": "A9 §3 item 1 (misc/ctt_v2_final/advisors/A9_s4_final_VERBATIM.md); "
                     "shift law verified in ltx_trainer/timestep_samplers.py:121-134",
        "method": "closed-form CDF of ShiftedLogitNormalTimestepSampler (reflection and "
                  "clamp modelled exactly); moments by Simpson quadrature of 1-F; "
                  "quantiles by bisection on the closed-form CDF. NO training run, NO sampling.",
        "shift_law": {"m": SHIFT_M, "b": SHIFT_B, "formula": "shift = m*tokens + b",
                      "seq_length_is": "TARGET token count only (F_lat*H_lat*W_lat); the "
                                       "IC-LoRA reference is concatenated AFTER the sigma "
                                       "draw (flexible.py Step 3 vs Step 5)"},
        "sampler": {"class": "ShiftedLogitNormalTimestepSampler", "std": STD, "eps": EPS,
                    "uniform_prob": UNIFORM_PROB, "z_999": Z_999, "z_005": Z_005,
                    "config": "ic_gen.yaml flow_matching.timestep_sampling_params: {} => defaults"},
        "per_stratum": rows,
        "distinct_shifts": sorted({r["shift"] for r in rows}),
        "pooled_mixture": pooled,
        "counterfactual_121f_only": {"mean": o_mean, "sd": o_sd,
                                     "shift": law_by_stratum["S0"].mu},
        "a9_premise_s4": {
            "latent_fhw": list(S4_A9_PREMISE_LATENT), "tokens": a9_tok, "shift": a9_law.mu,
            "mean": a9_law.mean, "sd": a9_law.sd,
            "status": "NOT ACHIEVABLE — see DOSSIER §10.9. Reported only to quantify the "
                      "delta from A9's stated premise to the delivered artifact.",
            "delta_shift_vs_artifact": law_by_stratum["S4"].mu - a9_law.mu,
        },
        "report_caveat": (
            "The round's claim is mix-level. S4's stratum-level contribution is confounded "
            "with its noise schedule: because the upstream sampler makes the logit-normal "
            "shift a deterministic function of the target token count, S4's 1,820-token "
            "samples train at shift {s4:.4f} while every 121-frame stratum trains at "
            "{s121:.4f}. S4's draws therefore concentrate at lower sigma (mean {m4:.4f} vs "
            "{m121:.4f}), attenuating its structural anti-copy signal. This is upstream "
            "by design, was not modified, and is disclosed."
        ).format(s4=law_by_stratum["S4"].mu, s121=law_by_stratum["S0"].mu,
                 m4=law_by_stratum["S4"].mean, m121=law_by_stratum["S0"].mean),
        "laws": law_by_stratum,          # stripped before JSON dump
    }


def fmt_table(rec: dict) -> str:
    L = []
    L.append("PER-STRATUM SIGMA DISTRIBUTION — analytic, no training run")
    L.append(f"shift = {SHIFT_M!r}*tokens + {SHIFT_B!r}   "
             f"(m = 1.1/3072, b = 0.5833...)")
    L.append("")
    hdr = (f"{'stratum':7s} {'w%':>5s} {'px (WxHxF)':>14s} {'latent':>10s} {'fps':>5s} "
           f"{'tokens':>7s} {'shift':>7s} {'E[s]':>7s} {'sd':>6s} "
           f"{'p10':>6s} {'p50':>6s} {'p90':>6s}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in rec["per_stratum"]:
        px = "x".join(str(v) for v in r["px_whf"])
        lat = ",".join(str(v) for v in r["latent_fhw"])
        L.append(f"{r['stratum']:7s} {r['weight_pct']:5.1f} {px:>14s} {lat:>10s} "
                 f"{r['fps']:5.1f} {r['tokens']:7d} {r['shift']:7.4f} {r['mean']:7.4f} "
                 f"{r['sd']:6.4f} {r['quantiles']['p10']:6.4f} {r['quantiles']['p50']:6.4f} "
                 f"{r['quantiles']['p90']:6.4f}")
    p = rec["pooled_mixture"]
    L.append("-" * len(hdr))
    L.append(f"{'POOLED':7s} {sum(p['weights_pct'].values()):5.1f} {'mixture':>14s} "
             f"{'-':>10s} {'-':>5s} {'-':>7s} {'-':>7s} {p['mean']:7.4f} {p['sd']:6.4f} "
             f"{p['quantiles']['p10']:6.4f} {p['quantiles']['p50']:6.4f} "
             f"{p['quantiles']['p90']:6.4f}")
    L.append("")
    L.append("sigma_tracker default buckets (the per-stratum split A9 §3 item 2 asks for)")
    bh = f"{'stratum':7s} " + " ".join(f"{k:>11s}" for k in rec["per_stratum"][0]["tracker_buckets"])
    L.append(bh)
    L.append("-" * len(bh))
    for r in rec["per_stratum"]:
        L.append(f"{r['stratum']:7s} " + " ".join(f"{v:11.5f}" for v in r["tracker_buckets"].values()))
    L.append(f"{'POOLED':7s} " + " ".join(f"{v:11.5f}" for v in p["tracker_buckets"].values()))
    L.append("")
    L.append(f"distinct shifts in the mix: {[round(s, 6) for s in rec['distinct_shifts']]}")
    a9 = rec["a9_premise_s4"]
    L.append(f"A9 §3's stated S4 premise ({','.join(str(v) for v in a9['latent_fhw'])} = "
             f"{a9['tokens']} tok -> shift {a9['shift']:.4f}) IS NOT ACHIEVABLE; the "
             f"delivered artifact is shift {rec['per_stratum'][-1]['shift']:.4f} "
             f"({a9['delta_shift_vs_artifact']:+.4f}).")
    return "\n".join(L)


def fmt_markdown(rec: dict) -> str:
    """A DATASET.md-ready block: paste as-is, no reformatting (A9 §3 item 1 'stamp')."""
    L = []
    L.append("### Noise schedule — per-stratum sigma distribution (analytic)")
    L.append("")
    L.append("The upstream timestep sampler makes the logit-normal shift a **deterministic "
             "function of the target token count** "
             "(`ltx_trainer/timestep_samplers.py:121-134`): "
             "`shift = m*tokens + b`, `m = 1.1/3072`, `b = 0.5833`. `tokens` is the target's "
             "`F_lat*H_lat*W_lat` (patch size 1); the IC-LoRA reference is concatenated "
             "*after* the sigma draw, so it does not enter. The trainer was NOT modified.")
    L.append("")
    L.append("| stratum | mix % | pixels (WxHxF) | latent (F,H,W) | fps | tokens | shift | "
             "E[σ] | sd | p10 | p50 | p90 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rec["per_stratum"]:
        px = "x".join(str(v) for v in r["px_whf"])
        lat = ",".join(str(v) for v in r["latent_fhw"])
        q = r["quantiles"]
        L.append(f"| {r['stratum']} | {r['weight_pct']:.1f} | {px} | ({lat}) | {r['fps']:.0f} "
                 f"| {r['tokens']:,} | **{r['shift']:.4f}** | {r['mean']:.4f} | {r['sd']:.4f} "
                 f"| {q['p10']:.4f} | {q['p50']:.4f} | {q['p90']:.4f} |")
    p = rec["pooled_mixture"]
    q = p["quantiles"]
    L.append(f"| **pooled** | 100.0 | mixture | — | — | — | — | {p['mean']:.4f} | "
             f"{p['sd']:.4f} | {q['p10']:.4f} | {q['p50']:.4f} | {q['p90']:.4f} |")
    L.append("")
    L.append("Mass in the four `sigma_tracker` default buckets (the bins the per-stratum "
             "training-health split would be read in):")
    L.append("")
    keys = list(rec["per_stratum"][0]["tracker_buckets"])
    L.append("| stratum | " + " | ".join(keys) + " |")
    L.append("|---" * (len(keys) + 1) + "|")
    for r in rec["per_stratum"]:
        L.append(f"| {r['stratum']} | "
                 + " | ".join(f"{v:.5f}" for v in r["tracker_buckets"].values()) + " |")
    L.append("| **pooled** | "
             + " | ".join(f"{v:.5f}" for v in p["tracker_buckets"].values()) + " |")
    L.append("")
    a9 = rec["a9_premise_s4"]
    L.append(f"> ⚠️ A9 §3 states S4 as `(5,20,15)` = {a9['tokens']:,} tokens ⇒ shift "
             f"{a9['shift']:.4f}. **That geometry is not achievable** — 832x464 is not "
             f"VAE-legal (464/32 = 14.5), the delivered bucket is 832x448x33 (a pure 16-row "
             f"centre crop, no resampling), and the real grid is `(5,14,26)` = "
             f"{rec['per_stratum'][-1]['tokens']:,} tokens ⇒ shift "
             f"{rec['per_stratum'][-1]['shift']:.4f} "
             f"({a9['delta_shift_vs_artifact']:+.4f}). See DOSSIER §10.9.")
    L.append("")
    L.append(f"**Disclosed caveat.** {rec['report_caveat']}")
    L.append("")
    mc = rec.get("monte_carlo_check")
    prov = [f"Method: {rec['method']}"]
    if mc:
        prov.append(f"Validated against the trainer's own `ShiftedLogitNormalTimestepSampler` "
                    f"by Monte Carlo: worst sup|F_emp − F_analytic| = "
                    f"{mc['worst_ks']:.5f} at "
                    f"{next(iter(mc.values()))['n_draws']:,} draws per stratum, seed 42.")
    L.append("*" + " ".join(prov) + "*")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(LAB / "misc/ctt_v2_final/artefacts/sigma"),
                    help="archive directory for SIGMA_SCHEDULE.json / .txt")
    ap.add_argument("--mc", type=int, default=0,
                    help="Monte-Carlo self-check draws against the trainer's own sampler")
    ap.add_argument("--no-trainer-check", action="store_true")
    args = ap.parse_args()

    rec = build()
    laws = rec.pop("laws")

    if not args.no_trainer_check:
        rec["trainer_verification"] = read_shift_law_from_trainer()
        if not rec["trainer_verification"]["agrees_with_module"]:
            print("[sigma] FAIL — the module's shift law disagrees with the trainer source:")
            for d in rec["trainer_verification"]["disagreements"]:
                print(f"        - {d}")
            return 1
        print("[sigma] shift law + sampler defaults VERIFIED against the trainer source")

    if args.mc:
        print(f"[sigma] Monte-Carlo self-check, {args.mc:,} draws per stratum "
              f"from the trainer's own sampler ...")
        rec["monte_carlo_check"] = mc_check(laws, args.mc)
        worst = max(v["ks_sup_deviation"] for v in rec["monte_carlo_check"].values())
        print(f"[sigma] worst sup|F_emp - F_analytic| over all strata = {worst:.5f}")
        rec["monte_carlo_check"]["worst_ks"] = worst

    table = fmt_table(rec)
    print()
    print(table)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SIGMA_SCHEDULE.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (out / "SIGMA_SCHEDULE.txt").write_text(table + "\n")
    (out / "SIGMA_SCHEDULE.md").write_text(fmt_markdown(rec) + "\n")
    print(f"\n[sigma] archived -> {out}/")
    print("[sigma]   SIGMA_SCHEDULE.json  machine-readable, full CDF/quantile record")
    print("[sigma]   SIGMA_SCHEDULE.txt   the console table")
    print("[sigma]   SIGMA_SCHEDULE.md    paste-as-is block for DATASET.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
