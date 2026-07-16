"""Apples-to-apples exam harness for the exploratory metric search.

This is NOT the closed E1' workbench (that program is dead and adjudicated). This
is a fresh exploratory search for an appearance metric that GENUINELY beats the
incumbent m1a__v3_sided on the frozen exam — same corpus, same 223-row order, same
frozen kernel (report.retrieval_eval + certify.diagnostics), same hubness gate,
same misretrieved convention. The metric CONSTRUCTION is open; the JUDGE is fixed.

  incumbent m1a__v3_sided:  acc 0.672646 · d 1.522006 · cov 1.0 · mis 73/223
  m1a = 1 - set_similarity(core_i, core_j),
        set_similarity = symmetric mean-of-max cosine over L2-normed DINO CLS
        embeddings of each clip's SIDED core frames (certify.exam +
        appearance.set_similarity, both deployed).

Discipline (no cheating):
  * every candidate is scored by exam.evaluate — the SAME function that judged the
    incumbent — over the SAME keys/labels/gates/corpus_facts;
  * verify_m1a() rebuilds the incumbent from the warm bundles and asserts a bitwise
    match to the frozen npz AND the pinned exam numbers, before any candidate runs;
  * coverage is reported next to accuracy (a win on shrunken support is not a win);
  * no label leakage, no per-class NaN-ing to inflate coverage-adjusted accuracy,
    no test-set threshold tuning that would not generalize.

Feature cache: the 223 warm bundles are loaded once and the per-clip substrate
(full per-frame CLS feats, the three core masks, sidedness, endpoint anchors,
profile window sizes) is persisted to $WB_CACHE/search/substrate.npz, so a metric
experiment is pure numpy over cached arrays and never re-reads the corpus.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from .. import exam as wb_exam           # workbench exam driver (evaluate/summary_line)
from .. import paths
from ...appearance import set_similarity
from ...certify import exam as cert_exam
from ...s_structure import core_mask_v3

SUBSTRATE = paths.WB_CACHE / "search" / "substrate.npz"
STEP0 = paths.WB_OUT / "step0" / "baselines.json"


# --- corpus-level facts (loaded once) ----------------------------------------

def load_context() -> dict:
    """keys / labels / sidedness / gates / corpus_facts / incumbent — the fixed
    frame every candidate is scored inside."""
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    sidedness = paths.sidedness_of(corpus, keys)
    gates = paths.load_gates(require_frozen=True)
    step0 = json.loads(STEP0.read_text())
    return {
        "corpus": corpus,
        "keys": keys,
        "labels": labels,
        "sidedness": sidedness,
        "gates": gates,
        "corpus_facts": step0["corpus_facts"],
        "incumbent": step0["regenerated"]["m1a__v3_sided"],
        "npz_frozen": paths.NPZ,
    }


# --- per-clip substrate cache ------------------------------------------------

def _build_substrate(ctx: dict) -> dict:
    """Load the 223 warm bundles once; extract the per-clip substrate the search
    needs. Zero GPU, zero writes to the shared cache (ReadOnlyExtractor)."""
    from .. import bundles as wb_bundles

    keys, sidedness = ctx["keys"], ctx["sidedness"]
    t0 = time.time()
    bs = wb_bundles.load_corpus_bundles(keys)
    print(f"[substrate] {len(bs)} warm bundles in {time.time()-t0:.1f}s")

    feats, mask_sided, mask_two, mask_all = [], [], [], []
    n_prefix, n_suffix, n_frames = [], [], []
    for b, s in zip(bs, sidedness):
        f = np.asarray(b["feats"], dtype=np.float32)   # [T,768] L2-normed CLS
        feats.append(f)
        mask_sided.append(np.asarray(core_mask_v3(b["profile"], s)[0], dtype=bool))
        mask_two.append(np.asarray(core_mask_v3(b["profile"], "twosided")[0], dtype=bool))
        T = len(f)
        pfx, sfx = b["profile"]["n_prefix"], b["profile"]["n_suffix"]
        ma = np.zeros(T, dtype=bool)
        ma[pfx:T - sfx] = True
        mask_all.append(ma)
        n_prefix.append(pfx)
        n_suffix.append(sfx)
        n_frames.append(T)

    Ts = set(n_frames)
    assert len(Ts) == 1, f"non-uniform frame counts {sorted(Ts)} — use object arrays"
    # uniform T -> dense regular arrays (vectorizable, no object-array indexing traps)
    out = {
        "keys": np.array(keys),
        "feats": np.stack(feats).astype(np.float32),        # [223, T, 768]
        "mask_sided": np.stack(mask_sided),                 # [223, T] bool
        "mask_two": np.stack(mask_two),
        "mask_all": np.stack(mask_all),
        "n_prefix": np.array(n_prefix),
        "n_suffix": np.array(n_suffix),
        "sidedness": np.array(sidedness),
    }
    SUBSTRATE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SUBSTRATE, **out)
    print(f"[substrate] cached -> {SUBSTRATE}")
    return out


def load_substrate(ctx: dict, rebuild: bool = False) -> dict:
    if SUBSTRATE.exists() and not rebuild:
        z = np.load(SUBSTRATE, allow_pickle=True)
        return {k: z[k] for k in z.keys()}
    return _build_substrate(ctx)


def cores(sub: dict, which: str = "sided") -> list[np.ndarray]:
    """Per-clip core-frame CLS embeddings under a mask variant."""
    mk = {"sided": "mask_sided", "two": "mask_two", "all": "mask_all"}[which]
    return [f[m] for f, m in zip(sub["feats"], sub[mk])]


# --- scoring (the fixed judge) -----------------------------------------------

def run(ctx: dict, name: str, D: np.ndarray,
        reasons: list[str | None] | None = None, stratum: str | None = None,
        quiet: bool = False) -> dict:
    """Score a candidate distance matrix by the frozen kernel — identical to how
    the incumbent was judged."""
    D = np.asarray(D, dtype=float)
    assert D.shape == (len(ctx["labels"]), len(ctx["labels"])), D.shape
    r = wb_exam.evaluate(name, D, list(ctx["keys"]), ctx["labels"], ctx["gates"],
                         ctx["corpus_facts"], reasons=reasons, stratum=stratum)
    if not quiet:
        print(wb_exam.summary_line(r))
    return r


def verdict(r: dict, ctx: dict) -> dict:
    """Head-to-head vs the pinned incumbent on the exam's own terms."""
    inc = ctx["incumbent"]
    beats_d = r["separation_cohens_d"] > inc["separation_cohens_d"]
    beats_mis = r["misretrieved"] < inc["misretrieved"]
    beats_acc = r["accuracy_1nn"] > inc["accuracy_1nn"]
    return {
        "beats_acc": bool(beats_acc),
        "beats_cohens_d": bool(beats_d),
        "beats_misretrieved": bool(beats_mis),
        "acc": r["accuracy_1nn"], "acc_inc": inc["accuracy_1nn"],
        "d": r["separation_cohens_d"], "d_inc": inc["separation_cohens_d"],
        "mis": r["misretrieved"], "mis_inc": inc["misretrieved"],
        "cov": r["coverage"], "hub_pass": r["hubness"]["pass"],
        # "beats the incumbent" the way the goal means it: strictly better on the
        # exam's headline accuracy, without shrinking coverage or failing hubness.
        "genuinely_beats": bool(beats_acc and r["coverage"] >= inc["coverage"]
                                and r["hubness"]["pass"]),
    }


# --- the base touch (run before trusting any candidate) ----------------------

def verify_m1a(ctx: dict, sub: dict) -> dict:
    """Rebuild the incumbent from the warm bundles two ways and prove both the
    matrix AND the exam numbers reproduce the pinned baseline bit-for-bit."""
    keys, labels, sidedness = ctx["keys"], ctx["labels"], ctx["sidedness"]

    # (1) rebuild from cached core features via the deployed set_similarity
    C = cores(sub, "sided")
    n = len(C)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - set_similarity(C[i], C[j])

    frozen = np.load(ctx["npz_frozen"])["m1a__v3_sided"]
    delta = float(np.abs(D - frozen).max())

    r = run(ctx, "m1a__v3_sided (rebuilt)", D, quiet=True)
    inc = ctx["incumbent"]
    checks = {
        "matrix_max_abs_delta_vs_frozen": delta,
        "matrix_bitexact": delta == 0.0,
        "acc": (r["accuracy_1nn"], inc["accuracy_1nn"],
                r["accuracy_1nn"] == inc["accuracy_1nn"]),
        "cohens_d": (r["separation_cohens_d"], inc["separation_cohens_d"],
                     r["separation_cohens_d"] == inc["separation_cohens_d"]),
        "coverage": (r["coverage"], inc["coverage"], r["coverage"] == inc["coverage"]),
        "misretrieved": (r["misretrieved"], inc["misretrieved"],
                         r["misretrieved"] == inc["misretrieved"]),
    }
    print(wb_exam.summary_line(r))
    print(f"[verify] matrix max|Δ| vs frozen npz = {delta!r}  bitexact={delta==0.0}")
    return {"D_m1a": D, "checks": checks, "result": r}


if __name__ == "__main__":
    ctx = load_context()
    sub = load_substrate(ctx)
    v = verify_m1a(ctx, sub)
    ok = (v["checks"]["matrix_bitexact"]
          and v["checks"]["acc"][2] and v["checks"]["cohens_d"][2]
          and v["checks"]["coverage"][2] and v["checks"]["misretrieved"][2])
    print("\nBASE TOUCH:", "PASS — harness reproduces the incumbent exactly"
          if ok else "FAIL — do not trust any candidate until this is green")
    raise SystemExit(0 if ok else 1)
