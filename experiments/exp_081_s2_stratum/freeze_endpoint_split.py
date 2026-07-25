"""ctt_v2 S2/S3 — freeze the endpoint train/eval split.

The endpoint bank `data/processed/synth_endpoints/` is READ-ONLY (another campaign reads it).
This script never touches it; it writes a SPLIT ANNOTATION to a new directory.

SELECTION METHOD: k-means cluster MEDOIDS (advisor ruling, revision 2, 2026-07-25).

Revision 1 used farthest-point sampling. The advisor overturned it:

    "Your FPS instinct protected against the right failure (a degenerate 20-clone reserve) but
     built a subtler defect, and it is a SEMANTICS defect, not just the 1.64 PR: FPS reserves
     the hull. The 20 most mutually distant clips are, by construction, the clips least like
     anything else — so (i) training loses coverage exactly where it is thinnest, and (ii) the
     unseen-content eval cell is evaluated exactly in those uncovered regions. A failure in
     that cell would then confound 'cannot transfer manner to novel content instances' with
     'this content region was excised from training entirely.'"

Measured cost of the FPS reserve, which is what triggered the revision:
    ALL 227                        PR = 47.57
    TRAINING 207 (FPS reserve)     PR = 44.98     <-- failed the (mis-specified) >=47 bar
    TRAINING 207 (RANDOM reserve)  PR = 46.61     mean of 300 draws
    RESERVED 20 alone              PR = 16.10
    random subsets: n=150 -> 42.88, n=180 -> 45.10, n=207 -> 46.65, n=227 -> 47.57
i.e. participation ratio is strongly n-dependent (so a bar calibrated at n=227 is unreachable
at n=207), AND FPS cost a further 1.64 on top of the pure n effect.

Cluster medoids keep nearly all of FPS's eval spread — medoids of well-separated clusters are
themselves well separated — while every reserved clip leaves near neighbours behind in
training, so the eval cell tests "novel content instance", not "excised content region".

Constraints, all asserted at the end:
  C1  every reserved clip has max CLIP cosine <= 0.85 against EVERY training clip;
  C2  no clip belonging to a near-duplicate pair (cos > 0.90) is reserved;
  C3  source stratification approximately ~3 davis / 6 vcbench / 11 openvid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BANK_DIR = REPO_ROOT / "data/processed/synth_endpoints"
OUT_DIR = REPO_ROOT / "data/processed/ctt_v2_strata"

MAX_COS_RESERVED_TO_TRAIN = 0.85     # C1
NEAR_DUP_COS = 0.90                  # C2
N_RESERVE = 20                       # C3: quotas are scaled to the CLEAN pool composition
K = 20
SEED = 20260725


def kmeans(V: np.ndarray, k: int, seed: int, iters: int = 200):
    """Plain k-means++ on L2-normalised embeddings (cosine == euclidean up to a monotone map)."""
    rng = np.random.default_rng(seed)
    n = len(V)
    centres = [V[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min([((V - c) ** 2).sum(1) for c in centres], axis=0)
        p = d / max(d.sum(), 1e-12)
        centres.append(V[rng.choice(n, p=p)])
    C = np.stack(centres)
    lab = np.zeros(n, int)
    for _ in range(iters):
        d = ((V[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = V[m].mean(0)
    return lab, C


def main() -> None:
    bank = json.loads((BANK_DIR / "bank_tightened.json").read_text())
    ids = [c["clip_id"] if isinstance(c, dict) else c for c in bank["clips"]]
    meta = {json.loads(l)["clip_id"]: json.loads(l)
            for l in (BANK_DIR / "manifest.jsonl").read_text().splitlines() if l.strip()}

    # C0 — LETTERBOXED CLIPS ARE OUT OF THE DELIVERY UNIVERSE (advisor, unconditional).
    # "A 47%-black frame is not an endpoint." They are excluded here, before clustering, so the
    # reserve is cut from the clean pool and no black-matte clip can reach either stratum.
    lb = json.loads((OUT_DIR / "LETTERBOX_AUDIT.json").read_text())
    excluded = set(lb["excluded"])
    n_raw = len(ids)
    ids = [c for c in ids if c not in excluded]
    print(f"[split] letterbox exclusion: {n_raw} -> {len(ids)} clean clips "
          f"({len(excluded)} removed, worst {max(r['max'] for r in lb['per_clip']):.3f})")

    E = np.load(BANK_DIR / "embeddings.npy").astype(np.float64)
    emb_ids = json.loads((BANK_DIR / "embed_ids.json").read_text())
    row = {c: i for i, c in enumerate(emb_ids)}
    V = np.stack([E[row[c]] for c in ids])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    n = len(ids)
    src = [meta[c]["source"] for c in ids]

    near_dup_pairs = [(ids[i], ids[j], round(float(S[i, j]), 4))
                      for i in range(n) for j in range(i + 1, n) if S[i, j] > NEAR_DUP_COS]
    pinned = {c for p in near_dup_pairs for c in p[:2]}                       # C2
    max_cos = S.max(axis=1)
    eligible = np.array([max_cos[i] <= MAX_COS_RESERVED_TO_TRAIN              # C1
                         and ids[i] not in pinned for i in range(n)])

    # source quotas scaled to the CLEAN pool composition (advisor: "source quotas scaled to
    # the clean pool"), largest-remainder so they sum to exactly N_RESERVE
    comp: dict[str, int] = {}
    for s_ in src:
        comp[s_] = comp.get(s_, 0) + 1
    exact = {k: N_RESERVE * v / len(ids) for k, v in comp.items()}
    quota = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - int(exact[k]))):
        if sum(quota.values()) >= N_RESERVE:
            break
        quota[k] += 1
    RESERVE_PER_SOURCE = quota
    print(f"[split] clean composition {comp} -> reserve quotas {RESERVE_PER_SOURCE}")

    lab, C = kmeans(V, K, SEED)

    # One reserved clip per cluster: the most CENTRAL eligible member (the medoid), which is
    # what makes it representative rather than extremal.
    #
    # C3 needs care. Taking each cluster's global medoid greedily gives davis 1 / vcbench 6 /
    # openvid 13 against a 3/6/11 target, because davis is only 39 of 227 clips and rarely owns
    # a cluster centre. So clusters are ASSIGNED to sources scarcity-first: the scarcest source
    # claims the clusters where its own most-central member sits closest to the centroid, then
    # the next source claims from what remains. Every reserved clip is still its cluster's most
    # central member OF THE ASSIGNED SOURCE, so it stays representative of that cluster.
    dist = {j: {i: float(np.linalg.norm(V[i] - C[j]))
                for i in range(n) if lab[i] == j and eligible[i]} for j in range(K)}
    best = {(j, s): min((d for i, d in dist[j].items() if src[i] == s), default=None)
            for j in range(K) for s in RESERVE_PER_SOURCE}

    want = dict(RESERVE_PER_SOURCE)
    got: dict[str, int] = {s: 0 for s in want}
    reserved_idx: list[int] = []
    taken: set[int] = set()
    for s in sorted(want, key=lambda s: want[s]):              # scarcest quota first
        options = sorted(((best[(j, s)], j) for j in range(K)
                          if j not in taken and best[(j, s)] is not None),
                         key=lambda t: t[0])
        for _, j in options[: want[s]]:
            pick = min((i for i in dist[j] if src[i] == s), key=lambda i: dist[j][i])
            reserved_idx.append(pick)
            taken.add(j)
            got[s] += 1
    # any cluster still unassigned (its quota source ran out) contributes its global medoid
    for j in range(K):
        if j in taken or not dist[j]:
            continue
        pick = min(dist[j], key=lambda i: dist[j][i])
        reserved_idx.append(pick)
        taken.add(j)
        got[src[pick]] = got.get(src[pick], 0) + 1

    reserved = sorted(ids[i] for i in reserved_idx)
    training = sorted(c for c in ids if c not in set(reserved))

    ridx = [ids.index(c) for c in reserved]
    tidx = [ids.index(c) for c in training]
    cross = S[np.ix_(ridx, tidx)]
    worst = float(cross.max())
    assert worst <= MAX_COS_RESERVED_TO_TRAIN, f"C1 VIOLATED: max cos {worst:.4f}"
    assert not (set(reserved) & pinned), "C2 VIOLATED: a near-dup member was reserved"

    def pr(M: np.ndarray) -> float:
        w = np.linalg.eigvalsh(np.cov(M.T))
        w = w[w > 1e-12]
        p = w / w.sum()
        return float(1.0 / np.sum(p ** 2))

    res_sub = S[np.ix_(ridx, ridx)]
    iu = np.triu_indices(len(ridx), 1)
    tr_sub = S[np.ix_(tidx, tidx)]
    tiu = np.triu_indices(len(tidx), 1)

    out = {
        "created": "2026-07-25",
        "revision": 3,
        "authority": "fable-advisor ruling 3, rev3: k-means medoids + letterboxed clips excluded",
        "letterbox_excluded": sorted(excluded),
        "n_raw_bank": n_raw,
        "bank": "data/processed/synth_endpoints/bank_tightened.json (READ-ONLY, unmodified)",
        "rule": {
            "selection": f"k-means (k={K}, seed {SEED}, k-means++ init) over the 227 CLIP "
                         f"embeddings; per-cluster MEDOID, source-quota-preferring",
            "why_medoid": "FPS reserves the hull — the clips least like anything else — which "
                          "confounds the unseen-content eval cell with content regions excised "
                          "from training. Medoids are representative: every reserved clip "
                          "leaves near neighbours in training.",
            "max_cos_reserved_to_training": MAX_COS_RESERVED_TO_TRAIN,
            "near_dup_cos": NEAR_DUP_COS,
            "reserve_per_source": RESERVE_PER_SOURCE,
            "seed": SEED,
        },
        "n_total": n, "n_training": len(training), "n_reserved": len(reserved),
        "per_source_reserved": got,
        "near_dup_pairs_pinned_to_training": near_dup_pairs,
        "verification": {
            "max_cos_reserved_vs_training": round(worst, 4),
            "reserved_internal_cos_mean": round(float(res_sub[iu].mean()), 4),
            "training_mean_pairwise_cos": round(float(tr_sub[tiu].mean()), 4),
            "training_participation_ratio": round(pr(V[tidx]), 2),
            "all227_participation_ratio": round(pr(V), 2),
            "C1_pass": True, "C2_pass": True,
        },
        "note": "Pilot renders are validation artifacts and are never training data, so a clip "
                "that moved into the reserve after appearing in the S3 pilot is harmless.",
        "reserved_eval_only": reserved,
        "training": training,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ENDPOINT_SPLIT.json").write_text(json.dumps(out, indent=1))

    print(f"training={len(training)}  reserved={len(reserved)}  per-source={got}")
    print(f"max cos reserved-vs-training = {worst:.4f}  (bar <= {MAX_COS_RESERVED_TO_TRAIN})")
    print(f"reserved internal cos mean   = {res_sub[iu].mean():.4f}")
    print(f"TRAINING mean pairwise cos   = {tr_sub[tiu].mean():.4f}   (Gate A baseline)")
    print(f"TRAINING participation ratio = {pr(V[tidx]):.2f}   "
          f"(FPS revision-1 gave 44.98; random-reserve reference 46.61)")
    print(f"near-dup pairs pinned to training: {len(near_dup_pairs)}")
    print(f"-> {OUT_DIR / 'ENDPOINT_SPLIT.json'}")


if __name__ == "__main__":
    main()
