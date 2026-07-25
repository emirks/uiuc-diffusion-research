"""ctt_v2 S2/S3 — freeze the endpoint train/eval split (advisor ruling 3, 2026-07-25).

The endpoint bank `data/processed/synth_endpoints/` is READ-ONLY (another campaign reads it).
This script never touches it; it writes a SPLIT ANNOTATION to a new directory that both S2 and
S3 read.

Advisor ruling being implemented, verbatim:

    "reserve ~20 of the 227 as eval-only endpoints (stratified by source: ~3 davis / ~6 vcbench
     / ~11 openvid), never used in any S2/S3 training grid. Ensure no reserved clip has
     cos > 0.85 to the training side, and both members of every near-dup pair stay on the
     training side. This is a manifest-level annotation in OUR delivery, not a bank mutation.
     Without it, every 'unseen content' eval cell would require corpus clips (illegal) or new
     collection (expensive)."

Two hard constraints, checked and asserted at the end:
  C1  every reserved clip has max CLIP cosine <= 0.85 against EVERY training clip;
  C2  no clip belonging to a near-duplicate pair (cos > 0.90) is reserved.

Within those, reserved clips are chosen by farthest-point sampling per source so the eval-only
content set is as spread as the bank allows (a reserved set of 20 near-identical clips would
technically satisfy C1/C2 while making the "unseen content" eval cell meaningless).
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
RESERVE_PER_SOURCE = {"davis": 3, "vcbench": 6, "openvid": 11}   # advisor's stratification
SEED = 20260725


def farthest_point(order_pool: list[int], S: np.ndarray, k: int, seed: int) -> list[int]:
    """Pick k indices from order_pool maximising the minimum pairwise distance (1 - cos)."""
    if k <= 0 or not order_pool:
        return []
    rng = np.random.default_rng(seed)
    pool = list(order_pool)
    # deterministic, diversity-seeking start: the pool member least similar to the pool mean
    sub = S[np.ix_(pool, pool)]
    start = pool[int(np.argmin(sub.mean(axis=1)))]
    picked = [start]
    while len(picked) < min(k, len(pool)):
        rest = [i for i in pool if i not in picked]
        if not rest:
            break
        # maximise distance to the nearest already-picked = minimise max cosine to picked
        maxcos = S[np.ix_(rest, picked)].max(axis=1)
        best = int(np.argmin(maxcos))
        ties = [r for r, m in zip(rest, maxcos) if abs(m - maxcos[best]) < 1e-9]
        picked.append(int(rng.choice(ties)) if len(ties) > 1 else rest[best])
    return picked


def main() -> None:
    bank = json.loads((BANK_DIR / "bank_tightened.json").read_text())
    clips = bank["clips"]
    ids = [c["clip_id"] if isinstance(c, dict) else c for c in clips]
    meta = {json.loads(l)["clip_id"]: json.loads(l)
            for l in (BANK_DIR / "manifest.jsonl").read_text().splitlines() if l.strip()}

    E = np.load(BANK_DIR / "embeddings.npy").astype(np.float64)
    emb_ids = json.loads((BANK_DIR / "embed_ids.json").read_text())
    row = {c: i for i, c in enumerate(emb_ids)}
    V = np.stack([E[row[c]] for c in ids])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    np.fill_diagonal(S, -1.0)                     # self-similarity must never win a max()

    n = len(ids)
    src = [meta[c]["source"] for c in ids]

    # ---- C2: any clip in a near-dup pair is pinned to the TRAINING side -------------------
    near_dup_pairs = [(ids[i], ids[j], round(float(S[i, j]), 4))
                      for i in range(n) for j in range(i + 1, n) if S[i, j] > NEAR_DUP_COS]
    pinned = {c for p in near_dup_pairs for c in p[:2]}

    # ---- C1: a clip is reservable only if it is <= 0.85 to EVERY other clip --------------
    # (conservative: "every other clip" is a superset of the eventual training side, so a clip
    #  passing this can never violate C1 no matter which other clips end up reserved)
    max_cos = S.max(axis=1)
    eligible = [i for i in range(n) if max_cos[i] <= MAX_COS_RESERVED_TO_TRAIN
                and ids[i] not in pinned]

    reserved_idx: list[int] = []
    per_source_report = {}
    for source, want in RESERVE_PER_SOURCE.items():
        pool = [i for i in eligible if src[i] == source]
        got = farthest_point(pool, S, want, SEED + hash(source) % 1000)
        reserved_idx += got
        per_source_report[source] = {"eligible": len(pool), "requested": want, "reserved": len(got)}

    reserved = sorted(ids[i] for i in reserved_idx)
    training = sorted(c for c in ids if c not in set(reserved))

    # ---- verify both constraints on the FINAL split --------------------------------------
    ridx = [ids.index(c) for c in reserved]
    tidx = [ids.index(c) for c in training]
    cross = S[np.ix_(ridx, tidx)]
    worst = float(cross.max())
    worst_where = np.unravel_index(int(np.argmax(cross)), cross.shape)
    assert worst <= MAX_COS_RESERVED_TO_TRAIN, (
        f"C1 VIOLATED: reserved {reserved[worst_where[0]]} vs training "
        f"{training[worst_where[1]]} cos={worst:.4f} > {MAX_COS_RESERVED_TO_TRAIN}")
    assert not (set(reserved) & pinned), "C2 VIOLATED: a near-dup member was reserved"

    res_sub = S[np.ix_(ridx, ridx)]
    iu = np.triu_indices(len(ridx), 1)
    out = {
        "created": "2026-07-25",
        "authority": "fable-advisor ruling 3 (ctt_v2 S2/S3 campaign, round 1)",
        "bank": "data/processed/synth_endpoints/bank_tightened.json (READ-ONLY, unmodified)",
        "rule": {
            "max_cos_reserved_to_training": MAX_COS_RESERVED_TO_TRAIN,
            "near_dup_cos": NEAR_DUP_COS,
            "reserve_per_source": RESERVE_PER_SOURCE,
            "selection": "farthest-point sampling per source over C1/C2-eligible clips",
            "seed": SEED,
        },
        "n_total": n, "n_training": len(training), "n_reserved": len(reserved),
        "per_source": per_source_report,
        "near_dup_pairs_pinned_to_training": near_dup_pairs,
        "verification": {
            "max_cos_reserved_vs_training": round(worst, 4),
            "reserved_internal_cos_mean": round(float(res_sub[iu].mean()), 4),
            "reserved_internal_cos_max": round(float(res_sub[iu].max()), 4),
            "C1_pass": True, "C2_pass": True,
        },
        "reserved_eval_only": reserved,
        "training": training,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ENDPOINT_SPLIT.json").write_text(json.dumps(out, indent=1))

    print(f"training={len(training)}  reserved={len(reserved)}")
    print("per source:", json.dumps(per_source_report))
    print(f"max cos reserved-vs-training = {worst:.4f}  (bar <= {MAX_COS_RESERVED_TO_TRAIN})")
    print(f"reserved internal cos: mean {res_sub[iu].mean():.4f} max {res_sub[iu].max():.4f}")
    print(f"near-dup pairs pinned to training: {len(near_dup_pairs)}")
    print("reserved:", ", ".join(reserved))
    print(f"-> {OUT_DIR / 'ENDPOINT_SPLIT.json'}")


if __name__ == "__main__":
    main()
