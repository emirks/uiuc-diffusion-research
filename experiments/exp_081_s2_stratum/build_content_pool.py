"""ctt_v2 S2/S3 — assemble the CONTENT POOL both strata sample their content pairs from.

Resolves the round-1 bank (READ-ONLY) and the additive v2 expansion bank into one manifest with
absolute mp4 paths and one aligned embedding matrix, then applies the advisor's bank-level
acceptance gates and, if needed, the deterministic trim.

GATES (advisor ruling, revision 2 — the original "participation ratio >= 47" was withdrawn as
mis-specified, because PR is strongly n-dependent on this bank and 47 was calibrated at n=227
while the training pool is 207, making it unreachable by construction):

  GATE A  homogenisation, global : mean pairwise CLIP cos over the training pool <= 0.52.
          "+0.026 headroom over the 0.4943 baseline is sized so that a new-clip cohort with
           internal cos ~0.6+ at expected cohort size ~235/442 trips it, while benign drift
           does not."
  GATE B  spread, size-matched   : mean PR over >=300 random subsets of the training pool, at
          the CLEAN pool's own size, >= (clean-pool PR - 1.15). Size-matched, therefore
          n-invariant. "A mildly shifted mixture may dip, a genuinely homogenized cohort
          craters far below — clones fill matched-n subsets and collapse PR."

The 20 LETTERBOXED bank clips are excluded upstream in ENDPOINT_SPLIT.json, so the clean pool
is 207 -> 187 training + 20 reserved, and Gate B re-derives its match size and bar from that
file automatically.

TRIM (advisor Q3): S2 launches at window close unconditionally, on the largest gate-passing
pool. If either gate fails, greedily remove the NEW clip (the clean v1 training clips are
untouchable) with the highest mean cosine to the rest, recompute, repeat. The v1 floor passes
by construction, so this always terminates in a launchable pool. Every removal is logged.
"If trimming eats most of the expansion, that IS the finding."

    python build_content_pool.py                # v1 only (the 207-endpoint floor)
    python build_content_pool.py --with-v2      # v1 + expansion, gates + trim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
V1 = REPO_ROOT / "data/processed/synth_endpoints"
V2 = REPO_ROOT / "data/processed/ctt_v2_strata/endpoints_v2"
OUT_DIR = REPO_ROOT / "data/processed/ctt_v2_strata"

GATE_A_MAX_MEAN_COS = 0.52          # absolute, unchanged by the letterbox exclusion
# Gate B is size-consistent by construction: the match size is the CLEAN training pool size and
# the bar is that pool's own PR minus a fixed tolerance. Advisor: "Gate B becomes: mean PR over
# >=300 random n=<clean> subsets of the expanded pool >= PR(clean training pool) - 1.15 — same
# statistic, same tolerance, size-consistent." Both are read from ENDPOINT_SPLIT.json so they
# re-derive automatically if the clean pool ever changes again.
GATE_B_TOLERANCE = 1.15
GATE_B_DRAWS = 300
GATE_SEED = 20260725


def participation_ratio(V: np.ndarray) -> float:
    w = np.linalg.eigvalsh(np.cov(V.T))
    w = w[w > 1e-12]
    p = w / w.sum()
    return float(1.0 / np.sum(p ** 2))


def matched_pr(V: np.ndarray, n: int, draws: int, seed: int) -> float:
    """Mean PR over random size-n subsets — n-invariant, so it measures spread not sample size."""
    if len(V) <= n:
        return participation_ratio(V)
    rng = np.random.default_rng(seed)
    return float(np.mean([participation_ratio(V[rng.choice(len(V), n, replace=False)])
                          for _ in range(draws)]))


def mean_cos(V: np.ndarray) -> float:
    S = V @ V.T
    iu = np.triu_indices(len(V), 1)
    return float(S[iu].mean())


def evaluate(V: np.ndarray, match_n: int, bar_b: float) -> dict:
    a = mean_cos(V)
    b = matched_pr(V, match_n, GATE_B_DRAWS, GATE_SEED)
    return {"gate_a_mean_cos": round(a, 4), "gate_a_bar": GATE_A_MAX_MEAN_COS,
            "gate_a_pass": a <= GATE_A_MAX_MEAN_COS,
            "gate_b_matched_pr": round(b, 2), "gate_b_bar": round(bar_b, 2),
            "gate_b_match_n": match_n,
            "gate_b_pass": b >= bar_b,
            "pass": a <= GATE_A_MAX_MEAN_COS and b >= bar_b,
            "participation_ratio_raw": round(participation_ratio(V), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-v2", action="store_true")
    args = ap.parse_args()

    split = json.loads((OUT_DIR / "ENDPOINT_SPLIT.json").read_text())
    v1_bank = {c["clip_id"]: c for c in
               json.loads((V1 / "bank_tightened.json").read_text())["clips"]}
    v1_E = np.load(V1 / "embeddings.npy").astype(np.float64)
    v1_ids = json.loads((V1 / "embed_ids.json").read_text())
    v1_row = {c: i for i, c in enumerate(v1_ids)}

    def v1_entry(cid: str, role: str) -> dict:
        return {"clip_id": cid, "role": role, "bank": "v1",
                "mp4": str((V1 / v1_bank[cid]["mp4"]).resolve()),
                "source": cid.split("_")[0]}

    MATCH_N = len(split["training"])
    BAR_B = split["verification"]["training_participation_ratio"] - GATE_B_TOLERANCE
    print(f"[pool] Gate B calibrated to the clean pool: match n={MATCH_N}, "
          f"bar = {split['verification']['training_participation_ratio']:.2f} - "
          f"{GATE_B_TOLERANCE} = {BAR_B:.2f}")

    training = [v1_entry(c, "training") for c in split["training"]]
    reserved = [v1_entry(c, "reserved") for c in split["reserved_eval_only"]]
    emb = {c: v1_E[v1_row[c]] for c in split["training"] + split["reserved_eval_only"]}
    n_v1 = len(training)

    new_ids: list[str] = []
    if args.with_v2:
        bank2_path = V2 / "bank_tightened_v2.json"
        if not bank2_path.exists():
            raise SystemExit(f"[pool] --with-v2 but {bank2_path} is missing — run the expansion "
                             f"(collect -> detect -> build -> tighten --stage finalize) first.")
        b2 = json.loads(bank2_path.read_text())
        E2 = np.load(V2 / "embeddings.npy").astype(np.float64)
        ids2 = json.loads((V2 / "embed_ids.json").read_text())
        row2 = {c: i for i, c in enumerate(ids2)}
        for c in b2["clips"]:
            cid = c["clip_id"]
            assert cid not in emb, f"v2 clip id collides with the round-1 bank: {cid}"
            training.append({"clip_id": cid, "role": "training", "bank": "v2",
                             "mp4": str((V2 / c["mp4"]).resolve()), "source": "vcbench_v2"})
            emb[cid] = E2[row2[cid]]
            new_ids.append(cid)
        print(f"[pool] expansion: +{len(new_ids)} candidate training endpoints")

    for e in training + reserved:
        assert Path(e["mp4"]).exists(), f"missing clip file: {e['mp4']}"

    def emb_matrix(entries):
        M = np.stack([emb[e["clip_id"]] for e in entries])
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    before = evaluate(emb_matrix(training), MATCH_N, BAR_B)
    print(f"[pool] gates BEFORE trim: A mean_cos {before['gate_a_mean_cos']} "
          f"(<= {GATE_A_MAX_MEAN_COS}) {'PASS' if before['gate_a_pass'] else 'FAIL'} | "
          f"B matched-PR {before['gate_b_matched_pr']} (>= {BAR_B:.2f}) "
          f"{'PASS' if before['gate_b_pass'] else 'FAIL'}")

    # ---- deterministic trim: only NEW clips are removable ---------------------------------
    trim_log: list[dict] = []
    cur = evaluate(emb_matrix(training), MATCH_N, BAR_B) if training else before
    while not cur["pass"] and new_ids:
        M = emb_matrix(training)
        S = M @ M.T
        np.fill_diagonal(S, 0.0)
        mean_to_rest = S.sum(1) / (len(M) - 1)
        removable = [(mean_to_rest[i], i) for i, e in enumerate(training) if e["bank"] == "v2"]
        if not removable:
            break
        _, worst = max(removable)
        drop = training[worst]
        trim_log.append({"clip_id": drop["clip_id"],
                         "mean_cos_to_pool": round(float(mean_to_rest[worst]), 4),
                         "gate_a": cur["gate_a_mean_cos"], "gate_b": cur["gate_b_matched_pr"]})
        training.pop(worst)
        new_ids.remove(drop["clip_id"])
        cur = evaluate(emb_matrix(training), MATCH_N, BAR_B)
        print(f"[pool]   trim {len(trim_log):3d}: dropped {drop['clip_id']:<44} "
              f"A {cur['gate_a_mean_cos']:.4f} B {cur['gate_b_matched_pr']:.2f} "
              f"({len(new_ids)} expansion clips left)", flush=True)
    after = cur
    if trim_log:
        print(f"[pool] TRIM: removed {len(trim_log)} expansion clips to satisfy the gates")

    ids = [e["clip_id"] for e in training + reserved]
    V = np.stack([emb[c] for c in ids])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.save(OUT_DIR / "content_pool_emb.npy", V.astype(np.float32))

    # ---- non-gating diagnostics the advisor asked to see -----------------------------------
    diag: dict = {"n_training_v1": n_v1, "n_training_v2": len(new_ids)}
    if new_ids:
        Mn = np.stack([emb[c] for c in new_ids])
        Mn /= np.linalg.norm(Mn, axis=1, keepdims=True)
        diag["new_cohort_internal_mean_cos"] = round(mean_cos(Mn), 4) if len(Mn) > 1 else None
    vcb = [c for c in split["training"] + split["reserved_eval_only"] if c.startswith("vcbench_")]
    if len(vcb) > 1:
        Mv = np.stack([emb[c] for c in vcb])
        Mv /= np.linalg.norm(Mv, axis=1, keepdims=True)
        diag["vcbench63_internal_mean_cos_reference"] = round(mean_cos(Mv), 4)
    comp: dict = {}
    for e in training:
        comp[e["source"]] = comp.get(e["source"], 0) + 1
    diag["training_per_source"] = comp

    pool = {
        "created": "2026-07-25", "with_v2": bool(args.with_v2),
        "authority": "fable-advisor: 207/20 split (medoid revision) + Gates A/B + trim ruling",
        "n_training": len(training), "n_reserved": len(reserved),
        "embeddings": "data/processed/ctt_v2_strata/content_pool_emb.npy "
                      "(L2-normalised, row-aligned to `ids`; v1 and v2 share CLIP weights)",
        "ids": ids,
        "gates": {"before_trim": before, "after_trim": after,
                  "definition": {
                      "A": f"mean pairwise CLIP cos over training <= {GATE_A_MAX_MEAN_COS}",
                      "B": f"mean PR over {GATE_B_DRAWS} random n={MATCH_N} subsets "
                           f">= {BAR_B:.2f} (= clean-pool PR - {GATE_B_TOLERANCE}; "
                           f"size-matched => n-invariant)"}},
        "letterbox_excluded": split.get("letterbox_excluded", []),
        "trim_log": trim_log,
        "diagnostics_non_gating": diag,
        "near_dup_pairs_pinned_to_training": split["near_dup_pairs_pinned_to_training"],
        "participation_ratio_pass": after["pass"],     # name kept: planners assert on it
        "training": training, "reserved": reserved,
    }
    (OUT_DIR / "CONTENT_POOL.json").write_text(json.dumps(pool, indent=1))

    print(f"[pool] training {len(training)} (v1 {n_v1} + v2 {len(new_ids)}) | "
          f"reserved {len(reserved)}")
    print(f"[pool] gates AFTER  trim: A {after['gate_a_mean_cos']} "
          f"{'PASS' if after['gate_a_pass'] else 'FAIL'} | B {after['gate_b_matched_pr']} "
          f"{'PASS' if after['gate_b_pass'] else 'FAIL'}  => "
          f"{'LAUNCHABLE' if after['pass'] else 'BLOCKED'}")
    print(f"[pool] diagnostics: {json.dumps(diag)}")
    print(f"[pool] -> {OUT_DIR / 'CONTENT_POOL.json'}")
    if not after["pass"]:
        raise SystemExit("[pool] BLOCKED: gates still fail after trimming every expansion clip. "
                         "The 207 floor passes by construction, so this indicates a bug.")


if __name__ == "__main__":
    main()
