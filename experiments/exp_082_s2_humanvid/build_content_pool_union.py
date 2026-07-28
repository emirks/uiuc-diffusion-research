"""ctt_v2 S2b — assemble the UNION CONTENT POOL: the synth endpoint bank + the HumanVid bank.

Same job as `build_content_pool.py` (the sibling copy in this directory, taken from
exp_081_s2_stratum), but the pool is the union of

  * the 311 SYNTH clips already resolved in data/processed/ctt_v2_strata/CONTENT_POOL.json
    (291 training + 20 reserved; v1 bank_tightened + the v2 expansion, post-trim), and
  * the 1,499 HumanVid clips in data/processed/humanvid_bank/

and every record carries `bank` in {"synth", "humanvid"} so the planner can enforce the
advisor's 25/50/25 pair quota and the "no bank-pure op" assert. Emits
CONTENT_POOL_union.json / content_pool_emb_union.npy.

GATE MACHINERY IS IMPORTED, NOT RE-IMPLEMENTED
----------------------------------------------
`participation_ratio`, `matched_pr`, `mean_cos`, `evaluate` and the four gate constants
(`GATE_A_MAX_MEAN_COS` = 0.52, `GATE_B_TOLERANCE` = 1.15, `GATE_B_DRAWS` = 300,
`GATE_SEED` = 20260725) are imported from the local copy of `build_content_pool.py`
(defined there at lines 47-89), so Gate A, Gate B, the tolerance, the draw count and the RNG
seed are literally the same code. Importing it is side-effect free: that module only builds
Path constants at import time and does all its work inside `main()`.

The BARS are the same numbers too, re-derived exactly as the original does, from
ENDPOINT_SPLIT.json:  MATCH_N = len(training) = 187 and
BAR_B = verification.training_participation_ratio - 1.15 = 43.97 - 1.15 = 42.82.

TRIM
----
Mirrors `build_content_pool.py:152-174`: greedily drop the removable clip with the highest
mean cosine to the rest, recompute, repeat. The role the v1 bank plays there — a protected
floor that passes the gates by construction — is played here by the 291 SYNTH training clips
(exp_081 shipped them at A 0.5008 / B 42.83, i.e. passing), so REMOVABLE == bank "humanvid",
exactly parallel to REMOVABLE == bank "v2" in the original. Three documented deviations, none
of which changes the removal order or the resulting pool:

  1. Gate B is SHORT-CIRCUITED while Gate A fails. `pass` is `A and B`, so the loop condition
     cannot depend on B while A fails, but the original still pays ~40 s of eigendecomposition
     per iteration for it. `_eval_fast` computes B only once A passes; trim-log rows recorded
     while A was failing carry `gate_b: null`.
  2. The similarity matrix is maintained incrementally (row sums over the alive set) rather
     than recomputed with `M @ M.T` each iteration. Same values, same argmax tie-break
     (highest mean-cos-to-rest; on an exact tie the later index, as `max()` on (value, index)
     tuples does in the original).
  3. An explicit safety valve stops the trim if training would fall to Gate B's match size.

RESERVED HOLDOUT
----------------
The 20 synth reserved clips stay reserved (inherited verbatim), plus `--reserve-humanvid`
(default 100) HumanVid clips, drawn deterministically from the clips NOT pinned by the
near-duplicate rule. Reserved clips are never trimmed and never enter the gates.

CROSS-BANK DUPLICATES (found here, flagged, NOT silently dropped)
----------------------------------------------------------------
The synth bank's `vcbench_*` clips are Pexels stock footage and HumanVid is Pexels stock
footage, so the two banks OVERLAP: five clips are the same Pexels asset under two clip_ids
(cos >= 0.995, identical Pexels numeric ids), plus one near-dup at 0.913. The near-dup rule
below pins every such pair to training and the planner's `banned` set already refuses to pair
them with each other, so nothing leaks into eval and no op can render a video against itself —
but the union still double-counts that content. Dropping one side is a design decision, so it
is REPORTED (`diagnostics_non_gating.cross_bank_duplicates`, and the pairs appear in
`near_dup_pairs_pinned_to_training`) and left to the advisor.

    python build_content_pool_union.py
    python build_content_pool_union.py --no-trim     # diagnostic: gates only, no removal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]            # the worktree root when run from the worktree
sys.path.insert(0, str(HERE))

from build_content_pool import (  # noqa: E402  — local copy, side-effect free at import
    GATE_A_MAX_MEAN_COS,
    GATE_B_DRAWS,
    GATE_B_TOLERANCE,
    GATE_SEED,
    evaluate,
    matched_pr,
    mean_cos,
    participation_ratio,
)

# Both source banks and the reference calibration live in the MAIN tree, READ-ONLY here.
MAIN_TREE = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
HV_BANK = MAIN_TREE / "data/processed/humanvid_bank"
OUT_DIR = REPO_ROOT / "data/processed/ctt_v2_strata"
REF_STRATA = [REPO_ROOT / "data/processed/ctt_v2_strata",          # worktree copy, if any
              MAIN_TREE / "data/processed/ctt_v2_strata"]          # else the main tree's

RESERVE_SEED = 20260727                # matches runtime.seed in this experiment's config_s2.yaml
PAIR_MAX_COS = 0.85                    # s2.pair_max_cos — used for the FEASIBILITY diagnostic


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_ref(name: str) -> Path:
    for d in REF_STRATA:
        if (d / name).exists():
            return d / name
    raise SystemExit(f"[pool] reference file not found in {[str(d) for d in REF_STRATA]}: {name}")


def cos_stats(V: np.ndarray, W: np.ndarray | None = None) -> dict:
    """Cosine distribution: within `V` (off-diagonal) or between `V` and `W`."""
    if W is None:
        S = V @ V.T
        c = S[np.triu_indices(len(V), 1)]
        n = int(len(V))
    else:
        c = (V @ W.T).ravel()
        n = int(len(V) * len(W))
    pct = np.percentile(c, [1, 5, 25, 50, 75, 95, 99])
    d = {
        "n_clips": int(len(V)) if W is None else [int(len(V)), int(len(W))],
        "n_pairs": int(len(c)),
        "mean_cos": round(float(c.mean()), 4),
        "std_cos": round(float(c.std()), 4),
        "min_cos": round(float(c.min()), 4),
        "max_cos": round(float(c.max()), 4),
        "pct_1_5_25_50_75_95_99": [round(float(x), 4) for x in pct],
        "frac_gt_0.80": round(float((c > 0.80).mean()), 6),
        "frac_gt_0.85": round(float((c > 0.85).mean()), 6),
        "frac_gt_0.90": round(float((c > 0.90).mean()), 6),
        # FEASIBILITY: how many pairs are actually available under s2.pair_max_cos
        "n_pairs_le_0.85": int((c <= PAIR_MAX_COS).sum()),
        "frac_pairs_le_0.85": round(float((c <= PAIR_MAX_COS).mean()), 6),
    }
    if W is None:
        mu = V.mean(0)
        Vc = V - mu
        Vc = Vc / np.maximum(np.linalg.norm(Vc, axis=1, keepdims=True), 1e-12)
        # common-mode diagnostic: how much of mean_cos is ONE shared direction?
        d["mean_vector_norm"] = round(float(np.linalg.norm(mu)), 4)
        d["mean_cos_after_centring"] = round(mean_cos(Vc), 4)
        d["participation_ratio_raw"] = round(participation_ratio(V), 2)
    return d


def _eval_fast(V: np.ndarray, match_n: int, bar_b: float) -> dict:
    """`evaluate()` with Gate B short-circuited while Gate A fails (see module docstring)."""
    a = mean_cos(V)
    if a > GATE_A_MAX_MEAN_COS:
        return {"gate_a_mean_cos": round(a, 4), "gate_a_bar": GATE_A_MAX_MEAN_COS,
                "gate_a_pass": False, "gate_b_matched_pr": None, "gate_b_bar": round(bar_b, 2),
                "gate_b_match_n": match_n, "gate_b_pass": None, "pass": False}
    return evaluate(V, match_n, bar_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-humanvid", type=int, default=100)
    ap.add_argument("--near-dup-cos", type=float, default=0.90)
    ap.add_argument("--no-trim", action="store_true",
                    help="diagnostic only: report the gates without removing anything")
    args = ap.parse_args()
    t_start = time.time()

    # ---- 1. bank A: the resolved synth pool (291 training + 20 reserved) -------------------
    syn_path = resolve_ref("CONTENT_POOL.json")
    syn = json.loads(syn_path.read_text())
    SE = np.load(resolve_ref("content_pool_emb.npy")).astype(np.float64)
    assert len(SE) == len(syn["ids"]) == syn["n_training"] + syn["n_reserved"]
    syn_emb = {c: SE[i] for i, c in enumerate(syn["ids"])}
    log(f"[pool] synth  bank: {syn['n_training']} training + {syn['n_reserved']} reserved "
        f"from {syn_path}")

    def syn_entry(e: dict) -> dict:
        return {"clip_id": e["clip_id"], "role": e["role"], "bank": "synth",
                "mp4": e["mp4"], "source": e["source"], "bank_v1v2": e["bank"]}

    # ---- 2. bank B: humanvid --------------------------------------------------------------
    recs = [json.loads(l) for l in (HV_BANK / "manifest.jsonl").read_text().splitlines()
            if l.strip()]
    HE = np.load(HV_BANK / "embeddings.npy").astype(np.float64)
    hv_ids_file = json.loads((HV_BANK / "embed_ids.json").read_text())
    assert len(recs) == len(HE) == len(hv_ids_file)
    for i, r in enumerate(recs):
        assert r["embed_row"] == i and hv_ids_file[i] == r["clip_id"], \
            f"humanvid embedding row misalignment at {i}: {r['clip_id']}"
    assert recs[0]["embed_model"] == "openai/clip-vit-base-patch32", "CLIP weights differ"
    hv_emb = {r["clip_id"]: HE[r["embed_row"]] for r in recs}
    hv_mp4 = {r["clip_id"]: str((HV_BANK / r["mp4"]).resolve()) for r in recs}
    log(f"[pool] humanvid bank: {len(recs)} clips ({recs[0]['embed_model']})")

    # ---- 3. merged embedding space --------------------------------------------------------
    syn_ids = list(syn["ids"])
    hv_ids = [r["clip_id"] for r in recs]
    assert not (set(syn_ids) & set(hv_ids)), "clip_id collision between the banks"
    all_ids = syn_ids + hv_ids
    emb = {**syn_emb, **hv_emb}
    A = np.stack([emb[c] for c in all_ids])
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    row = {c: i for i, c in enumerate(all_ids)}
    bank_of = {**{c: "synth" for c in syn_ids}, **{c: "humanvid" for c in hv_ids}}
    S_all = A @ A.T
    np.fill_diagonal(S_all, 0.0)
    log(f"[pool] union: {len(all_ids)} clips ({len(syn_ids)} synth + {len(hv_ids)} humanvid), "
        f"one {A.shape} CLIP matrix")

    # ---- 4. bars: imported constants + the reference calibration ---------------------------
    split_path = resolve_ref("ENDPOINT_SPLIT.json")
    split = json.loads(split_path.read_text())
    MATCH_N = len(split["training"])
    BAR_B = split["verification"]["training_participation_ratio"] - GATE_B_TOLERANCE
    log(f"[pool] bars (unchanged, from {split_path}): A <= {GATE_A_MAX_MEAN_COS} | "
        f"B: mean PR over {GATE_B_DRAWS} random n={MATCH_N} subsets >= "
        f"{split['verification']['training_participation_ratio']:.2f} - {GATE_B_TOLERANCE} "
        f"= {BAR_B:.2f}")

    # ---- 5. near-dup pinning over the WHOLE union -----------------------------------------
    iu = np.triu_indices(len(all_ids), 1)
    cvals = S_all[iu]
    hi = np.flatnonzero(cvals >= args.near_dup_cos)
    near_dup = [[all_ids[int(iu[0][k])], all_ids[int(iu[1][k])], round(float(cvals[k]), 4)]
                for k in hi]
    near_dup.sort(key=lambda p: -p[2])
    pinned = {c for p in near_dup for c in p[:2]}
    cross_dup = [p for p in near_dup if bank_of[p[0]] != bank_of[p[1]]]
    log(f"[pool] near-dup pairs at cos >= {args.near_dup_cos}: {len(near_dup)} "
        f"({len(pinned)} clips pinned to training) | CROSS-BANK: {len(cross_dup)}")
    for p in cross_dup:
        log(f"[pool]   CROSS-BANK DUPLICATE {p[2]:.4f}  {p[0]}  <->  {p[1]}")

    # ---- 6. roles -------------------------------------------------------------------------
    syn_training = [syn_entry(e) for e in syn["training"]]
    syn_reserved = [syn_entry(e) for e in syn["reserved"]]
    leaked = [e["clip_id"] for e in syn_reserved if e["clip_id"] in pinned]
    assert not leaked, ("a synth RESERVED clip is in a near-dup pair and would have to be "
                        f"pinned to training: {leaked}")

    rng = np.random.default_rng(RESERVE_SEED)
    eligible = [c for c in hv_ids if c not in pinned]
    hv_reserved_ids = sorted(rng.choice(eligible, args.reserve_humanvid, replace=False).tolist())
    hv_res = set(hv_reserved_ids)

    def hv_entry(cid: str, role: str) -> dict:
        return {"clip_id": cid, "role": role, "bank": "humanvid",
                "mp4": hv_mp4[cid], "source": "humanvid"}

    training = syn_training + [hv_entry(c, "training") for c in hv_ids if c not in hv_res]
    reserved = syn_reserved + [hv_entry(c, "reserved") for c in hv_reserved_ids]
    for e in training + reserved:
        assert Path(e["mp4"]).exists(), f"missing clip file: {e['mp4']}"
    log(f"[pool] roles: training {len(training)} "
        f"({len(syn_training)} synth + {len(training) - len(syn_training)} humanvid) | "
        f"reserved {len(reserved)} ({len(syn_reserved)} synth + {len(hv_reserved_ids)} humanvid)")

    n_syn_train = len(syn_training)
    alive = [row[e["clip_id"]] for e in training]
    before = evaluate(A[alive], MATCH_N, BAR_B)
    log(f"[pool] gates BEFORE trim: A mean_cos {before['gate_a_mean_cos']} "
        f"(<= {GATE_A_MAX_MEAN_COS}) {'PASS' if before['gate_a_pass'] else 'FAIL'} | "
        f"B matched-PR {before['gate_b_matched_pr']} (>= {BAR_B:.2f}) "
        f"{'PASS' if before['gate_b_pass'] else 'FAIL'}")

    # ---- 7. trim: only HUMANVID clips are removable (synth = the protected floor) ----------
    trim_log: list[dict] = []
    live = np.zeros(len(all_ids), bool)
    live[alive] = True
    rowsum = (S_all * live).sum(1)
    removable = np.array([bank_of[all_ids[i]] == "humanvid" for i in range(len(all_ids))])
    cur = before
    if not args.no_trim:
        while not cur["pass"] and len(alive) > 2:
            if len(alive) <= MATCH_N:
                log(f"[pool] SAFETY VALVE: training is down to {len(alive)} = Gate B's match "
                    f"size; refusing to trim further")
                break
            idx = np.array(alive)
            ok = removable[idx]
            if not ok.any():
                log("[pool] no removable clips left — the protected synth floor is all that "
                    "remains")
                break
            mtr = np.where(ok, rowsum[idx] / (len(alive) - 1), -np.inf)
            best = float(mtr.max())
            k = int(np.max(np.flatnonzero(mtr == best)))   # tie -> later index, as max() does
            gone = alive[k]
            trim_log.append({"clip_id": all_ids[gone], "bank": bank_of[all_ids[gone]],
                             "mean_cos_to_pool": round(best, 4),
                             "gate_a": cur["gate_a_mean_cos"],
                             "gate_b": cur["gate_b_matched_pr"]})
            alive.pop(k)
            live[gone] = False
            rowsum -= S_all[gone]
            cur = _eval_fast(A[alive], MATCH_N, BAR_B)
            if len(trim_log) % 50 == 0 or cur["gate_a_pass"]:
                log(f"[pool]   trim {len(trim_log):4d}: dropped {all_ids[gone]:<24} "
                    f"A {cur['gate_a_mean_cos']:.4f} B {cur['gate_b_matched_pr']} "
                    f"({len(alive)} training clips left)")
        kept = {all_ids[i] for i in alive}
        training = [e for e in training if e["clip_id"] in kept]
        if trim_log:
            log(f"[pool] TRIM: removed {len(trim_log)} humanvid clips "
                f"({len(trim_log) / (len(hv_ids) - len(hv_res)):.1%} of the humanvid training "
                f"candidates); the {n_syn_train} synth clips are the protected floor")
    after = cur

    # ---- 8. write the pool FIRST (diagnostics are appended after) --------------------------
    ids = [e["clip_id"] for e in training + reserved]
    V = A[[row[c] for c in ids]]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "content_pool_emb_union.npy", V.astype(np.float32))

    n_hv_train = len(training) - n_syn_train
    pool = {
        "created": time.strftime("%Y-%m-%d"),
        "bank": "union(synth+humanvid)",
        "authority": "advisor ruling S2b: merged union pool, bank mixing enforced inside every "
                     "op (25/50/25 synth-synth / cross / humanvid-humanvid), no bank-pure op. "
                     "Gate definitions and bars imported verbatim from build_content_pool.py "
                     "(exp_081_s2_stratum). n_ops / n_content_pairs are NOT set here.",
        "n_training": len(training), "n_reserved": len(reserved),
        "n_training_synth": n_syn_train, "n_training_humanvid": n_hv_train,
        "n_reserved_synth": len(syn_reserved), "n_reserved_humanvid": len(hv_reserved_ids),
        "embeddings": "data/processed/ctt_v2_strata/content_pool_emb_union.npy "
                      "(L2-normalised float32, row-aligned to `ids`; both banks embedded with "
                      "openai/clip-vit-base-patch32, so the two matrices are comparable)",
        "ids": ids,
        "gates": {"before_trim": before, "after_trim": after,
                  "definition": {
                      "A": f"mean pairwise CLIP cos over training <= {GATE_A_MAX_MEAN_COS}",
                      "B": f"mean PR over {GATE_B_DRAWS} random n={MATCH_N} subsets "
                           f">= {BAR_B:.2f} (= clean-pool PR - {GATE_B_TOLERANCE}; "
                           f"size-matched => n-invariant)"},
                  "bar_provenance": str(split_path),
                  "protected_floor": f"the {n_syn_train} synth training clips (exp_081 shipped "
                                     f"them passing at A 0.5008 / B 42.83); only humanvid clips "
                                     f"are removable, parallel to v1/v2 in the original"},
        "reserved_cut": {"synth": "inherited verbatim from CONTENT_POOL.json",
                         "humanvid_n": args.reserve_humanvid, "seed": RESERVE_SEED,
                         "near_dup_cos": args.near_dup_cos,
                         "rule": "uniform deterministic sample of the humanvid clips not pinned "
                                 "by the near-dup rule; never trimmed, never in the gates"},
        "trim_log": trim_log,
        "diagnostics_non_gating": {},
        "near_dup_pairs_pinned_to_training": near_dup,
        "participation_ratio_pass": after["pass"],     # name kept: planners assert on it
        "training": training, "reserved": reserved,
    }
    (OUT_DIR / "CONTENT_POOL_union.json").write_text(json.dumps(pool, indent=1))
    log(f"[pool] wrote {OUT_DIR / 'CONTENT_POOL_union.json'} ({time.time()-t_start:.0f}s) — "
        f"computing diagnostics")

    # ---- 9. non-gating diagnostics: the three-way head-to-head + quota feasibility ---------
    Vtr = A[[row[e["clip_id"]] for e in training]]
    syn_tr = A[[row[e["clip_id"]] for e in training if e["bank"] == "synth"]]
    hv_tr = A[[row[e["clip_id"]] for e in training if e["bank"] == "humanvid"]]
    hv_all = A[[row[c] for c in hv_ids]]
    diag: dict = {
        "n_training_synth": n_syn_train, "n_training_humanvid": n_hv_train,
        "training_per_source": {},
        "head_to_head": {
            "a_synth_291_training": cos_stats(syn_tr),
            "b_humanvid_1499_all": cos_stats(hv_all),
            "b2_humanvid_training_after_trim": cos_stats(hv_tr) if len(hv_tr) > 1 else None,
            "c_union_training_after_trim": cos_stats(Vtr),
            "cross_bank_synth_x_humanvid_after_trim": cos_stats(syn_tr, hv_tr),
        },
        "cross_bank_duplicates": cross_dup,
        "cross_bank_duplicate_note":
            "the synth bank's vcbench_* clips and HumanVid are both Pexels stock footage, so "
            "the banks overlap. These pairs are pinned to training and the planner's `banned` "
            "set refuses to pair them with each other, but the union still double-counts that "
            "content. Dropping one side is a DESIGN DECISION and was left to the advisor.",
    }
    for e in training:
        diag["training_per_source"][e["source"]] = diag["training_per_source"].get(e["source"],
                                                                                   0) + 1
    # quota feasibility at s2.pair_max_cos, on the FINAL training pool
    h = diag["head_to_head"]
    diag["quota_feasibility_at_pair_max_cos_0.85"] = {
        "synth_synth_available": h["a_synth_291_training"]["n_pairs_le_0.85"],
        "cross_bank_available": h["cross_bank_synth_x_humanvid_after_trim"]["n_pairs_le_0.85"],
        "humanvid_humanvid_available": (h["b2_humanvid_training_after_trim"]["n_pairs_le_0.85"]
                                        if h["b2_humanvid_training_after_trim"] else 0),
        "needed_at_800_pairs_25_50_25": {"synth_synth": 200, "cross": 400, "humanvid": 200},
        "endpoint_slots_at_800_pairs_25_50_25": {"synth": 200 * 2 + 400, "humanvid": 400 + 200 * 2},
        "mean_pair_degree_per_clip": {
            "synth": round((200 * 2 + 400) / max(n_syn_train, 1), 2),
            "humanvid": round((400 + 200 * 2) / max(n_hv_train, 1), 2)},
    }
    diag["gate_b_self_calibrated_bar_VACUOUS"] = round(participation_ratio(Vtr)
                                                       - GATE_B_TOLERANCE, 2)
    diag["gate_b_matched_pr"] = {
        "union_training": round(matched_pr(Vtr, MATCH_N, GATE_B_DRAWS, GATE_SEED), 2),
        "synth_291_training": round(matched_pr(syn_tr, MATCH_N, GATE_B_DRAWS, GATE_SEED), 2),
        "humanvid_1499_all": round(matched_pr(hv_all, MATCH_N, GATE_B_DRAWS, GATE_SEED), 2)}
    res_all = A[[row[e["clip_id"]] for e in reserved]]
    Sres = res_all @ Vtr.T
    diag["max_cos_reserved_vs_training"] = round(float(Sres.max()), 4)
    diag["mean_cos_reserved_vs_training"] = round(float(Sres.mean()), 4)
    diag["reference_exp081_gates_after_trim"] = syn["gates"]["after_trim"]

    pool["diagnostics_non_gating"] = diag
    (OUT_DIR / "CONTENT_POOL_union.json").write_text(json.dumps(pool, indent=1))

    log(f"[pool] training {len(training)} ({n_syn_train} synth + {n_hv_train} humanvid) | "
        f"reserved {len(reserved)}")
    log(f"[pool] gates AFTER  trim: A {after['gate_a_mean_cos']} "
        f"{'PASS' if after['gate_a_pass'] else 'FAIL'} | B {after['gate_b_matched_pr']} "
        f"{'PASS' if after['gate_b_pass'] else 'FAIL'}  => "
        f"{'LAUNCHABLE' if after['pass'] else 'BLOCKED'}")
    for k, v in h.items():
        log(f"[pool] {k:42s} {json.dumps(v)}")
    log(f"[pool] quota feasibility: "
        f"{json.dumps(diag['quota_feasibility_at_pair_max_cos_0.85'])}")
    log(f"[pool] -> {OUT_DIR / 'CONTENT_POOL_union.json'}  ({time.time()-t_start:.0f}s)")
    if not after["pass"] and not args.no_trim:
        raise SystemExit("[pool] BLOCKED: gates still fail after trimming every humanvid clip.")


if __name__ == "__main__":
    main()
