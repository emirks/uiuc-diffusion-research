#!/usr/bin/env python3
"""Verify the content-addressed conditions tree, and settle what the OLD S0 embeds encode.

Checks, and nothing beyond them:

  coverage   every distinct caption of every stratum has an embed file
  shape      `video_prompt_embeds (1024,3840) bf16`, `prompt_attention_mask (1024,) int64`,
             `audio_prompt_embeds (1024,3840) bf16` -- the format S0's certified embeds carry
  distinct   two DIFFERENT caption texts never produce the same tensor.  This is the check that
             would have caught the rehearsal's 168,184-samples-share-one-file defect, which every
             path-alignment assert passes on BY CONSTRUCTION (a placeholder aligns perfectly).

             ⚠ It is stated over the UNION of referenced caption hashes, not the per-stratum sum.
             Cross-stratum sharing is the DESIGN of a content-addressed store, not a defect: S1's
             s0cf layer carries the certified S0 caption verbatim, so all 139 S0 hashes are a
             subset of S1's 434 and the per-stratum sum (3,704) legitimately exceeds the distinct
             count (3,565) by exactly that 139.  Comparing the sum against distinct tensors made
             correct sharing indistinguishable from a placeholder collapse, and failed the job.
  s0_text    ⚠ advisor A21's open question: the pre-existing `eval_ladder/dataset/conditions/`
             embeds date from Jul 22 and their TEXT provenance was never recorded. We now
             encode S0 ourselves from the certified captions, so comparing the new tensor
             against the old one for the same clip ANSWERS it: identical => the old embeds
             encode the certified training text; different => they encode something else and
             nothing may cite them as "real".

Comparison is by TENSOR, never by file bytes: `torch.save`'s container records an mtime, so
two saves of an identical tensor differ (A20.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
IN = REPO / "outputs/ctt_v2/conditions_inputs"
OUT = REPO / "outputs/ctt_v2/conditions/by_caption"
S0_OLD = REPO / "eval_ladder/dataset/conditions"

EXPECT = {
    "video_prompt_embeds": ((1024, 3840), "torch.bfloat16"),
    "prompt_attention_mask": ((1024,), "torch.int64"),
    "audio_prompt_embeds": ((1024, 3840), "torch.bfloat16"),
}


def content_sha(t) -> str:
    c = t.contiguous()
    return hashlib.sha256(
        f"{tuple(c.shape)}|{c.dtype}|".encode() + c.float().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    import torch

    man = json.loads((IN / "ENCODE_INPUTS_MANIFEST.json").read_text())
    rep = {"schema": "ctt_v2_conditions_report/v1",
           "at": datetime.now(timezone.utc).isoformat(),
           "tree": str(OUT.relative_to(REPO)), "strata": {}}
    hard: list[str] = []
    #: caption_hash -> tensor content sha.  Keyed by hash so a file referenced by two strata is
    #: examined ONCE; the distinct check is then a bijection test over the union.
    file_content: dict[str, str] = {}

    for st, rec in man["strata"].items():
        rows = json.loads((IN / f"{st}_captions.json").read_text())
        want = [r["media_path"][:-4] for r in rows]
        present = [h for h in want if (OUT / f"{h}.pt").exists()]
        missing = [h for h in want if not (OUT / f"{h}.pt").exists()]
        sr = {"expected": len(want), "present": len(present), "missing": len(missing),
              "missing_examples": missing[:5]}
        if missing:
            hard.append(f"{st}: {len(missing)} of {len(want)} embeds absent")

        badshape = []
        for h in present:
            if h in file_content:      # already examined via another stratum -- same file
                continue
            d = torch.load(OUT / f"{h}.pt", map_location="cpu", weights_only=True)
            if set(d) != set(EXPECT):
                badshape.append(f"{h}: keys {sorted(d)}")
                continue
            for k, (shp, dt) in EXPECT.items():
                if tuple(d[k].shape) != shp or str(d[k].dtype) != dt:
                    badshape.append(f"{h}.{k}: {tuple(d[k].shape)}/{d[k].dtype}")
            file_content[h] = content_sha(d["video_prompt_embeds"])
        sr["shape_violations"] = len(badshape)
        sr["shape_violation_examples"] = badshape[:5]
        if badshape:
            hard.append(f"{st}: {len(badshape)} shape/key violations")
        rep["strata"][st] = sr

    per_stratum_sum = sum(v["present"] for v in rep["strata"].values())
    tensors: Counter = Counter(file_content.values())
    collisions = {sha: n for sha, n in tensors.items() if n > 1}
    rep["distinct_check"] = {
        "distinct_caption_hashes_examined": len(file_content),
        "per_stratum_sum": per_stratum_sum,
        "cross_stratum_shared": per_stratum_sum - len(file_content),
        "distinct_video_prompt_embeds": len(tensors),
        "most_shared_tensor_count": max(tensors.values()) if tensors else 0,
        "rule": "BIJECTION over the union of referenced caption hashes: N distinct caption "
                "hashes must yield N distinct tensors. Two hashes sharing a tensor means the "
                "encoder ignored its input (the placeholder defect). Cross-stratum reuse of "
                "the SAME hash is the design and is excluded by keying on the hash.",
    }
    if collisions:
        hard.append(f"{len(collisions)} tensor(s) shared by >1 distinct caption hash "
                    f"(max {max(collisions.values())}) -- the encoder ignored its input")

    # ---- the S0 text-provenance question ---------------------------------------------------
    s0_rows = json.loads((IN / "S0_captions.json").read_text())
    c2h = json.loads((IN / "S0_clip_to_hash.json").read_text())
    inv = json.loads((REPO / "outputs/ctt_v2/inventories/S0.json").read_text())
    cmp_out = []
    for clip in sorted(c2h)[:8]:
        new_p = OUT / f"{c2h[clip]}.pt"
        old_p = S0_OLD / inv["clips"][clip]["group"] / f"{clip}.pt"
        if not (new_p.exists() and old_p.exists()):
            continue
        a = torch.load(new_p, map_location="cpu", weights_only=True)["video_prompt_embeds"].float()
        b = torch.load(old_p, map_location="cpu", weights_only=True)["video_prompt_embeds"].float()
        # NOT `torch.equal`: these are bf16 tensors produced by two different runs on
        # different hardware, so the same text re-encodes to a bf16-rounding difference
        # (max_abs_diff lands on exact powers of two -- 1/64, 1/128 -- which is the giveaway).
        # Exact equality here is the same error class as comparing `torch.save` file bytes:
        # a precision the process never promised. The threshold is calibrated against a
        # MEASURED control -- two genuinely different captions give cos 0.27 / rel_l2 1.31,
        # while same-text re-encodes give cos 0.99996 / rel_l2 5e-3. Any bar between those
        # separates them by ~250x, so its exact value is not load-bearing.
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        rel = float((a - b).norm() / b.norm())
        cmp_out.append({"clip": clip, "same_text": cos >= 0.999 and rel <= 1e-2,
                        "cosine": round(cos, 6), "rel_l2": round(rel, 6),
                        "max_abs_diff": float((a - b).abs().max())})
    n_same = sum(c["same_text"] for c in cmp_out)
    rep["s0_old_vs_new"] = {
        "compared": len(cmp_out), "identical": n_same, "detail": cmp_out,
        "bar": "cos >= 0.999 AND rel_l2 <= 1e-2 (measured control: different captions give "
               "cos 0.27 / rel_l2 1.31; same text re-encoded gives cos 0.99996 / rel_l2 5e-3)",
        "verdict": ("OLD S0 EMBEDS CONFIRMED = the certified training text (differences are "
                    "bf16 re-encode rounding, ~250x smaller than any real text difference)"
                    if cmp_out and n_same == len(cmp_out) else
                    "MISMATCH -- the pre-existing S0 embeds encode DIFFERENT text; do not cite "
                    "them as real, use the freshly encoded ones"
                    if cmp_out else "not comparable (files absent)"),
        "authority": "advisor A21 open question (b)",
    }

    rep["hard_fail"] = hard
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "strata"}, indent=2)[:2600])
    for st, v in rep["strata"].items():
        print(f"[{st}] {v['present']}/{v['expected']} present, "
              f"{v['shape_violations']} shape violations")
    print(f"[hard_fail] {hard or 'NONE'}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
