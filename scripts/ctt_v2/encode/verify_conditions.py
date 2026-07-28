#!/usr/bin/env python3
"""Verify the content-addressed conditions tree, and settle what the OLD S0 embeds encode.

Checks, and nothing beyond them:

  coverage   every distinct caption of every stratum has an embed file
  shape      `video_prompt_embeds (1024,3840) bf16`, `prompt_attention_mask (1024,) int64`,
             `audio_prompt_embeds (1024,3840) bf16` -- the format S0's certified embeds carry
  distinct   the embeds are not all the same tensor.  This is the check that would have caught
             the rehearsal's 168,184-samples-share-one-file defect, which every path-alignment
             assert passes on BY CONSTRUCTION (a placeholder aligns perfectly).
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
    seen_content: Counter = Counter()

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
            d = torch.load(OUT / f"{h}.pt", map_location="cpu", weights_only=True)
            if set(d) != set(EXPECT):
                badshape.append(f"{h}: keys {sorted(d)}")
                continue
            for k, (shp, dt) in EXPECT.items():
                if tuple(d[k].shape) != shp or str(d[k].dtype) != dt:
                    badshape.append(f"{h}.{k}: {tuple(d[k].shape)}/{d[k].dtype}")
            seen_content[content_sha(d["video_prompt_embeds"])] += 1
        sr["shape_violations"] = len(badshape)
        sr["shape_violation_examples"] = badshape[:5]
        if badshape:
            hard.append(f"{st}: {len(badshape)} shape/key violations")
        rep["strata"][st] = sr

    n_files = sum(v["present"] for v in rep["strata"].values())
    rep["distinct_check"] = {
        "embed_files_examined": n_files,
        "distinct_video_prompt_embeds": len(seen_content),
        "most_shared_tensor_count": max(seen_content.values()) if seen_content else 0,
        "rule": "distinct tensor count must equal the file count: each file is one distinct "
                "caption by construction (content-addressed), so two files sharing a tensor "
                "means the encoder ignored its input -- the placeholder defect.",
    }
    if seen_content and len(seen_content) != n_files:
        hard.append(f"distinct tensors {len(seen_content)} != files {n_files} -- embeds are "
                    f"being shared where they should be unique")

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
        a = torch.load(new_p, map_location="cpu", weights_only=True)["video_prompt_embeds"]
        b = torch.load(old_p, map_location="cpu", weights_only=True)["video_prompt_embeds"]
        cmp_out.append({"clip": clip, "identical": bool(torch.equal(a, b)),
                        "max_abs_diff": float((a.float() - b.float()).abs().max())})
    n_same = sum(c["identical"] for c in cmp_out)
    rep["s0_old_vs_new"] = {
        "compared": len(cmp_out), "identical": n_same, "detail": cmp_out,
        "verdict": ("OLD S0 EMBEDS CONFIRMED = certified training text"
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
