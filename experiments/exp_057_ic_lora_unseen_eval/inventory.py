#!/usr/bin/env python
"""exp_057 step 0 — inventory + integrity + dedup of the labeled transitions tree.

Scans data/processed/transitions/{onesided,twosided}_transitions/<label_dir>/*.mp4,
parses sidedness + taxonomy tags + class from the dir name, probes every clip with cv2
(frames/fps/res + actually decodes first/mid/last frames), computes md5 (exact dup)
and 8x8 aHash of first/mid/last frames (near-dup, same-class pairwise Hamming).

Outputs (next to this script):
  inventory.json  — per-clip records + per-class summary
  dedup_report.md — exact dups, near-dup candidate pairs, unreadable/short clips
"""
import cv2
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
SRC = ROOT / "data/processed/transitions"
OUT = Path(__file__).parent

MIN_FRAMES = 121  # standardization target; shorter clips can't be used


def parse_label(dirname: str):
    """'onesided_style_camera_run-set-on-fire' -> ('onesided', ['style','camera'], 'run_set_on_fire')
    Handles the 'onesided_object-monstrosity' typo (hyphen where '_' intended)."""
    m = re.match(r"^(onesided|twosided)[_-](.+)$", dirname)
    sided, rest = m.group(1), m.group(2)
    parts = rest.split("_")
    tags = []
    while parts and parts[0] in ("object", "camera", "style"):
        # a bare taxonomy word followed by nothing else would be the class itself
        if len(parts) == 1:
            break
        tags.append(parts.pop(0))
    cls = "_".join(parts).replace("-", "_")
    if not tags:  # 'onesided_object-monstrosity': rest == 'object-monstrosity'
        bits = rest.split("-", 1)
        if bits[0] in ("object", "camera", "style") and len(bits) > 1:
            tags = [bits[0]]
            cls = bits[1].replace("-", "_")
    return sided, tags, cls


def ahash(frame, size=8):
    g = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY)
    return int("".join("1" if v else "0" for v in (g > g.mean()).flatten()), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def probe(path: Path):
    rec = {"path": str(path.relative_to(SRC))}
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rec.update(
        frames=n,
        fps=round(cap.get(cv2.CAP_PROP_FPS), 3),
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    hashes, ok_all = [], True
    for idx in (0, n // 2, n - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
        ok, fr = cap.read()
        if not ok:
            ok_all = False
            break
        hashes.append(ahash(fr))
    cap.release()
    rec["decodable"] = ok_all
    rec["ahash"] = hashes if ok_all else None
    rec["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    rec["bytes"] = path.stat().st_size
    return rec


def main():
    clips = []
    for side_dir in ("onesided_transitions", "twosided_transitions"):
        for d in sorted((SRC / side_dir).iterdir()):
            if not d.is_dir():
                continue
            sided, tags, cls = parse_label(d.name)
            for f in sorted(d.glob("*.mp4")):  # non-recursive: skips _dup/
                rec = probe(f)
                rec.update(sided=sided, tags=tags, cls=cls, label_dir=d.name)
                clips.append(rec)
                print(f"{rec['path']}: {rec['frames']}f @{rec['fps']} {rec['w']}x{rec['h']}"
                      f"{'' if rec['decodable'] else ' UNDECODABLE'}", flush=True)

    # ---- dedup ----
    by_md5 = defaultdict(list)
    for r in clips:
        by_md5[r["md5"]].append(r["path"])
    exact_dups = {k: v for k, v in by_md5.items() if len(v) > 1}

    near_dups = []
    by_cls = defaultdict(list)
    for r in clips:
        if r["ahash"]:
            by_cls[(r["sided"], r["cls"])].append(r)
    for key, rs in by_cls.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["md5"] == b["md5"]:
                    continue  # already exact
                d = sum(hamming(x, y) for x, y in zip(a["ahash"], b["ahash"]))
                if d <= 12:  # ≤12/192 bits across 3 frames — near-identical
                    near_dups.append({"a": a["path"], "b": b["path"], "dist": d})

    problems = [r["path"] for r in clips if not r["decodable"]]
    short = [(r["path"], r["frames"]) for r in clips if r["frames"] < MIN_FRAMES]

    # ---- summaries ----
    cls_summary = {}
    for (sided, cls), rs in sorted(by_cls.items()):
        cls_summary[f"{sided}/{cls}"] = {
            "n": len(rs),
            "tags": rs[0]["tags"],
            "frames": sorted(set(r["frames"] for r in rs)),
            "fps": sorted(set(r["fps"] for r in rs)),
            "res": sorted(set(f"{r['w']}x{r['h']}" for r in rs)),
            "usable_len": sum(1 for r in rs if r["frames"] >= MIN_FRAMES),
        }

    (OUT / "inventory.json").write_text(json.dumps(
        {"clips": clips, "classes": cls_summary}, indent=1))

    lines = ["# exp_057 corpus dedup / integrity report", "",
             f"- clips probed: {len(clips)}",
             f"- undecodable: {len(problems)} {problems}",
             f"- shorter than {MIN_FRAMES}f: {len(short)}", ""]
    for p, n in short:
        lines.append(f"  - {p}: {n}f")
    lines += ["", f"## Exact duplicates (same md5): {len(exact_dups)} groups", ""]
    for k, v in exact_dups.items():
        lines.append(f"- {v}")
    lines += ["", f"## Near-duplicates (aHash dist<=12/192, same class): {len(near_dups)} pairs", ""]
    for nd in sorted(near_dups, key=lambda x: x["dist"]):
        lines.append(f"- d={nd['dist']}: {nd['a']}  <->  {nd['b']}")
    (OUT / "dedup_report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(clips)} clips, {len(exact_dups)} exact-dup groups, "
          f"{len(near_dups)} near-dup pairs, {len(problems)} undecodable, {len(short)} short")


if __name__ == "__main__":
    main()
