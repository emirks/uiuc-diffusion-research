"""exp_077 D2-FULL — tile per-clip filmstrips into labelled contact sheets for visual audit.

    python contact_sheet.py --sub d2full_firstchunk --per-sheet 8 --out <dir> [--role target]

Each row is one clip's 16-frame filmstrip (t = 0,4,8,14,22,32,44,56,68,80,92,100,108,112,116,120)
with a caption bar carrying stem / shader / easing / params, so the visual taxonomy (flat-colour
matte, black frame, extreme zoom, chromatic glitch, mush) can be scored against the metrics.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SHEET_W = 1800
CAP_H = 34


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="d2full_firstchunk")
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--role", default="both", choices=["both", "target", "reference"])
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--blind", action="store_true",
                    help="caption an INDEX only (no shader / params / metrics) and write the "
                         "index->stem key to a separate JSON, so grading cannot be primed")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config_d2full.yaml").read_text())
    run = REPO_ROOT / cfg["outputs"]["dir"] / args.sub
    out = Path(args.out) if args.out else run / "sheets"
    out.mkdir(parents=True, exist_ok=True)

    tuples = [json.loads(l) for f in sorted((run / "meta").glob("tuples_shard*.jsonl"))
              for l in f.read_text().splitlines() if l.strip()]
    tuples.sort(key=lambda t: t["tuple_id"])
    rows = []
    for t in tuples:
        roles = ("target", "reference") if args.role == "both" else (args.role,)
        for r in roles:
            stem = t["target_stem"] if r == "target" else t["reference_stem"]
            c = t["clips"][r]
            pr = " ".join(f"{k}={v}" for k, v in (t.get("params") or {}).items())
            rows.append({
                "jpg": run / "filmstrips" / f"{stem}.jpg",
                "cap": (f"{stem} [{r[:3]}] {t['shader']} | {t['easing']}"
                        f"{' flip=' + t['flip'] if t['flip'] != 'none' else ''}"
                        f"{' swap' if t['swap'] else ''} | {pr}"),
                "cap2": (f"m1p10={c['m1_p10']:.3f} m1min={c['m1_min']:.3f} "
                         f"seam={c['assert2']['seam_max_ratio']:.2f} dq={c['m2_max_dq']:.3f} "
                         f"pure={c['assert1']['max_pure']:.2f} "
                         f"onset={t['timing']['onset']:.0f}->{t['timing']['release']:.0f}"
                         f"{'  [M1MIN-FLAG]' if c['m1_min_flag'] else ''}"),
            })
    rows = rows[: args.limit]
    if args.blind:
        key = [{"idx": i, "stem": r["cap"].split()[0], "shader": r["cap"].split()[2]}
               for i, r in enumerate(rows)]
        for i, r in enumerate(rows):
            r["cap"], r["cap2"] = f"clip #{i:02d}", ""
        (out / "blind_key.json").write_text(json.dumps(key, indent=1))
        print(f"[sheet] BLIND: key -> {out / 'blind_key.json'}")
    print(f"[sheet] {len(rows)} filmstrips from {run.name}")

    n = args.per_sheet
    sheets = []
    for si in range(0, len(rows), n):
        chunk = rows[si:si + n]
        imgs = []
        for r in chunk:
            if not r["jpg"].exists():
                continue
            im = PIL.Image.open(r["jpg"]).convert("RGB")
            h = max(1, int(round(im.height * SHEET_W / im.width)))
            imgs.append((im.resize((SHEET_W, h), PIL.Image.LANCZOS), r))
        if not imgs:
            continue
        total_h = sum(im.height + CAP_H for im, _ in imgs)
        sheet = PIL.Image.new("RGB", (SHEET_W, total_h), (18, 18, 22))
        d = PIL.ImageDraw.Draw(sheet)
        y = 0
        for im, r in imgs:
            d.text((6, y + 3), r["cap"][:210], fill=(235, 235, 240))
            d.text((6, y + 17), r["cap2"][:210], fill=(150, 200, 255))
            sheet.paste(im, (0, y + CAP_H))
            y += im.height + CAP_H
        p = out / f"sheet_{si // n:02d}.jpg"
        sheet.save(p, quality=88)
        sheets.append(str(p))
        print(f"[sheet] {p}  ({len(imgs)} clips, {sheet.width}x{sheet.height})")
    (out / "sheets.json").write_text(json.dumps(sheets, indent=1))


if __name__ == "__main__":
    main()
