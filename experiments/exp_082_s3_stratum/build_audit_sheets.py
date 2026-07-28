"""Blind visual-audit sheets for the S3 pilot (and, with --dir, for any S3 render).

Advisor ruling 4 requires a blind visual audit of all 63 pilot clips, <=3 BAD to proceed and no
single camera family contributing >=3 of its 9. "Blind" means the audit order is shuffled and
the clip's family/tag/metrics are NOT printed on the sheet — otherwise knowing that a clip is
the `fog` arm of `crane` primes the judgement. Identities live in BLIND_ORDER.json and are
joined back only after every score is recorded.

Each row is one clip's filmstrip (10 frames spanning the full 121, so a frozen stream, a melt,
a jump cut or a collapsed frame is visible without playing video), labelled ONLY with its blind
id.

    python build_audit_sheets.py [--per-sheet 8] [--dir <run dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO_ROOT / "outputs/videos/ctt_v2_s3/pilot"))
    ap.add_argument("--per-sheet", type=int, default=8)
    a = ap.parse_args()

    run = Path(a.dir)
    order = json.loads((run / "BLIND_ORDER.json").read_text())["order"]
    strip_dir = run / "filmstrips"
    out_dir = run / "audit_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    LABEL_W = 62
    rows_all, sheets = [], []
    for e in order:
        p = strip_dir / f"{e['stem']}.jpg"
        if not p.exists():
            print(f"[audit] MISSING filmstrip: {p.name}")
            continue
        img = PIL.Image.open(p).convert("RGB")
        # scale every strip to a common width so rows align
        W = 1400
        img = img.resize((W, int(img.height * W / img.width)), PIL.Image.LANCZOS)
        canvas = PIL.Image.new("RGB", (W + LABEL_W, img.height), "white")
        canvas.paste(img, (LABEL_W, 0))
        PIL.ImageDraw.Draw(canvas).text((6, img.height // 2 - 6), f"#{e['blind_id']:03d}",
                                        fill="black")
        rows_all.append((e["blind_id"], np.asarray(canvas)))

    for i in range(0, len(rows_all), a.per_sheet):
        chunk = rows_all[i: i + a.per_sheet]
        sheet = np.concatenate([c[1] for c in chunk], axis=0)
        n = len(sheets)
        path = out_dir / f"audit_{n:02d}.png"
        PIL.Image.fromarray(sheet).save(path)
        sheets.append({"sheet": str(path), "blind_ids": [c[0] for c in chunk]})
        print(f"[audit] {path.name}  blind ids {chunk[0][0]}-{chunk[-1][0]}")

    (run / "AUDIT_SHEETS.json").write_text(json.dumps(
        {"n_sheets": len(sheets), "per_sheet": a.per_sheet, "n_clips": len(rows_all),
         "sheets": sheets,
         "note": "score every blind id GOOD/BAD before joining identities from BLIND_ORDER.json"},
        indent=1))
    print(f"[audit] {len(sheets)} sheets covering {len(rows_all)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
