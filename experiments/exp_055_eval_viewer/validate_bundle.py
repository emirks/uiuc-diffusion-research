#!/usr/bin/env python
"""Validate a viewer bundle's integrity WITHOUT a browser:
  - data.json parses and has the expected top-level sections
  - index.html present and non-trivial
  - EVERY asset path referenced in data.json resolves on disk (symlinks followed)
    and is non-empty
  - report null / missing assets (allowed but surfaced)

    python validate_bundle.py outputs/eval/exp_053/viewer
Exit code 0 = all referenced assets present & non-empty; 1 = a referenced asset
is missing or empty (a hard failure).
"""

import json
import pathlib
import sys


def main():
    bundle = pathlib.Path(sys.argv[1]).resolve()
    dj = bundle / "data.json"
    assert dj.exists(), f"missing {dj}"
    data = json.loads(dj.read_text())  # raises on invalid JSON
    print(f"[ok] data.json parses ({dj.stat().st_size/1e6:.2f} MB)")

    idx = bundle / "index.html"
    assert idx.exists() and idx.stat().st_size > 2000, "index.html missing/too small"
    print(f"[ok] index.html present ({idx.stat().st_size/1024:.0f} KB)")

    for sec in ("meta", "corpus", "exam", "glossary", "figures", "items"):
        assert sec in data, f"data.json missing section '{sec}'"
    print("[ok] top-level sections present:", ", ".join(sorted(data.keys())))

    refs = []   # (label, relpath)
    def add(label, p):
        if p:
            refs.append((label, p))

    for it in data.get("items", []):
        for k, v in (it.get("videos") or {}).items():
            add(f"item {it['item_id']} video.{k}", v)
        add(f"item {it['item_id']} filmstrip", it.get("filmstrip"))
    for name, rec in (data["exam"].get("clip_records") or {}).items():
        add(f"exam {name} video", rec.get("video"))
        add(f"exam {name} filmstrip", rec.get("filmstrip"))
    for style, vs in (data["corpus"].get("controls") or {}).items():
        for v in vs:
            add(f"control {style}", v)
    for f in data.get("figures", []):
        add(f"figure {f['name']}", f.get("path"))

    missing, empty, ok = [], [], 0
    for label, rel in refs:
        p = (bundle / rel)
        rp = p.resolve()  # follows symlink
        if not rp.exists():
            missing.append((label, rel))
        elif rp.stat().st_size == 0:
            empty.append((label, rel))
        else:
            ok += 1

    print(f"[ok] {ok}/{len(refs)} referenced assets exist and are non-empty")
    # count videos vs images
    vids = sum(1 for _, r in refs if r.endswith(".mp4"))
    print(f"     ({vids} mp4 symlinks, {len(refs)-vids} copied images)")

    # informational: assets present but null (degraded gracefully)
    null_items = sum(1 for it in data.get("items", [])
                     if not (it.get("videos") or {}).get("generated"))
    null_strips = sum(1 for it in data.get("items", []) if not it.get("filmstrip"))
    print(f"[info] items without generated video: {null_items}; without filmstrip: {null_strips}")
    exam_no_strip = sum(1 for r in (data["exam"].get("clip_records") or {}).values() if not r.get("filmstrip"))
    print(f"[info] exam clips without filmstrip: {exam_no_strip}/{len(data['exam'].get('clip_records') or {})}")

    if missing or empty:
        print("\n[FAIL] integrity problems:")
        for label, rel in missing:
            print(f"   MISSING  {label}: {rel}")
        for label, rel in empty:
            print(f"   EMPTY    {label}: {rel}")
        sys.exit(1)
    print("\n[PASS] bundle integrity OK — every referenced asset resolves and is non-empty.")


if __name__ == "__main__":
    main()
