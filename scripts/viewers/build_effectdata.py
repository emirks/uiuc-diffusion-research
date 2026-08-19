#!/usr/bin/env python3
"""Build the EffectData viewer under outputs/viewers/effectdata/.

Wiring only: copies the two tracked page sources into the viewer dir and
symlinks ./media at the external dataset staging dir. The media (previews,
cf_media clips) and the data JSONs (effects_index.json, cf_data.json,
preview_manifest.json) live OUTSIDE the repo, in the staging dir, and are the
source of truth — never copied into outputs/.

Dataset staging: $LAB/data_external/EffectData  (LAB defaults to the taiga path).
Regenerate the data there with the scripts in scripts/viewers/effectdata_pipeline/
(build_index.py, build_counterfactual.py, remote_zip.py) if it was wiped —
the raw set is re-downloadable from HF (ysy31415926/EffectData, Apache-2.0).
"""
import os, pathlib, shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
LAB = os.environ.get("LAB", "/taiga/illinois/eng/cs/jrehg/users/emirkisa")
STAGE = pathlib.Path(LAB) / "data_external" / "EffectData"
VIEW = REPO / "outputs" / "viewers" / "effectdata"
SRC = REPO / "scripts" / "viewers"

def main():
    VIEW.mkdir(parents=True, exist_ok=True)
    # copy tracked page sources
    shutil.copyfile(SRC / "effectdata_index.html", VIEW / "index.html")
    shutil.copyfile(SRC / "effectdata_counterfactual.html", VIEW / "counterfactual.html")
    print(f"pages: index.html, counterfactual.html -> {VIEW.relative_to(REPO)}")
    # media symlink (relative, points outside the repo into the staging dir)
    link = VIEW / "media"
    if not STAGE.exists():
        print(f"  WARNING: staging dir missing: {STAGE}")
    rel = os.path.relpath(STAGE, VIEW)
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            raise SystemExit(f"real file in the way: {link}")
    link.symlink_to(rel)
    print(f"  media -> {rel}")
    # quick presence check
    for f in ["effects_index.json", "cf_data.json", "preview_manifest.json",
              "example_preview", "cf_media"]:
        ok = (link / f).exists()
        print(f"  media/{f:<22} {'ok' if ok else 'MISSING'}")

if __name__ == "__main__":
    main()
