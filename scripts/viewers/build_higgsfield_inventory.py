#!/usr/bin/env python3
"""Higgsfield inventory viewer — which part of the Higgsfield pull is used, and how.

Builds outputs/viewers/higgsfield_inventory/{data.json, media -> raw, curated -> curated, thumbs/}.
The page (index.html, hand-written beside this generator's output) reads data.json.

Sources (all derived, nothing hand-kept):
  raw      data/processed/higgsfield_transitions/<class>/<class>_<i>.mp4  (+ _manifest.json: cdn url -> i)
  curated  data/processed/transitions/{onesided,twosided}_transitions/<folder>/<stem>.mp4  (sidedness)
  corpus   data/processed/transitions_std121/corpus_manifest.json          (the 222-clip eval corpus)
  split    data/processed/transitions_std121/split_v1.2.json               (10 zero-shot classes)
  S0       datasets/ctt_v2/inventories/S0.json                             (139 clips trained as S0)
  grid     store/prompts/001_ctt152_neutral/grid.jsonl                     (152-row CTT grid)

Class tiers (one per class, in this priority):
  grid_seen     trained (S0) and on the grid as seen/unseen reference class
  grid_zs       held-out class on the grid as zero-shot
  trained_off   trained (S0) but not on the grid
  corpus_drop   in the corpus, not trained (<2 trainable clips), not on the grid
  curated_only  curated with sidedness, never processed into the corpus
  raw_only      raw pull only, never curated

Usage:
  python3 scripts/viewers/build_higgsfield_inventory.py            # data.json + symlinks
  python3 scripts/viewers/build_higgsfield_inventory.py thumbs     # contact-sheet thumbnails (ffmpeg)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/processed/higgsfield_transitions"
CURATED = REPO / "data/processed/transitions"
CORPUS_MANIFEST = REPO / "data/processed/transitions_std121/corpus_manifest.json"
SPLIT = REPO / "data/processed/transitions_std121/split_v1.2.json"
S0_INV = REPO / "datasets/ctt_v2/inventories/S0.json"
GRID = REPO / "store/prompts/001_ctt152_neutral/grid.jsonl"
OUT = REPO / "outputs/viewers/higgsfield_inventory"

STEM_RE = re.compile(r"^(.*)_(\d+)$")


def stem_class(stem: str) -> str:
    m = STEM_RE.match(stem)
    return m.group(1) if m else stem


def clip_index(stem: str) -> int:
    m = STEM_RE.match(stem)
    return int(m.group(2)) if m else -1


def load_sources():
    corpus = json.load(open(CORPUS_MANIFEST))["clips"]          # "class/basename.mp4" -> {class, source, ...}
    corpus_by_base = {}
    corpus_by_source = {}     # curated source path (repo-relative) -> (class, corpus stem); catches renamed/typo'd sources
    prefix_class = {}         # stem prefix -> class (e.g. action_run_setonfire -> run_set_on_fire)
    for key, c in corpus.items():
        corpus_by_base[Path(key).name] = {"class": c["class"], "source": c["source"]}
        corpus_by_source[c["source"]] = (c["class"], Path(key).stem)
        prefix_class[stem_class(Path(c["source"]).stem)] = c["class"]

    split = json.load(open(SPLIT))
    zs_classes = set(split["generalist_holdout"])

    s0 = json.load(open(S0_INV))
    s0_stems = {stem: v["group"] for stem, v in s0["clips"].items()}   # 139
    s0_groups = s0["groups"]                                            # class -> {sided, clips}

    grid = [json.loads(l) for l in open(GRID) if l.strip()]
    return corpus_by_base, corpus_by_source, prefix_class, zs_classes, s0_stems, s0_groups, grid


def scan_raw():
    """class folder -> list of (stem, relpath-from-RAW, cdn url)"""
    out = {}
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        idx2url = {}
        mf = d / "_manifest.json"
        if mf.exists():
            try:
                idx2url = {v: k for k, v in json.load(open(mf)).items()}
            except Exception:
                idx2url = {}
        clips = []
        for f in d.glob("*.mp4"):
            stem = f.stem
            clips.append((stem, f"{d.name}/{f.name}", idx2url.get(clip_index(stem))))
        clips.sort(key=lambda t: clip_index(t[0]))
        out[d.name] = clips
    return out


def scan_curated():
    """basename -> {sided, rel (from CURATED), folder}"""
    out = {}
    for sided in ("onesided", "twosided"):
        root = CURATED / f"{sided}_transitions"
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in d.glob("*.mp4"):
                out[f.name] = {"sided": "one" if sided == "onesided" else "two",
                               "rel": f"{sided}_transitions/{d.name}/{f.name}", "folder": d.name}
    return out


def build_data():
    corpus_by_base, corpus_by_source, prefix_class, zs_classes, s0_stems, s0_groups, grid = load_sources()
    raw = scan_raw()
    curated = scan_curated()

    # canonical class per raw folder: the corpus class of any corpus clip found in it, else the folder name
    folder2class = {}
    for folder, clips in raw.items():
        cls = None
        for stem, rel, _ in clips:
            b = Path(rel).name
            if b in corpus_by_base:
                cls = corpus_by_base[b]["class"]
                break
        folder2class[folder] = cls or folder

    # grid usage
    grid_rows_by_class = Counter(r["gt_pool_class"] for r in grid)
    grid_novelty_by_class = defaultdict(set)
    for r in grid:
        grid_novelty_by_class[r["gt_pool_class"]].add(r["ref_novelty"])
    ep_rows = Counter(r["endpoint"] for r in grid)
    ref_rows = Counter(r["reference"] for r in grid if r.get("reference"))
    grid_cells_by_stem = defaultdict(set)
    for r in grid:
        grid_cells_by_stem[r["endpoint"]].add(r["cell"] + ":ep")
        if r.get("reference"):
            grid_cells_by_stem[r["reference"]].add(r["cell"] + ":ref")

    classes = {}   # canonical name -> record

    def rec(name):
        if name not in classes:
            classes[name] = {"name": name, "raw_folder": None, "sided": None, "clips": {}, "aliases": []}
        return classes[name]

    # 1. raw clips
    for folder, clips in raw.items():
        name = folder2class[folder]
        c = rec(name)
        c["raw_folder"] = folder
        if folder != name:
            c["aliases"].append(folder)
        for stem, rel, url in clips:
            c["clips"][stem] = {"stem": stem, "src": f"media/{rel}", "url": url, "raw": True}

    # 2. curated clips (sidedness; also clips whose raw counterpart is missing)
    for base, info in curated.items():
        stem = Path(base).stem
        src_rel = f"data/processed/transitions/{info['rel']}"
        in_corpus_by_source = src_rel in corpus_by_source
        if in_corpus_by_source:                       # corpus names win (fixes e.g. raven_transiton_2 -> raven_transition_2)
            name, stem = corpus_by_source[src_rel]
        elif base in corpus_by_base:
            name = corpus_by_base[base]["class"]
        else:
            name = prefix_class.get(stem_class(stem), stem_class(stem))
        c = rec(name)
        c["sided"] = c["sided"] or info["sided"]
        if stem in c["clips"]:
            c["clips"][stem]["curated"] = True
        else:
            c["clips"][stem] = {"stem": stem, "src": f"curated/{info['rel']}", "url": None, "raw": False, "curated": True}
        if in_corpus_by_source:
            c["clips"][stem]["corpus_src"] = True
        if stem_class(Path(base).stem) != name and stem_class(Path(base).stem) not in c["aliases"]:
            c["aliases"].append(stem_class(Path(base).stem))

    # 3. per-clip flags
    for name, c in classes.items():
        for stem, k in c["clips"].items():
            base = stem + ".mp4"
            k.setdefault("curated", False)
            k["corpus"] = (base in corpus_by_base) or bool(k.pop("corpus_src", False))
            k["s0"] = stem in s0_stems
            k["grid_ep"] = ep_rows.get(stem, 0)
            k["grid_ref"] = ref_rows.get(stem, 0)
            k["grid_cells"] = sorted(grid_cells_by_stem.get(stem, ()))
            # test band = in corpus, class is trained, clip itself not trained
            k["test_band"] = bool(k["corpus"] and name in s0_groups and not k["s0"])

    # 4. class tier + counts
    tier_order = ["grid_seen", "grid_zs", "trained_off", "corpus_drop", "curated_only", "raw_only"]
    for name, c in classes.items():
        clips = list(c["clips"].values())
        n_corpus = sum(k["corpus"] for k in clips)
        n_s0 = sum(k["s0"] for k in clips)
        on_grid = grid_rows_by_class.get(name, 0)
        if name in s0_groups and on_grid:
            tier = "grid_seen"
        elif name in zs_classes:
            tier = "grid_zs"
        elif name in s0_groups:
            tier = "trained_off"
        elif n_corpus:
            tier = "corpus_drop"
        elif any(k["curated"] for k in clips):
            tier = "curated_only"
        else:
            tier = "raw_only"
        if c["sided"] is None and name in s0_groups:
            c["sided"] = s0_groups[name]["sided"]
        c.update({
            "tier": tier,
            "n_raw": sum(k["raw"] for k in clips),
            "n_curated": sum(k["curated"] for k in clips),
            "n_corpus": n_corpus,
            "n_s0": n_s0,
            "n_test": sum(k["test_band"] for k in clips),
            "grid_rows": on_grid,
            "grid_novelty": sorted(grid_novelty_by_class.get(name, ())),
            "n_grid_ep": sum(1 for k in clips if k["grid_ep"]),
            "n_grid_ref": sum(1 for k in clips if k["grid_ref"]),
        })
        c["clips"] = sorted(clips, key=lambda k: (not k["corpus"], clip_index(k["stem"])))
        for k in c["clips"]:
            k["thumb"] = f"thumbs/{name}/{k['stem']}.jpg"

    # 5. summary
    summary = {"tiers": {}, "n_classes": len(classes), "n_clips": sum(len(c["clips"]) for c in classes.values())}
    for t in tier_order:
        cs = [c for c in classes.values() if c["tier"] == t]
        summary["tiers"][t] = {
            "classes": len(cs),
            "clips": sum(len(c["clips"]) for c in cs),
            "names": sorted(c["name"] for c in cs),
        }
    summary["corpus_clips"] = len(corpus_by_base)
    summary["s0_clips"] = len(s0_stems)
    summary["grid_rows"] = len(grid)
    summary["grid_classes"] = len(grid_rows_by_class)
    summary["zs_classes"] = sorted(zs_classes)

    data = {
        "generated_by": "scripts/viewers/build_higgsfield_inventory.py",
        "tier_order": tier_order,
        "tier_labels": {
            "grid_seen": "On the grid: seen / unseen tier (trained as S0)",
            "grid_zs": "On the grid: zero-shot tier (held out, never trained)",
            "trained_off": "Trained (S0), not on the grid",
            "corpus_drop": "In the corpus, not trained, not on the grid (fewer than 2 trainable clips)",
            "curated_only": "Curated with sidedness, never processed into the corpus",
            "raw_only": "Raw pull only, never curated",
        },
        "summary": summary,
        "classes": sorted(classes.values(), key=lambda c: (tier_order.index(c["tier"]), c["name"])),
    }
    return data


def ensure_links():
    OUT.mkdir(parents=True, exist_ok=True)
    for link, target in (("media", RAW), ("curated", CURATED)):
        p = OUT / link
        rel = os.path.relpath(target, OUT)
        if p.is_symlink() or p.exists():
            if p.is_symlink() and os.readlink(p) == rel:
                continue
            p.unlink()
        p.symlink_to(rel)


def cmd_data():
    ensure_links()
    data = build_data()
    (OUT / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    s = data["summary"]
    print(f"classes {s['n_classes']}  clips {s['n_clips']}  corpus {s['corpus_clips']}  S0 {s['s0_clips']}  grid rows {s['grid_rows']} over {s['grid_classes']} classes")
    for t in data["tier_order"]:
        ti = s["tiers"][t]
        print(f"  {t:13s} classes {ti['classes']:3d}  clips {ti['clips']:4d}  {', '.join(ti['names'])}")
    print(f"-> {OUT/'data.json'}")


# ───────────────────────────────────────────── thumbnails (contact sheets) ──

def ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


DUR_RE = re.compile(r"Duration: (\d+):(\d+):(\d+\.\d+)")


def duration(ff: str, path: Path) -> float:
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True)
    m = DUR_RE.search(r.stderr)
    if not m:
        return 5.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def make_sheet(ff: str, src: Path, dst: Path, n: int = 6, h: int = 120) -> str:
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    d = max(duration(ff, src), 0.5)
    # n evenly spaced frames, tiled in one row
    vf = f"fps={n}/{d:.4f},scale=-2:{h},tile={n}x1"
    # -threads 1 on both sides: the login node's pthread limit makes ffmpeg's frame-thread encoder init fail under parallel workers
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-threads", "1", "-i", str(src), "-vf", vf,
                        "-frames:v", "1", "-q:v", "4", "-threads", "1", str(dst)],
                       capture_output=True, text=True)
    return "ok" if r.returncode == 0 and dst.exists() else f"FAIL {r.stderr.strip()[:120]}"


def cmd_thumbs(workers: int = 4):
    data = json.loads((OUT / "data.json").read_text()) if (OUT / "data.json").exists() else build_data()
    ff = ffmpeg_bin()
    jobs = []
    for c in data["classes"]:
        for k in c["clips"]:
            jobs.append((OUT / k["src"], OUT / k["thumb"]))
    print(f"{len(jobs)} clips -> {OUT/'thumbs'}  (ffmpeg: {ff})")
    stats = Counter()
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(lambda j: make_sheet(ff, *j), jobs):
            stats["ok" if res == "ok" else ("skip" if res == "skip" else "fail")] += 1
            if res.startswith("FAIL"):
                print(res)
    print(dict(stats))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "thumbs":
        cmd_thumbs()
    else:
        cmd_data()
