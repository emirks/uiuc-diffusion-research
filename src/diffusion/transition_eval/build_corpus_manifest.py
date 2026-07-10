"""Build corpus_manifest.json — the single source of truth for the reference
corpus (SPEC §5, OPEN item O1).

Consolidates what was previously scattered across directory names, per-exp
dataset jsons, and ALLOCATION.md into one versioned, hashable document:
per class → sidedness, taxonomy tags, dedup provenance; per clip → source
path, source resolution, verified std121 contract fields.

Deterministic on purpose: sorted keys, no timestamps (git owns dates) — so
rebuilding an unchanged corpus yields an identical hash (versioning.corpus_sha).

Every mapping failure is LOUD (SPEC §2: reject, never adapt silently):
unmatched std class, ambiguous raw match, contract violation → listed and
the build exits nonzero unless --allow-partial.

Usage (any python ≥3.9; ffprobe on PATH for probing):
    python src/diffusion/transition_eval/build_corpus_manifest.py \
        --repo /path/to/diffusion-research [--no-probe] [--allow-partial] \
        [--out data/processed/transitions_std121/corpus_manifest.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

TAG_VOCAB = {"object", "camera", "style"}

# Which dedup pass certified each class (SPEC §5 provenance; see notes/exp/
# exp_053 §6, exp_057 dedup_report, exp_058 standardize_train skips).
DEDUP_PASS = {
    "exp_053/054": [
        "shadow_smoke", "earth_wave", "melt_transition", "display_transition",
        "flame", "raven_transition", "water_bending", "flying_cam_transition",
        "jump_transition", "air_bending", "firelava",
    ],
    "exp_057": [
        "hero_flight", "super_fast_run", "plasma_explosion", "shadow",
        "fire_element", "wireframe", "illustration_scene", "animalization",
        "gas_transformation", "portal", "giant_grab", "money_rain",
        "hole_transition", "seamless_transition",
    ],
    "exp_058": [
        "color_rain", "cotton_cloud", "earth_element", "live_concert",
        "luminous_gaze", "monstrosity", "mystification", "nature_bloom",
        "polygon", "run_set_on_fire", "saint_glow", "sakura_petals",
        "water_element", "wonderland",
    ],
}
CLASS_DEDUP = {c: p for p, cs in DEDUP_PASS.items() for c in cs}

# Portrait: the corpus notation "480x640" is width x height (probed 2026-07-10:
# all 223 std clips are 480w x 640h x 121f @ 24fps).
STD_CONTRACT = {"width": 480, "height": 640, "frames": 121}


def parse_raw_dirname(name: str) -> tuple[str, list[str], str] | None:
    """'onesided_style_camera_illustration-scene' -> ('onesided', ['style','camera'],
    'illustration-scene'). Tolerates the malformed 'onesided_object-monstrosity'."""
    tokens = name.split("_")
    if tokens[0] not in ("onesided", "twosided"):
        return None
    sidedness, rest = tokens[0], tokens[1:]
    tags, cls_parts = [], []
    for tok in rest:
        head = tok.split("-")[0]
        if not cls_parts and tok in TAG_VOCAB:
            tags.append(tok)
        elif not cls_parts and head in TAG_VOCAB and tok != head:
            # malformed 'object-monstrosity': tag glued to class with '-'
            tags.append(head)
            cls_parts.append(tok[len(head) + 1:])
        else:
            cls_parts.append(tok)
    return sidedness, tags, "_".join(cls_parts)


def std_name_candidates(raw_class_slug: str) -> list[str]:
    """Raw slug -> possible std121 dir names ('flame-transition' -> flame_transition,
    flame). Hyphens normalize to underscores; trailing '_transition' optional."""
    base = raw_class_slug.replace("-", "_")
    cands = [base]
    if base.endswith("_transition"):
        cands.append(base[: -len("_transition")])
    return cands


def ffprobe(path: pathlib.Path) -> dict | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        s = json.loads(out.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return {"width": int(s["width"]), "height": int(s["height"]),
                "frames": int(s.get("nb_frames") or 0),
                "fps": round(int(num) / int(den), 3)}
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="diffusion-research root (data source)")
    ap.add_argument("--out", default="data/processed/transitions_std121/corpus_manifest.json")
    ap.add_argument("--no-probe", action="store_true", help="skip ffprobe verification")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    std_root = repo / "data/processed/transitions_std121"
    raw_root = repo / "data/processed/transitions"

    # --- index the raw labeled tree ------------------------------------------
    raw_index = {}  # std_candidate_name -> (sidedness, tags, raw_dir)
    for side_dir in ("onesided_transitions", "twosided_transitions"):
        for d in sorted((raw_root / side_dir).iterdir()):
            if not d.is_dir():
                continue
            parsed = parse_raw_dirname(d.name)
            if parsed is None:
                print(f"[warn] unparseable raw dir skipped: {d.name}")
                continue
            sidedness, tags, slug = parsed
            for cand in std_name_candidates(slug):
                if cand in raw_index:
                    print(f"[error] ambiguous raw mapping for '{cand}': "
                          f"{raw_index[cand][2].name} vs {d.name}")
                    if not args.allow_partial:
                        return 1
                raw_index[cand] = (sidedness, tags, d)

    # --- walk the std corpus ---------------------------------------------------
    classes, clips, problems = {}, {}, []
    std_dirs = sorted(p for p in std_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for cdir in std_dirs:
        cls = cdir.name
        vids = sorted(cdir.glob("*.mp4"))
        if not vids:
            continue
        if cls not in raw_index:
            problems.append(f"std class '{cls}' has no match in the raw labeled tree")
            sidedness, tags, raw_dir = None, [], None
        else:
            sidedness, tags, raw_dir = raw_index[cls]
        if cls not in CLASS_DEDUP:
            problems.append(f"std class '{cls}' missing a dedup-pass assignment")
        classes[cls] = {
            "sidedness": sidedness,
            "tags": sorted(tags),
            "dedup_pass": CLASS_DEDUP.get(cls),
            "n_clips": len(vids),
            "raw_dir": str(raw_dir.relative_to(repo)) if raw_dir else None,
        }
        # exact source matches first; fuzzy recovery may only use sources no
        # sibling has exactly claimed (else 'shadow_smoke_0' steals '_9')
        claimed = {v.name for v in vids if raw_dir and (raw_dir / v.name).exists()}
        for v in vids:
            key = f"{cls}/{v.name}"
            src = raw_dir / v.name if raw_dir else None
            source_match = "exact"
            if raw_dir and (src is None or not src.exists()):
                # raw tree carries occasional filename quirks (a typo'd
                # 'raven_transiton_2', a suffixless 'shadow_smoke.mp4') —
                # recover via unique close-match among UNCLAIMED sources only,
                # loudly recorded as fuzzy.
                import difflib
                cands = [p.name for p in raw_dir.glob("*.mp4") if p.name not in claimed]
                near = difflib.get_close_matches(v.name, cands, n=2, cutoff=0.6)
                if near:
                    src = raw_dir / near[0]
                    source_match = f"fuzzy:{near[0]}"
                    claimed.add(near[0])
            entry = {
                "class": cls,
                "source": str(src.relative_to(repo)) if src and src.exists() else None,
                "source_match": source_match if src and src.exists() else None,
                "source_resolution": None,
                "std": None,
                "contract_ok": None,
            }
            if not args.no_probe:
                meta = ffprobe(v)
                entry["std"] = meta
                if meta:
                    entry["contract_ok"] = all(
                        meta[k] == STD_CONTRACT[k] for k in STD_CONTRACT)
                    if not entry["contract_ok"]:
                        problems.append(f"{key}: std contract violation {meta}")
                else:
                    problems.append(f"{key}: ffprobe failed on std clip")
                if src and src.exists():
                    smeta = ffprobe(src)
                    if smeta:
                        entry["source_resolution"] = [smeta["width"], smeta["height"]]
            if entry["source"] is None:
                problems.append(f"{key}: no source clip found in raw tree")
            clips[key] = entry

    manifest = {
        "schema": 1,
        "corpus_root": "data/processed/transitions_std121",
        "std_contract": {**STD_CONTRACT, "fps": 24},
        "n_classes": len(classes),
        "n_clips": len(clips),
        "classes": classes,
        "clips": clips,
        "problems": sorted(problems),
        "built_by": "src/diffusion/transition_eval/build_corpus_manifest.py",
    }
    out = repo / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    print(f"[done] {len(classes)} classes / {len(clips)} clips -> {out}")
    if problems:
        print(f"[problems] {len(problems)}:")
        for p in problems[:30]:
            print(f"  - {p}")
        if not args.allow_partial:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
