#!/usr/bin/env python3
"""Build the flowsig SIGNAL-INSPECTION viewer.

This is NOT a results page -- it is an instrument for *looking at the 18-channel transition
program* phi(R) itself (the appearance-free signal the campaign feeds LTX-2), one clip at a
time, channel by channel, across the K=16 phase bins, with the generated video(s) that were
conditioned on that exact program shown alongside where they exist.

The signal is ALWAYS extracted from a REAL demonstration clip (DINO-feature correspondence
between its endpoint frames) -- never from generated pixels. So every card shows:
    INPUT  = the program (from our own dataset clip)  ->  OUTPUT = the generation it drove.

Two populations, both 18-channel, both prepared by the campaign:
  * train (v1, RAW)        cache/flowsig_programs/v1_c18_20x15     -- what the LoRA trained on
  * eval  (v2, NORMALISED) cache/flowsig_programs/eval_v2_c18_20x15 -- what conditioned the
                                                                       generated eval samples
plus the __null__ program (the zero signal used for the NULL control).

Store-first: programs come from the registered program stores; media (generated videos +
source demos) are symlinked from store/gens and data/processed. Field data is quantised to
uint8 per channel and written one small JSON per clip so the browser fetches lazily.

  python3 scripts/viewers/build_flowsig_signal.py [--slug flowsig_signal] [--max-train 0]
"""
from __future__ import annotations

import argparse, base64, json, re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO.parent                              # $LAB (repo is $LAB/diffusion-research)
PROG = LAB / "cache/flowsig_programs"          # program stores live outside the repo
STD = REPO / "data/processed/transitions_std121"
GENS = REPO / "store/gens"
GEN_DIRS = ["022_flowsig_ball", "023_flowsig_split"]

STORES = [
    # (population key, store dir, descriptor label, human note)
    ("eval_v2", PROG / "eval_v2_c18_20x15", "v2 (normalised)",
     "the program that CONDITIONED the generated eval samples"),
    ("train_v1", PROG / "v1_c18_20x15", "v1 (raw)",
     "the program population the LoRA was TRAINED on"),
]

# channels that are physically signed -> diverging colormap around 0
SIGNED = {"pi", "pi_n", "dpi", "dnu", "f_x", "f_y",
          "u_A", "v_A", "u_B", "v_B", "da_A", "da_B"}

VID_RE = re.compile(r"__ref_(?P<ref>.+?)__s(?P<seed>\d+)\.mp4$")


def clip_class(clip_id: str) -> str:
    m = re.match(r"^(.*)_\d+$", clip_id)
    return m.group(1) if m else clip_id


def load_sidedness() -> dict:
    """class -> 'one'/'two' from the std121 corpus manifest (eval population)."""
    man = STD / "corpus_manifest.json"
    if not man.exists():
        return {}
    cls = json.load(open(man))["classes"]
    m = {"onesided": "one", "twosided": "two"}
    return {k: m.get(v.get("sidedness"), "?") for k, v in cls.items()}


def find_demo(clip_id: str) -> str | None:
    hits = list(STD.rglob(f"{clip_id}.mp4"))
    return hits[0] if hits else None


def build_video_map() -> dict:
    """clip_id (== program_source / reference) -> list of generated video records."""
    vmap: dict[str, list] = {}
    for gd in GEN_DIRS:
        arm = "split" if "split" in gd else "b_all"
        for cell_dir in sorted((GENS / gd).glob("*/videos")):
            cell = cell_dir.parent.name.replace("__dai", "")
            for mp4 in sorted(cell_dir.glob("*.mp4")):
                m = VID_RE.search(mp4.name)
                if not m:
                    continue
                ref = m.group("ref")
                # endpoint token sits between the arm tag and __ref_
                stem = mp4.name[: m.start()]
                endpoint = stem.split("__")[-1]
                rel = mp4.relative_to(GENS).as_posix()
                vmap.setdefault(ref, []).append({
                    "arm": arm, "cell": cell, "endpoint": endpoint,
                    "seed": int(m.group("seed")),
                    "url": f"media/gens/{rel}",
                })
    return vmap


def quantise(F: np.ndarray, names: list[str]):
    """F: (K,H,W,C) float -> flat uint8 (C,K,H,W order) + per-channel {lo,hi,div}."""
    K, H, W, C = F.shape
    chans = []
    out = np.empty((C, K, H, W), np.uint8)
    for c in range(C):
        v = F[..., c].astype(np.float32)
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        div = names[c] in SIGNED           # only physically-signed channels get a diverging map
        if div:
            m = max(abs(vmin), abs(vmax), 1e-6)
            lo, hi = -m, m
        else:
            lo, hi = vmin, vmax
            if hi - lo < 1e-6:
                hi = lo + 1e-6
        q = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).round().astype(np.uint8)
        out[c] = q  # (K,H,W)
        chans.append({"name": names[c], "lo": round(lo, 5), "hi": round(hi, 5), "div": bool(div)})
    return out.tobytes(), chans, (K, H, W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="flowsig_signal")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap #train clips (0 = all)")
    args = ap.parse_args()

    out_dir = REPO / "outputs/viewers" / args.slug
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # media symlinks (idempotent)
    (out_dir / "media").mkdir(exist_ok=True)
    for name, target in [("gens", GENS), ("std121", STD)]:
        link = out_dir / "media" / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)

    vmap = build_video_map()
    print(f"[video map] {len(vmap)} source clips have generated videos "
          f"({sum(len(v) for v in vmap.values())} clips total)")
    sided = load_sidedness()
    print(f"[sidedness] {len(sided)} classes labelled from corpus_manifest")

    index = []
    for pop, store, desc, note in STORES:
        meta = json.load(open(store / "STORE.json"))
        names = meta["channel_names"]
        npzs = sorted(store.glob("*.npz"))
        if pop == "train_v1" and args.max_train:
            # even sample across classes so the picker stays varied under a cap
            npzs = npzs[:: max(1, len(npzs) // args.max_train)][: args.max_train]
        print(f"[{pop}] {len(npzs)} programs  desc={desc}")
        for npz in npzs:
            raw_id = npz.stem                       # EV__x / S0__x / __null__
            is_null = raw_id.strip("_") == "null"
            clip_id = re.sub(r"^(EV|S0|S1)__", "", raw_id)
            stratum = raw_id.split("__")[0] if "__" in raw_id and not is_null else ("NULL" if is_null else "?")
            z = np.load(npz, allow_pickle=True)
            F = z["F"].astype(np.float32)           # (K,H,W,C)
            field_b, chans, (K, H, W) = quantise(F, names)
            vids = [] if is_null else vmap.get(clip_id, [])
            demo = None if is_null else find_demo(clip_id)
            demo_url = f"media/std121/{demo.relative_to(STD).as_posix()}" if demo else None

            safe = f"{pop}__{raw_id}"
            cls = "null" if is_null else clip_class(clip_id)
            side = "" if is_null else sided.get(cls, "?")
            rec = {
                "id": safe, "clip": clip_id, "pop": pop, "desc": desc,
                "stratum": "NULL" if is_null else stratum, "side": side,
                "cls": cls,
                "K": K, "H": H, "W": W, "channels": chans,
                "g": [[round(float(x), 4) for x in row] for row in z["g"]],
                "d": [round(float(x), 4) for x in z["d"]],
                "field": base64.b64encode(field_b).decode(),
                "videos": vids, "demo_url": demo_url,
            }
            json.dump(rec, open(data_dir / f"{safe}.json", "w"))
            index.append({
                "id": safe, "clip": clip_id, "pop": pop, "desc": desc,
                "stratum": rec["stratum"], "cls": rec["cls"], "side": side,
                "nvid": len(vids), "demo": bool(demo_url),
            })

    # channel doc (shared across both descriptors, by position)
    channel_doc = [
        ("similarity", "s/p_A_loc", "match to the START endpoint, local (r=2) window -- how much this location still looks like A"),
        ("similarity", "s/p_B_loc", "match to the END endpoint, local window -- how much it already looks like B"),
        ("similarity", "s/p_A_glob", "best match to ANY A patch (global) -- appearance-free 'came-from-A'"),
        ("similarity", "s/p_B_glob", "best match to ANY B patch (global) -- 'arrived-at-B'"),
        ("similarity", "pi / pi_n", "PROGRESS = B_loc - A_loc: the transition front (negative early, positive late)"),
        ("novelty",    "nu / nu_p", "NOVELTY = 1 - max(A_glob,B_glob): stuff that matches NEITHER endpoint (born mid-transition)"),
        ("displacement","u_A", "soft-argmax x-offset of the best A match (where content came FROM)"),
        ("displacement","v_A", "y-offset of the best A match"),
        ("displacement","u_B", "x-offset of the best B match (where content is going TO)"),
        ("displacement","v_B", "y-offset of the best B match"),
        ("displacement","a_A/da_A", "entropy/change of the A match softmax -- ambiguity of the A correspondence"),
        ("displacement","a_B/da_B", "ambiguity of the B correspondence"),
        ("flow",       "f_x", "RAFT optical-flow x (fraction of frame/frame)"),
        ("flow",       "f_y", "RAFT optical-flow y"),
        ("flow",       "f_mag", "flow magnitude |f|"),
        ("novelty",    "flicker", "1 - cos(f_t, f_{t-1}): incoherent per-frame motion (max-pooled in each phase bin)"),
        ("similarity", "dpi", "d(progress)/d(phase): the RATE of the transition front"),
        ("novelty",    "dnu", "d(novelty)/d(phase): the rate novelty appears"),
    ]

    payload = {
        "slug": args.slug,
        "stores": [{"pop": p, "desc": d, "note": n, "store": str(s.relative_to(LAB))}
                   for p, s, d, n in STORES],
        "channel_doc": channel_doc,
        "n": len(index), "items": index,
    }
    json.dump(payload, open(out_dir / "index.json", "w"))
    print(f"[done] {len(index)} programs -> {out_dir}/index.json "
          f"({sum(1 for i in index if i['nvid'])} linked to generated video)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
