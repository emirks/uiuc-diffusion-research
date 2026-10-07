"""Build the lerp-collapse viewer: per-clip DR / M (tau mid-band) / class across the
interaction axes (arm x conditioning x class), with the actual clips. Regenerable.
Reuses the certified classifier (phase2_classify) + calibration. Decode-only."""
import os, sys, json, glob, re, shutil
import numpy as np, cv2
cv2.setNumThreads(1)
DR = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
CAMP = f"{DR}/misc/2026-08-24_lerp_collapse"
sys.path.insert(0, CAMP)
from phase2_classify import load_matrix, geom, classify
from phase1 import load_grid
VIEW = f"{DR}/outputs/viewers/lerp_collapse"
MEDIA = f"{VIEW}/media"
PREFIX, SUFFIX = 9, 8
cal = json.load(open(f"{CAMP}/phase2_calibration.json"))["cal"]
MTH, STH = cal["M_thr"], cal["S_thr"]

# (arm_label, videos_dir, grid_path, mode)  mode: per_row | force_one
ARMS = [
    ("BASE LTX (no adapter)", "store/gens/005_base_cond/02_neutral__dai", "per_row", "base_cond"),
    ("CTT-v3 (trained)",      "store/gens/009_ctt_v3/04_neutral__dai",    "per_row", "ctt_v3"),
    ("BASE LTX · same-rows start-only (causal control)",
     "misc/2026-08-24_lerp_collapse/tier2_gen/out", "force_one", "tier2"),
]


def classify_clip(v, two):
    M = load_matrix(v)
    if M is None: return None
    g = geom(M, two=two, prefix=PREFIX, suffix=SUFFIX)
    static = (not np.isfinite(g["gap"])) or g["gap"] < 12.0   # gap proxy for the floor guard
    cls = "STATIC" if static else classify(g["DR"], g["M"], MTH, STH, g["R"], g["S"], False)
    return g, cls


def main():
    os.makedirs(MEDIA, exist_ok=True)
    clips = []
    for arm_label, vdir_rel, mode, key in ARMS:
        vdir = f"{DR}/{vdir_rel}"
        grid = load_grid(vdir) if mode == "per_row" else {}
        vids = sorted(glob.glob(f"{vdir}/videos/*.mp4"))
        # symlink the arm's videos dir into the viewer
        link = f"{MEDIA}/{key}"
        if os.path.islink(link) or os.path.exists(link): os.remove(link) if os.path.islink(link) else None
        if not os.path.exists(link): os.symlink(f"{vdir}/videos", link)
        print(f"[{key}] {len(vids)} clips", flush=True)
        for v in vids:
            item = re.sub(r"__s\d+\.mp4$", "", os.path.basename(v))
            m = re.search(r"__s(\d+)\.mp4$", v); seed = m.group(1) if m else "?"
            if mode == "per_row":
                two = grid.get(item, {}).get("sided") == "two"
            else:
                two = False
            r = classify_clip(v, two)
            if r is None: continue
            g, cls = r
            clips.append(dict(arm=arm_label, key=key, cond=("both-endpoint" if two else "start-only"),
                              seed=seed, item=item, foreign=("davis_" in item or "foreign" in item),
                              DR=round(g["DR"], 3) if np.isfinite(g["DR"]) else None,
                              M=round(g["M"], 3) if np.isfinite(g["M"]) else None,
                              cls=cls, media=f"media/{key}/{os.path.basename(v)}"))
    json.dump(clips, open(f"{VIEW}/clips.json", "w"))
    print(f"wrote {len(clips)} clips", flush=True)
    # copy static figures
    for f in ["fig_class_distribution.png", "fig_dr_m_scatter.png", "blindcheck.png", "stepc_montage.png"]:
        src = f"{CAMP}/{f}"
        if os.path.exists(src): shutil.copy(src, f"{VIEW}/{f}")
    print("figures copied", flush=True)


if __name__ == "__main__":
    main()
