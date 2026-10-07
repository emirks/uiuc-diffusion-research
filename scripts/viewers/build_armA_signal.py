#!/usr/bin/env python3
"""Build the Arm-A (44-channel DINO-basis operator signal) inspection viewer — v2 of the
flowsig signal viewer.

Reads the per-clip Arm-A fields from $LAB/cache/armA_signals/feat/*.npz (each F is
(16,20,15,44)), quantises each channel to uint8, derives the global temporal profile g(k)
from the field itself, links the source demo clip + any generation conditioned on it, and
writes one small JSON per clip for lazy loading.

  python3 scripts/viewers/build_armA_signal.py [--slug armA_signal]
"""
from __future__ import annotations
import argparse, base64, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO.parent
CAMP = REPO / "misc/2026-08-24_flow_signal_conditioning"
FEAT = LAB / "cache/armA_signals/feat"
STD = REPO / "data/processed/transitions_std121"
GENS = REPO / "store/gens"
GEN_DIRS = ["022_flowsig_ball", "023_flowsig_split"]
sys.path.insert(0, str(CAMP / "armA"))
import armA_extract as A                      # CH_NAMES, CH_GROUPS, SIGNED

VID_RE = re.compile(r"__ref_(?P<ref>.+?)__s(?P<seed>\d+)\.mp4$")

CH_DESC = {
    "u": "transport x — matched displacement to next chunk (latent cells/step, windowed r=5)",
    "v": "transport y — matched displacement to next chunk",
    "conf": "match confidence — mean top-3 cosine at the matched window (0 if fwd-bwd fails)",
    "entropy": "match entropy — spread of the windowed softmax (÷log|window|); flat ⇒ no correspondence",
    "dF_norm": "‖ΔF‖ — raw DINO appearance-energy change, same location (t→t+1)",
    "one_minus_cos": "1−cos — direction-only semantic change, same location",
    "s_A": "similarity to START endpoint (global bank max over A)",
    "s_B": "similarity to END endpoint (global bank max over B)",
    "nu": "novelty = 1 − max(s_A, s_B), per patch before pooling",
    "dLab": "Δcolor — ‖ΔLab‖ same location (t→t+1)",
    "csim_A": "colour closeness to START (−‖Lab−Lab_A‖)",
    "csim_B": "colour closeness to END (−‖Lab−Lab_B‖)",
}
for i in range(32):
    CH_DESC[f"pca_{i:02d}"] = f"DINO-PCA component {i} (frozen 768→32 basis of ℓ2-normed cell features)"

GROUP_OF = {n: g for g, ns in A.CH_GROUPS.items() for n in ns}


def clip_class(cid):
    m = re.match(r"^(.*)_\d+$", cid)
    return m.group(1) if m else cid


def load_sidedness():
    man = STD / "corpus_manifest.json"
    if not man.exists():
        return {}
    cls = json.load(open(man))["classes"]
    m = {"onesided": "one", "twosided": "two"}
    return {k: m.get(v.get("sidedness"), "?") for k, v in cls.items()}


def find_demo(cid):
    hits = list(STD.rglob(f"{cid}.mp4"))
    return hits[0] if hits else None


def build_video_map():
    vmap = {}
    for gd in GEN_DIRS:
        arm = "split" if "split" in gd else "b_all"
        for cell_dir in sorted((GENS / gd).glob("*/videos")):
            cell = cell_dir.parent.name.replace("__dai", "")
            for mp4 in sorted(cell_dir.glob("*.mp4")):
                m = VID_RE.search(mp4.name)
                if not m:
                    continue
                ref = m.group("ref")
                endpoint = mp4.name[: m.start()].split("__")[-1]
                vmap.setdefault(ref, []).append({
                    "arm": arm, "cell": cell, "endpoint": endpoint, "seed": int(m.group("seed")),
                    "url": f"media/gens/{mp4.relative_to(GENS).as_posix()}"})
    return vmap


def quantise(F, names, signed, group_of):
    K, H, W, C = F.shape
    chans, out = [], np.empty((C, K, H, W), np.uint8)
    for c in range(C):
        v = F[..., c].astype(np.float32)
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        div = names[c] in signed
        if div:
            m = max(abs(vmin), abs(vmax), 1e-6); lo, hi = -m, m
        else:
            lo, hi = vmin, vmax
            if hi - lo < 1e-6: hi = lo + 1e-6
        out[c] = np.clip((v - lo) / (hi - lo) * 255, 0, 255).round().astype(np.uint8)
        chans.append({"name": names[c], "lo": round(lo, 5), "hi": round(hi, 5),
                      "div": bool(div), "group": group_of[names[c]]})
    return out.tobytes(), chans, (K, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="armA_signal")
    a = ap.parse_args()
    out_dir = REPO / "outputs/viewers" / a.slug
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "media").mkdir(exist_ok=True)
    for name, target in [("gens", GENS), ("std121", STD)]:
        link = out_dir / "media" / name
        if link.is_symlink() or link.exists(): link.unlink()
        link.symlink_to(target)

    # discover per-metric extension caches ($LAB/cache/armA_signals/metrics/<name>/)
    import armA_metrics as M
    signed = set(A.SIGNED); grp = dict(GROUP_OF); desc = dict(CH_DESC)
    extra_groups, metrics_present = [], []
    for name, meta in M.METRICS.items():
        d = Path(M.METRICS_DIR) / name
        if not (d.exists() and any(d.glob("*.npz"))):
            continue
        metrics_present.append(name)
        for ch in meta["channels"]:
            grp[ch] = meta["group"]
            if ch in meta["signed"]:
                signed.add(ch)
        desc.update(meta["desc"])
        if meta["group"] not in A.CH_GROUPS and meta["group"] not in extra_groups:
            extra_groups.append(meta["group"])

    # ctt_v2 source clips for S2/S4 (mp4 lives in encodes/<stratum>/clips/<stem>.mp4)
    ctt_link = out_dir / "media" / "ctt_v2"
    if ctt_link.is_symlink() or ctt_link.exists(): ctt_link.unlink()
    ctt_link.symlink_to(REPO / "datasets/ctt_v2/encodes")

    vmap = build_video_map()
    sided = load_sidedness()
    allf = sorted(FEAT.glob("*.npz"))
    # sample S2 for browsability (full signal is cached; the viewer shows a slice)
    s2 = [f for f in allf if f.name.startswith(("S2a__", "S2b__"))]
    step = max(1, len(s2) // 1000)
    s2keep = set(s2[::step])
    files = [f for f in allf if not f.name.startswith(("S2a__", "S2b__")) or f in s2keep]
    print(f"[armA] {len(allf)} cached | showing {len(files)} (S2 sampled 1/{step}) | "
          f"metrics: {metrics_present or 'none'}")

    index = []
    for npz in files:
        z = np.load(npz, allow_pickle=True)
        F = z["F"].astype(np.float32)
        cid = str(z["clip"]); pop = str(z["pop"]); the_id = str(z["id"])
        strat = str(z["stratum"])
        cls = str(z["cls"]) if "cls" in z.files else clip_class(cid)
        side = str(z["sided"]) if "sided" in z.files else sided.get(cls, "?")
        names = list(A.CH_NAMES)
        for name in metrics_present:
            mf = Path(M.METRICS_DIR) / name / f"{the_id}.npz"
            if mf.exists():
                F = np.concatenate([F, np.load(mf)["F"].astype(np.float32)], axis=-1)
                names += list(M.METRICS[name]["channels"])
        field_b, chans, (K, H, W) = quantise(F, names, signed, grp)
        gmean = F.mean(axis=(1, 2)); gstd = F.std(axis=(1, 2))
        g = np.concatenate([gmean, gstd], axis=1)
        vids = vmap.get(cid, [])
        if pop in ("S2", "S4"):
            demo_url = f"media/ctt_v2/{strat}/clips/{cid}.mp4"
        else:
            d = find_demo(cid)
            demo_url = f"media/std121/{d.relative_to(STD).as_posix()}" if d else None
        rec = {
            "id": the_id, "clip": cid, "pop": pop, "stratum": strat, "cls": cls, "side": side,
            "K": K, "H": H, "W": W, "channels": chans,
            "g": [[round(float(x), 4) for x in row] for row in g],
            "field": base64.b64encode(field_b).decode(),
            "videos": vids, "demo_url": demo_url,
        }
        json.dump(rec, open(data_dir / f"{the_id}.json", "w"))
        index.append({"id": the_id, "clip": cid, "pop": pop, "stratum": strat,
                      "cls": cls, "side": side, "nvid": len(vids), "demo": bool(demo_url)})

    group_keys = list(A.CH_GROUPS.keys()) + extra_groups
    groups_doc = [{"key": g} for g in group_keys]
    payload = {"slug": a.slug, "n": len(index), "channel_desc": desc,
               "groups": groups_doc, "items": index}
    json.dump(payload, open(out_dir / "index.json", "w"))
    print(f"[done] {len(index)} clips -> {out_dir}/index.json "
          f"({sum(1 for i in index if i['nvid'])} video-linked)")


if __name__ == "__main__":
    main()
