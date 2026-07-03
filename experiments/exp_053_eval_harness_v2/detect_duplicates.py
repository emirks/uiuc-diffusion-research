"""Duplicate / near-duplicate detection over the reference corpus.

Layered, cheapest-first (best practice):
  L1 exact      — SHA-256 of file bytes (literal copies, cross-style too).
  L2 semantic   — cached DINO per-frame features, ORDER-INVARIANT set-similarity
                  (symmetric mean-of-max). Order-invariance is the point: it is
                  robust to TRIMS / re-cuts / offsets, which temporally-aligned
                  per-frame cosine is NOT (a v1 that gated on aligned cosine
                  missed flying_cam_0/1 — a 193-frame trim of a 242-frame clip:
                  aligned 0.876 < gate, but set-sim 0.977). Cheap (cached).
  L3 perceptual — for each candidate, decode and confirm with a bag-of-dHash
                  (order-invariant, INDEPENDENT of DINO — no circularity) plus
                  aligned cosine for temporal-identity context.

A pair is a duplicate iff set-sim is high AND dHash-bag is low (semantic and
perceptual agree) OR the files are byte-identical. Thresholds calibrated against
the genuine distinct-pair distribution. Why it matters: duplicate references
trivially inflate the LOO 1-NN exam and bias the real-clip ceiling upward.
"""

import hashlib
import json
import pathlib

import av
import numpy as np
import yaml
from PIL import Image

REPO = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
cfg = yaml.safe_load((REPO / "experiments/exp_053_eval_harness_v2/config.yaml").read_text())
MODEL, SHORT = cfg["features"]["model"], str(cfg["features"]["short_side"])
CACHE = REPO / cfg["features"]["cache_dir"]
OUT = REPO / "outputs/eval/exp_053/dedup"; OUT.mkdir(parents=True, exist_ok=True)

SETSIM_CAND = 0.80      # decode+confirm anything this similar (order-invariant)
SETSIM_DUP = 0.90       # dup gate: semantic
DHASH_DUP = 10.0        # dup gate: perceptual (bag Hamming), must ALSO hold


def file_key(path, *parts):
    st = pathlib.Path(path).stat()
    return hashlib.sha1("|".join([str(pathlib.Path(path).resolve()), str(st.st_mtime_ns),
                                  str(st.st_size), *parts]).encode()).hexdigest()[:16]


def cache_npz(vid):
    return CACHE / f"dino_arr_{hashlib.sha1(file_key(vid, MODEL, SHORT).encode()).hexdigest()[:16]}.npz"


def scan(root, exclude):
    return {d.name: sorted(d.glob("*.mp4")) for d in sorted(pathlib.Path(root).iterdir())
            if d.is_dir() and d.name not in exclude and any(d.glob("*.mp4"))}


def setsim(A, B):
    S = A @ B.T
    return float(0.5 * (S.max(1).mean() + S.max(0).mean()))


def resample(X, L):
    return X[np.linspace(0, len(X) - 1, L).round().astype(int)]


def aligned_cos(A, B, L=96):
    """Best mean per-frame cosine over ALL offsets — temporal-identity context."""
    A, B = resample(A, L), resample(B, L)
    best = -1.0
    for s in range(-(L - 8), L - 7):
        a, b = (A[s:], B[:L - s]) if s >= 0 else (A[:L + s], B[-s:])
        m = min(len(a), len(b))
        if m >= 12:
            best = max(best, float((a[:m] * b[:m]).sum(1).mean()))
    return best


def decode(path):
    with av.open(str(path)) as c:
        fr = [f.to_image() for f in c.decode(c.streams.video[0])]
    return fr


def dhash_seq(frames, n=24):
    picks = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return np.array([(np.asarray(frames[i].convert("L").resize((9, 8), Image.BILINEAR), float)[:, 1:]
                      > np.asarray(frames[i].convert("L").resize((9, 8), Image.BILINEAR), float)[:, :-1]
                      ).flatten() for i in picks])


def dhash_bag(H1, H2):
    """Order-invariant: mean nearest-hash Hamming, symmetric."""
    d = (H1[:, None, :] != H2[None, :, :]).sum(2)
    return float(min(d.min(1).mean(), d.min(0).mean()))


# --------------------------------------------------------------------------- #
refs = scan(REPO / cfg["data"]["transitions_root"], tuple(cfg["data"]["exclude"]))
names, styles, paths, feats, sha, nfr = [], [], [], [], [], []
for style, vids in refs.items():
    for vid in vids:
        npz = cache_npz(vid)
        if not npz.exists():
            print(f"[skip] {style}/{vid.stem}: no cached features (new/unprocessed)")
            continue
        f = np.load(npz)["feats"]
        names.append(f"{style}/{vid.stem}"); styles.append(style); paths.append(vid)
        feats.append(f); sha.append(hashlib.sha256(vid.read_bytes()).hexdigest()); nfr.append(len(f))
n = len(names)
print(f"{len(set(styles))} styles, {n} cached clips\n")

# ---- L1 exact ----
print("=== L1 — exact file-hash duplicates ===")
by_hash = {}
for i, h in enumerate(sha):
    by_hash.setdefault(h, []).append(i)
exact = {frozenset(v) for v in by_hash.values() if len(v) > 1}
print("  " + ("\n  ".join(" == ".join(names[i] for i in g) for g in exact) if exact else "none"))

# ---- L2 semantic (order-invariant) over all pairs ----
SS = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        SS[i, j] = SS[j, i] = setsim(feats[i], feats[j])
same = [SS[i, j] for i in range(n) for j in range(i + 1, n) if styles[i] == styles[j]]
cross = [SS[i, j] for i in range(n) for j in range(i + 1, n) if styles[i] != styles[j]]
print("\n=== L2 calibration — order-invariant set-similarity ===")
print(f"  distinct (cross-style): median {np.median(cross):.3f}  p99 {np.percentile(cross,99):.3f}"
      f"  max {max(cross):.3f}")
print(f"  same-style (incl dups): median {np.median(same):.3f}  p95 {np.percentile(same,95):.3f}"
      f"  max {max(same):.3f}")
print(f"  candidate cut {SETSIM_CAND}; dup gate set-sim>={SETSIM_DUP} AND dHash-bag<={DHASH_DUP}\n")

cands = sorted(((SS[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                if SS[i, j] >= SETSIM_CAND), reverse=True)
print("=== L3 confirm candidates (decode + independent dHash) ===")
print(f"{'set-sim':>7s} {'dHbag':>6s} {'align':>6s}  frames  pair")
dh = {}
flagged = []
for ss, i, j in cands:
    for k in (i, j):
        if k not in dh:
            dh[k] = dhash_seq(decode(paths[k]))
    hb = dhash_bag(dh[i], dh[j])
    ac = aligned_cos(feats[i], feats[j])
    is_dup = (sha[i] == sha[j]) or (ss >= SETSIM_DUP and hb <= DHASH_DUP)
    tag = "DUP" if is_dup else "review"
    print(f"{ss:7.3f} {hb:6.1f} {ac:6.3f}  {nfr[i]:3d}/{nfr[j]:<3d}  "
          f"{names[i]:32s} <> {names[j]:24s} [{tag}]")
    if is_dup:
        flagged.append((i, j, ss, hb, ac))
if not cands:
    print("  (no candidate pairs above the cut)")

# ---- cluster ----
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j, *_ in flagged:
    parent[find(i)] = find(j)
groups = {}
for i in range(n):
    groups.setdefault(find(i), []).append(i)
dups = [sorted(g, key=lambda k: (-nfr[k], names[k])) for g in groups.values() if len(g) > 1]

print("\n=== duplicate clusters (keep the longest/highest-res; drop the rest) ===")
manifest = []
for g in dups:
    keep, drop = g[0], g[1:]
    st = {styles[k] for k in g}
    print(f"  {'cross-style!' if len(st) > 1 else styles[g[0]]}: KEEP {names[keep]} "
          f"({nfr[keep]}f) | DROP {', '.join(names[k]+f' ({nfr[k]}f)' for k in drop)}")
    manifest.append({"style": styles[g[0]], "keep": names[keep].split("/")[1],
                     "drop": [names[k].split("/")[1] for k in drop],
                     "keep_frames": nfr[keep], "drop_frames": [nfr[k] for k in drop],
                     "set_sim": round(min(SS[keep, k] for k in drop), 3)})
if not dups:
    print("  none")
else:
    print(f"\n  => {len(dups)} cluster(s), {sum(len(g)-1 for g in dups)} redundant clip(s)")
(OUT / "duplicates_v2.json").write_text(json.dumps(
    {"method": "order-invariant set-sim (gate) + independent dHash-bag confirm; see detect_duplicates.py",
     "gates": {"set_sim": SETSIM_DUP, "dhash_bag": DHASH_DUP},
     "clusters": manifest}, indent=2))
print(f"\n[done] {OUT}/duplicates_v2.json")
