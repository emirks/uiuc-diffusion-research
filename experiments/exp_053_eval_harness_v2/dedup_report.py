"""Quantify the eval impact of the 3 near-duplicate clusters and produce
non-destructive artifacts: a visual-proof montage, a duplicates.json manifest
(cluster + keep/drop + scores), and the corrected LOO 1-NN exam with the
redundant low-res twins removed. Touches no data files."""

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
NP, NS = cfg["morph"]["n_prefix"], cfg["morph"]["n_suffix"]
OUT = REPO / "outputs/eval/exp_053/dedup"; OUT.mkdir(parents=True, exist_ok=True)

# clusters found by detect_duplicates.py; DROP the lower-resolution twin.
CLUSTERS = [
    {"style": "melt_transition",    "keep": "melt_transition_3",    "drop": "melt_transition_0",
     "aligned_cos": 0.992, "dhash_hamming": 0.0},
    {"style": "jump_transition",    "keep": "jump_transition_1",    "drop": "jump_transition_0",
     "aligned_cos": 0.990, "dhash_hamming": 0.0},
    {"style": "display_transition", "keep": "display_transition_3", "drop": "display_transition_0",
     "aligned_cos": 0.988, "dhash_hamming": 0.0},
]
DROP = {f"{c['style']}/{c['drop']}" for c in CLUSTERS}


def file_key(path, *parts):
    st = pathlib.Path(path).stat()
    raw = "|".join([str(pathlib.Path(path).resolve()), str(st.st_mtime_ns),
                    str(st.st_size), *parts])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cache_npz(vid):
    return CACHE / f"dino_arr_{hashlib.sha1(file_key(vid, MODEL, SHORT).encode()).hexdigest()[:16]}.npz"


def scan(root, exclude):
    return {d.name: sorted(d.glob("*.mp4")) for d in sorted(pathlib.Path(root).iterdir())
            if d.is_dir() and d.name not in exclude and any(d.glob("*.mp4"))}


def morph_env(f):
    eA = f[:NP].mean(0); eA /= np.linalg.norm(eA) + 1e-12
    eB = f[-NS:].mean(0); eB /= np.linalg.norm(eB) + 1e-12
    cross = float(eA @ eB); den = max(1 - cross, 1e-6)
    return np.maximum(np.clip((f @ eA - cross) / den, -.25, 1.25),
                      np.clip((f @ eB - cross) / den, -.25, 1.25))


def core_mask(env):
    T = len(env); m = env < 0.5; m[:NP] = False; m[T - NS:] = False
    if not m.any():
        me = env.copy(); me[:NP] = np.inf; me[T - NS:] = np.inf; m[int(np.argmin(me))] = True
    return m


def setsim(A, B):
    S = A @ B.T
    return float(0.5 * (S.max(1).mean() + S.max(0).mean()))


refs = scan(REPO / cfg["data"]["transitions_root"], tuple(cfg["data"]["exclude"]))
names, styles, paths, fcore = [], [], [], []
for style, vids in refs.items():
    for vid in vids:
        f = np.load(cache_npz(vid))["feats"]
        names.append(f"{style}/{vid.stem}"); styles.append(style); paths.append(vid)
        fcore.append(f[core_mask(morph_env(f))])
n = len(names); styles = np.array(styles); idx = {nm: i for i, nm in enumerate(names)}

S = np.full((n, n), -np.inf)
for i in range(n):
    for j in range(i + 1, n):
        S[i, j] = S[j, i] = setsim(fcore[i], fcore[j])


def exam(keep_mask):
    """LOO 1-NN over the kept sub-corpus; returns acc, per-class, testable set."""
    keep = np.where(keep_mask)[0]
    lab = styles[keep]
    counts = {s: int((lab == s).sum()) for s in set(lab)}
    correct, per = [], {s: [] for s in sorted(set(styles))}
    for i in keep:
        cand = [j for j in keep if j != i]
        Srow = np.array([S[i, j] for j in cand])
        pred = styles[cand[int(Srow.argmax())]]
        hit = pred == styles[i]
        correct.append(hit); per[styles[i]].append(hit)
    acc = float(np.mean(correct))
    per_recall = {s: (float(np.mean(v)) if v else None) for s, v in per.items()}
    testable = {s: c for s, c in counts.items() if c >= 2}
    acc_testable = float(np.mean([h for i, h in zip(keep, correct)
                                  if counts[styles[i]] >= 2]))
    return acc, per_recall, counts, acc_testable, testable


full = np.ones(n, bool)
keep_mask = np.array([nm not in DROP for nm in names])
acc0, per0, cnt0, _, _ = exam(full)
acc1, per1, cnt1, acc1t, testable = exam(keep_mask)

print("=== corrected exam (core-mask LOO 1-NN) ===")
print(f"original  41 clips:  acc {acc0:.3f}")
print(f"dedup     38 clips:  acc {acc1:.3f}   (testable-only, styles w/ >=2 real clips: {acc1t:.3f})")
print(f"\n{'style':22s} {'n->n':>7s}  {'recall0':>7s} {'recall1':>7s}")
for s in sorted(set(styles)):
    c0, c1 = cnt0[s], cnt1.get(s, 0)
    r0 = per0[s]; r1 = per1.get(s)
    note = "  SINGLETON after dedup — untestable" if c1 < 2 else ""
    print(f"{s:22s} {c0:2d}->{c1:<2d}   {r0 if r0 is None else f'{r0:7.2f}':>7} "
          f"{r1 if r1 is None else f'{r1:7.2f}':>7}{note}")

# ---- visual proof montage ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mid_frame(nm):
    with av.open(str(paths[idx[nm]])) as c:
        fr = [f.to_image() for f in c.decode(c.streams.video[0])]
    im = fr[len(fr) // 2]; h = 170; w = round(im.width * h / im.height)
    return np.asarray(im.resize((w, h), Image.BILINEAR)), (im.width, im.height)


fig, axes = plt.subplots(len(CLUSTERS), 2, figsize=(6.5, 2.3 * len(CLUSTERS)))
for r, c in enumerate(CLUSTERS):
    for k, role in enumerate(("keep", "drop")):
        nm = f"{c['style']}/{c[role]}"
        img, (W, H) = mid_frame(nm)
        axes[r][k].imshow(img); axes[r][k].axis("off")
        axes[r][k].set_title(f"{'KEEP' if role=='keep' else 'DROP'}  {c[role]}  ({W}x{H})",
                             fontsize=9, color=("green" if role == "keep" else "crimson"))
    axes[r][0].text(-0.04, 0.5, f"aligned cos {c['aligned_cos']:.3f}\ndHash Ham {c['dhash_hamming']:.0f}",
                    transform=axes[r][0].transAxes, ha="right", va="center",
                    fontsize=8, family="monospace")
fig.suptitle("Near-duplicate reference clips (same content, two resolutions)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "duplicates.png", dpi=120, bbox_inches="tight")

manifest = {"method": "aligned DINO cosine + independent dHash Hamming; see detect_duplicates.py",
            "clusters": CLUSTERS, "drop_from_exam": sorted(DROP),
            "corrected_exam": {"acc_41": acc0, "acc_38_dedup": acc1,
                               "acc_38_testable_only": acc1t}}
(OUT / "duplicates.json").write_text(json.dumps(manifest, indent=2))
print(f"\n[done] {OUT}/duplicates.png + duplicates.json")
