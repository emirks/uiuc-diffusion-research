"""Build `pair_examples.ipynb` — a self-contained, re-runnable notebook that
shows how the harness turns 41 clips into pairwise metrics and the LOO 1-NN
exam, then displays ~30 diverse example pairs with their core-mask vs
all-frames similarities.

No Jupyter kernel is needed: this builder execs each code cell in-process
(plain `exec`, shared namespace), captures stdout + matplotlib figures, and
writes the .ipynb JSON with outputs pre-embedded so it renders on open. The
cell sources are the single source of truth and are fully runnable.

Run:  $LAB/envs/diffusion/bin/python .../build_pair_examples_notebook.py
"""

import base64
import io
import json
import pathlib
import sys
import contextlib

REPO = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
NB_PATH = REPO / "experiments/exp_053_eval_harness_v2/pair_examples.ipynb"
OUT_DIR = REPO / "outputs/eval/exp_053/pair_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Notebook cells. (md, source) or (code, source). Code cells share one ns.
# --------------------------------------------------------------------------- #

MD_INTRO = r"""# exp_053 — Transition pairs & the appearance exam, by example

This notebook makes the harness's core-frame appearance metric (**M3**) concrete:
how the 41 reference clips become pairwise similarities, how the leave-one-out
retrieval **exam** turns those into the headline accuracy, and what ~30 real
pairs look like — including the six cases where the **core mask** rescues a
retrieval that all-frames similarity gets wrong.

Everything runs on cached DINOv2 features (no GPU); the montages decode a few
frames per clip for display only.
"""

MD_METHOD = r"""## How pairs and metrics are built

**1 clip → a feature set.** Each clip is decoded to frames and embedded with
DINOv2 (one L2-normalized vector per frame), so clip *i* is a set `F_i` of `T_i`
frame vectors. The **core mask** (from the morph profile M1) keeps only the
"neither-endpoint" frames — the effect medium — so we compare `F_i[core]` for
the core-mask variant vs the full `F_i` for the all-frames variant.

**Pair similarity — `set_similarity`.** For two clips we score set-to-set with
*symmetric mean-of-max cosine*:

```
S = F_i @ F_j.T                     # cosine, features are unit-norm
sim(i,j) = 0.5 * ( mean_r max_c S[r,c] + mean_c max_r S[r,c] )
```

i.e. every frame of *i* finds its best match in *j* (and vice-versa), then
average. One scalar per pair.

**41 clips → 820 pairs.** We fill a symmetric `41×41` matrix `S[i,j]` for every
unordered pair — `C(41,2) = 820` off-diagonal entries (the diagonal, a clip vs
itself, is excluded). This is the "pair it with every other clip, same-style and
different-style" step. What we do *with* those pairs depends on the metric — and
there are **two different aggregations**, which is the crux of your question:

**(A) The exam — leave-one-out 1-NN (the headline 0.927).** This is **not** an
average over pairs. For each clip we take its single **nearest** other clip
(`argmax_j S[i,j]`), predict *that* neighbor's style, and score it right iff the
neighbor shares the clip's style. **Accuracy** = fraction of the 41 clips whose
top-1 neighbor matched. **Per-class recall** (the per-style table) = that
fraction restricted to one style's clips. Only the *best* match per clip counts.

**(B) Separation / motion sanity — mean over the axis.** Cohen's *d* and the
motion within-vs-cross check *do* average: mean `S` over all same-style pairs vs
mean `S` over all different-style pairs. This is the "average over that axis"
mode; it measures how far apart the two similarity *distributions* sit, and
complements (A) because 1-NN accuracy saturates at 1.0 long before the margin does.

So: **one matrix, two readouts** — retrieval uses the max per row (decision
quality); separation uses the means (distributional headroom).
"""

CODE_SETUP = r'''import hashlib, pathlib
import numpy as np, yaml

REPO = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
cfg = yaml.safe_load((REPO / "experiments/exp_053_eval_harness_v2/config.yaml").read_text())
MODEL, SHORT = cfg["features"]["model"], str(cfg["features"]["short_side"])
CACHE = REPO / cfg["features"]["cache_dir"]
NP, NS = cfg["morph"]["n_prefix"], cfg["morph"]["n_suffix"]

# --- pure-numpy reimplementation of the pieces we need (no torch) ----------
def file_key(path, *parts):
    st = pathlib.Path(path).stat()
    raw = "|".join([str(pathlib.Path(path).resolve()), str(st.st_mtime_ns),
                    str(st.st_size), *parts])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def cache_npz(vid):                     # process_video_file caches under a double hash
    inner = file_key(vid, MODEL, SHORT)
    return CACHE / f"dino_arr_{hashlib.sha1(inner.encode()).hexdigest()[:16]}.npz"

def set_similarity(F1, F2):
    S = F1 @ F2.T
    return float(0.5 * (S.max(1).mean() + S.max(0).mean()))

def morph_env(feats):                   # endpoint envelope max(a_hat, b_hat)
    eA = feats[:NP].mean(0); eA /= np.linalg.norm(eA) + 1e-12
    eB = feats[-NS:].mean(0); eB /= np.linalg.norm(eB) + 1e-12
    cross = float(eA @ eB); denom = max(1 - cross, 1e-6)
    ah = np.clip((feats @ eA - cross) / denom, -0.25, 1.25)
    bh = np.clip((feats @ eB - cross) / denom, -0.25, 1.25)
    return np.maximum(ah, bh)

def core_mask(env):
    T = len(env); m = env < 0.5
    m[:NP] = False; m[T - NS:] = False
    if not m.any():
        me = env.copy(); me[:NP] = np.inf; me[T - NS:] = np.inf
        m[int(np.argmin(me))] = True
    return m

def scan(root, exclude=("higgsfield",)):
    return {d.name: sorted(d.glob("*.mp4")) for d in sorted(pathlib.Path(root).iterdir())
            if d.is_dir() and d.name not in exclude and any(d.glob("*.mp4"))}

refs = scan(REPO / cfg["data"]["transitions_root"], tuple(cfg["data"]["exclude"]))
names, styles, paths, feats_all, feats_core, core_idx = [], [], [], [], [], {}
skipped = []
for style, vids in refs.items():
    for vid in vids:
        npz = cache_npz(vid)
        if not npz.exists():                 # new/unprocessed style: no cached features yet
            skipped.append(f"{style}/{vid.stem}"); continue
        f = np.load(npz)["feats"]
        env = morph_env(f); m = core_mask(env)
        nm = f"{style}/{vid.stem}"
        names.append(nm); styles.append(style); paths.append(vid)
        feats_all.append(f); feats_core.append(f[m])
        mid = env.copy(); mid[~m] = np.inf; core_idx[nm] = int(np.argmin(mid))
n = len(names); styles = np.array(styles)
idx = {nm: i for i, nm in enumerate(names)}
print(f"{len(set(styles))} styles, {n} clips, {n*(n-1)//2} unordered pairs")
if skipped:
    print(f"[note] excluded {len(skipped)} clip(s) with no cached features yet "
          f"(new/unprocessed): {', '.join(skipped)}")
'''

CODE_MATRICES = r'''# Build both 41x41 symmetric similarity matrices.
Sc = np.zeros((n, n)); Sa = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        Sc[i, j] = Sc[j, i] = set_similarity(feats_core[i], feats_core[j])
        Sa[i, j] = Sa[j, i] = set_similarity(feats_all[i], feats_all[j])
np.fill_diagonal(Sc, -np.inf); np.fill_diagonal(Sa, -np.inf)

def exam(S):
    nn = S.argmax(1); pred = styles[nn]; ok = pred == styles
    per = {s: float((pred[styles == s] == s).mean()) for s in sorted(set(styles))}
    return nn, pred, ok, per

nn_c, pred_c, ok_c, per_c = exam(Sc)
nn_a, pred_a, ok_a, per_a = exam(Sa)

def rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
iu = np.triu_indices(n, 1)
rho = float(np.corrcoef(rank(Sc[iu]), rank(Sa[iu]))[0, 1])

print(f"LOO 1-NN accuracy   core-mask {ok_c.mean():.3f}   all-frames {ok_a.mean():.3f}")
print(f"{len(iu[0])}-pair rank corr (core vs all)   Spearman rho = {rho:.3f}\n")
print(f"{'style':22s} {'core':>5s} {'all':>5s}   per-class recall")
for s in sorted(set(styles)):
    tag = "" if abs(per_c[s]-per_a[s]) < 1e-9 else "  <- mask changes it"
    print(f"{s:22s} {per_c[s]:5.2f} {per_a[s]:5.2f}{tag}")

flips = [i for i in range(n) if ok_c[i] and not ok_a[i]]
print(f"\n{len(flips)} clips: correct under core-mask, WRONG under all-frames (the mask saves them)")
'''

CODE_HELPERS = r'''# Decode a few display frames per clip (cached in-memory) and a montage helper.
import av
from PIL import Image
import matplotlib.pyplot as plt

def _thumb(arr, h=150):
    im = Image.fromarray(arr); w = round(im.width * h / im.height)
    return np.asarray(im.resize((w, h), Image.BILINEAR))

_repcache = {}
def reps(nm):
    """(start, core, end) display thumbnails for a clip."""
    if nm in _repcache: return _repcache[nm]
    frames = []
    with av.open(str(paths[idx[nm]])) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            frames.append(np.asarray(fr.to_image(), np.uint8))
    T = len(frames); ci = min(core_idx[nm], T - 1)
    out = {"start": _thumb(frames[min(2, T-1)]), "core": _thumb(frames[ci]),
           "end": _thumb(frames[max(0, T-3)])}
    _repcache[nm] = out; return out

def show_pairs(entries, title):
    """entries: list of dict(a, b, core_sim, all_sim, note, color)."""
    N = len(entries)
    fig, axes = plt.subplots(N, 3, figsize=(9.2, 1.7 * N),
                             gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    if N == 1: axes = axes[None, :]
    for r, e in enumerate(entries):
        tax, ax_a, ax_b = axes[r]
        tax.axis("off")
        txt = (f"{e['note']}\n\nA: {e['a']}\nB: {e['b']}\n\n"
               f"core-mask sim = {e['core_sim']:.3f}\nall-frames sim = {e['all_sim']:.3f}")
        tax.text(0.0, 0.5, txt, va="center", ha="left", fontsize=8.5,
                 family="monospace", color=e["color"])
        for ax, key in ((ax_a, "a"), (ax_b, "b")):
            ax.imshow(reps(e[key])["core"]); ax.axis("off")
            ax.set_title(e[key].split("/")[0], fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.0, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return fig
'''

CODE_DILUTION = r'''# Why the mask helps: endpoints are generic scenery; the core frame is the effect.
FIGTITLE = "dilution"
demo = [names[i] for i in flips][:2] or [names[0]]
fig, axes = plt.subplots(len(demo), 3, figsize=(7.5, 2.4 * len(demo)))
if len(demo) == 1: axes = axes[None, :]
for r, nm in enumerate(demo):
    R = reps(nm)
    for ax, key, lab in ((axes[r][0], "start", "endpoint A\n(generic — dilutes all-frames)"),
                         (axes[r][1], "core",  "CORE frame\n(effect medium — the mask keeps this)"),
                         (axes[r][2], "end",   "endpoint B\n(generic — dilutes all-frames)")):
        ax.imshow(R[key]); ax.axis("off")
        ax.set_title(lab, fontsize=8.5, color=("green" if key == "core" else "gray"))
    axes[r][0].set_ylabel(nm.split("/")[0], fontsize=9)
fig.suptitle("Core mask isolates the effect from generic endpoint scenery",
             fontsize=12, fontweight="bold")
fig.tight_layout()
'''

CODE_CEILING = r'''# Category 1 — same-style "ceiling" pairs (the strongest same-style match per style).
FIGTITLE = "ceiling"
ent = []
for s in sorted(set(styles)):
    ids = [i for i in range(n) if styles[i] == s]
    best = max(((i, j) for a, i in enumerate(ids) for j in ids[a+1:]),
              key=lambda p: Sc[p[0], p[1]], default=None)
    if best is None: continue
    i, j = best
    ent.append(dict(a=names[i], b=names[j], core_sim=Sc[i, j], all_sim=Sa[i, j],
                    color="green", note=f"SAME style ({s})\nstrongest in-style pair"))
show_pairs(ent, "Category 1 — same-style ceiling pairs (high core-mask similarity)")
'''

CODE_FLIPS = r'''# Category 2 — the mask saves the retrieval: query vs core-1NN (right) and all-1NN (wrong).
FIGTITLE = "flips"
ent = []
for i in flips:
    jc, ja = nn_c[i], nn_a[i]
    ent.append(dict(a=names[i], b=names[jc], core_sim=Sc[i, jc], all_sim=Sa[i, jc],
                    color="green",
                    note=f"core-mask 1-NN  ✓\nsame style ({styles[i]})"))
    ent.append(dict(a=names[i], b=names[ja], core_sim=Sc[i, ja], all_sim=Sa[i, ja],
                    color="crimson",
                    note=f"all-frames 1-NN  ✗\nwrong style ({styles[ja]})"))
show_pairs(ent, "Category 2 — where the core mask rescues the retrieval (6 clips × 2 candidates)")
'''

CODE_NEG = r'''# Category 3 — clear cross-style negatives (low sim) + genuine core-mask confusions.
FIGTITLE = "negatives"
ent, seen = [], set()
order = sorted(((Sc[i, j], i, j) for i in range(n) for j in range(i+1, n)
                if styles[i] != styles[j]), key=lambda t: t[0])
for _, i, j in order:                       # lowest-similarity cross-style, deduped by style-pair
    k = tuple(sorted((styles[i], styles[j])))
    if k in seen: continue
    seen.add(k)
    ent.append(dict(a=names[i], b=names[j], core_sim=Sc[i, j], all_sim=Sa[i, j],
                    color="dimgray", note="DIFFERENT styles\nclear negative (low sim)"))
    if len(ent) >= 6: break
for i in range(n):                          # honest failures: core-mask retrieves wrong style
    if not ok_c[i]:
        j = nn_c[i]
        ent.append(dict(a=names[i], b=names[j], core_sim=Sc[i, j], all_sim=Sa[i, j],
                        color="darkorange",
                        note=f"core-mask CONFUSION\nretrieved {styles[j]}\n(true {styles[i]})"))
show_pairs(ent, "Category 3 — clear negatives + honest core-mask confusions")
'''

MD_TAKEAWAY = r"""## Reading guide

- **Category 1** shows what "same style" means to M3 — the effect medium looks
  alike across totally different scenes; core-mask similarity is high.
- **Category 2** is the whole argument for the mask: for these six clips the
  *nearest all-frames neighbor is the wrong style* (endpoints dominate), while
  the *nearest core-mask neighbor is the right style*. The mask converts 6
  wrong answers into 6 right ones and costs zero (no clip flips the other way).
- **Category 3** shows the floor (unrelated styles score low) and the metric's
  honest failure cases — the handful of clips even the core mask retrieves
  wrongly (concentrated in `flying_cam`, whose exam recall is 0.6).

Aggregation reminder: the accuracy in the exam cell uses **max per row**
(1-NN), while separation/motion checks use **means over same vs different** —
same 820-pair matrix, two different readouts.
"""

CELLS = [
    ("md", MD_INTRO), ("md", MD_METHOD),
    ("code", CODE_SETUP), ("code", CODE_MATRICES), ("code", CODE_HELPERS),
    ("code", CODE_DILUTION), ("code", CODE_CEILING), ("code", CODE_FLIPS),
    ("code", CODE_NEG), ("md", MD_TAKEAWAY),
]

# --------------------------------------------------------------------------- #
#  Exec code cells, capture stdout + figures, assemble ipynb.
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def src_lines(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


ns = {}
nb_cells = []
exec_count = 0
for kind, text in CELLS:
    if kind == "md":
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src_lines(text)})
        continue
    exec_count += 1
    outputs = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(text.strip("\n"), "<cell>", "exec"), ns)
    out_text = buf.getvalue()
    if out_text:
        outputs.append({"output_type": "stream", "name": "stdout",
                        "text": [l + "\n" for l in out_text.rstrip("\n").split("\n")]})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=115, bbox_inches="tight")
        png = b.getvalue()
        title = ns.get("FIGTITLE", f"fig{exec_count}")
        (OUT_DIR / f"{title}.png").write_bytes(png)
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(png).decode(),
                                 "text/plain": ["<Figure>"]}})
        plt.close(fig)
    ns.pop("FIGTITLE", None)
    nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": exec_count,
                     "outputs": outputs, "source": src_lines(text)})
    print(f"[cell {exec_count}] {len(outputs)} output(s)", file=sys.stderr)

nb = {"cells": nb_cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"[done] wrote {NB_PATH}", file=sys.stderr)
print(f"[done] montages in {OUT_DIR}", file=sys.stderr)
