"""Static PNG figures for a finished certification run (representation only).

Reads record.json + analysis/analysis.json and writes <cert_dir>/figures/*.png:
bar verdicts, per-metric exam accuracy vs chance and the bar-1 floor, one
row-normalized confusion heatmap per metric (stratum-ordered), R1 accuracy per
tag group, and per-clip margin distributions for R1 and R2. Never feeds a
verdict; the driver treats any failure here as non-gating. Manual rerun:

    PYTHONPATH=src python -m diffusion.transition_eval.certify.figures \
        --cert-dir outputs/eval/certification/<version>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, MUTED, LINE = "#21262B", "#64707A", "#E1E3DD"
ACC, OK, BAD = "#3D5A9E", "#2F7A3B", "#B5472E"
CMAP = LinearSegmentedColormap.from_list("seq", ["#F1F2EE", "#2C4770"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "savefig.dpi": 150,
})


def _class_order(clips: list[dict]) -> list[str]:
    """Stratum ordering used everywhere: two-sided, one-sided, one-sided+camera."""
    side, cam = {}, {}
    for c in clips:
        side[c["class"]] = c["sidedness"]
        cam[c["class"]] = cam.get(c["class"], False) or ("camera" in c["tags"])

    def stratum(c):
        if side[c] == "twosided":
            return 0
        return 2 if cam[c] else 1
    return sorted(side, key=lambda c: (stratum(c), c))


def fig_bars(record: dict, path: pathlib.Path) -> None:
    verdicts = record["verdicts"]
    names = list(verdicts)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, name in enumerate(names):
        ok = verdicts[name]
        y = len(names) - 1 - i
        ax.barh(y, 1, height=0.62, color=(OK if ok else BAD), alpha=0.18)
        ax.text(0.02, y, name, va="center", fontsize=9, color=INK)
        ax.text(0.98, y, "PASS" if ok else "FAIL", va="center", ha="right",
                fontsize=9, fontweight="bold", color=(OK if ok else BAD))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.axis("off")
    overall = record["overall_pass"]
    ax.set_title(f"{record['version']} — overall "
                 f"{'PASS' if overall else 'FAIL'}", loc="left",
                 color=(OK if overall else BAD))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_exam_accuracy(ana: dict, record: dict, path: pathlib.Path) -> None:
    names = list(ana["metrics"])
    accs = [ana["metrics"][m]["retrieval"]["accuracy_1nn"] for m in names]
    chance = ana["metrics"][names[0]]["retrieval"]["chance"]
    floor = record["exam"]["bar1"]["acc_min"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ys = range(len(names) - 1, -1, -1)
    ax.axvline(chance, color=MUTED, lw=1, ls=":")
    ax.axvline(floor, color=BAD, lw=1, ls="--")
    ax.text(chance, len(names) - 0.3, f" chance {chance:.3f}", fontsize=8, color=MUTED)
    ax.text(floor, len(names) - 0.3, f" bar-1 floor {floor:.2f}", fontsize=8, color=BAD)
    ax.scatter(accs, list(ys), s=46, color=ACC, zorder=3)
    for a, y in zip(accs, ys):
        ax.text(a, y - 0.38, f"{a:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys), names)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(names))
    ax.set_xlabel("R1 LOO 1-NN accuracy")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Exam accuracy per metric", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(ana: dict, metric: str, path: pathlib.Path) -> None:
    order = _class_order(ana["clips"])
    idx = {c: i for i, c in enumerate(order)}
    n = len(order)
    conf = ana["metrics"][metric]["retrieval"]["confusion"]
    M = np.zeros((n, n))
    for a, row in conf.items():
        for b, cnt in row.items():
            M[idx[a], idx[b]] = cnt
    row_n = M.sum(axis=1, keepdims=True)
    R = np.divide(M, row_n, out=np.zeros_like(M), where=row_n > 0)
    acc = ana["metrics"][metric]["retrieval"]["accuracy_1nn"]
    fig, ax = plt.subplots(figsize=(8.6, 8))
    ax.imshow(R, cmap=CMAP, vmin=0, vmax=1)
    n_two = sum(1 for c in order if _side_of(ana["clips"], c) == "twosided")
    n_cam = sum(1 for c in order if _side_of(ana["clips"], c) != "twosided"
                and _cam_of(ana["clips"], c))
    for cut in (n_two - 0.5, n - n_cam - 0.5):
        ax.axhline(cut, color=BAD, lw=0.7, alpha=0.6)
        ax.axvline(cut, color=BAD, lw=0.7, alpha=0.6)
    ax.set_xticks(range(n), order, rotation=90, fontsize=5.5)
    ax.set_yticks(range(n), order, fontsize=5.5)
    ax.set_xlabel("predicted (1-NN)")
    ax.set_ylabel("true")
    ax.set_title(f"{metric} — row-normalized confusion (acc {acc:.3f})\n"
                 "separators: two-sided / one-sided / one-sided+camera",
                 loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _side_of(clips, cls):
    return next(c["sidedness"] for c in clips if c["class"] == cls)


def _cam_of(clips, cls):
    return any("camera" in c["tags"] for c in clips if c["class"] == cls)


def fig_tag_accuracy(ana: dict, path: pathlib.Path) -> None:
    bt = ana["by_tag"]
    rows = bt["coarse"] + bt["patterns"]
    metrics = list(ana["metrics"])
    A = np.array([[np.nan if r.get(m) is None else r[m] for m in metrics]
                  for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.6))
    ax.imshow(np.nan_to_num(A), cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(metrics)):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=("white" if A[i, j] > 0.55 else INK))
    ax.set_xticks(range(len(metrics)), metrics, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)),
                  [f"{r['group']}  (n={r['n']})" for r in rows], fontsize=8)
    ax.axhline(len(bt["coarse"]) - 0.5, color=BAD, lw=0.8)
    ax.set_title("R1 accuracy by tag group\n"
                 "coarse pools above the line, exact patterns below",
                 loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_margins(ana: dict, path: pathlib.Path) -> None:
    metrics = list(ana["metrics"])
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharex=False)
    for ax, m in zip(axes.flat, metrics):
        vals = [r["margin"] for r in ana["metrics"][m]["rows"]
                if r.get("margin") is not None]
        ax.hist(vals, bins=30, color=ACC, alpha=0.85)
        ax.axvline(0, color=BAD, lw=1, ls="--")
        neg = sum(1 for v in vals if v <= 0)
        ax.set_title(f"{m}  ({neg}/{len(vals)} clips ≤ 0)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-clip R1 margin (nearest cross-class − nearest within-class "
                 "distance; ≤ 0 ⇒ misretrieved)", x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def fig_r2_margins(ana: dict, path: pathlib.Path) -> None:
    rows = [r for r in ana["r2"]["rows"] if r.get("margin") is not None]
    vals = [r["margin"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.hist(vals, bins=40, color=ACC, alpha=0.85)
    ax.axvline(0, color=BAD, lw=1, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("pool intrusion margin (> 0 ⇒ correct class pool)")
    ax.set_title(f"R2 pool margins under {ana['r2']['winner_mask']} "
                 f"(acc {ana['r2']['accuracy']:.3f}, n={ana['r2']['n_graded']})",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_all(cert_dir: pathlib.Path) -> list[pathlib.Path]:
    cert = pathlib.Path(cert_dir)
    record = json.load(open(cert / "record.json"))
    ana = json.load(open(cert / "analysis" / "analysis.json"))
    if "by_tag" not in ana:      # analysis written before exam persisted tags
        from . import diagnostics
        ana["by_tag"] = diagnostics.tag_accuracy(
            {m: ana["metrics"][m]["rows"] for m in ana["metrics"]}, ana["clips"])
    out = cert / "figures"
    out.mkdir(exist_ok=True)
    done = []

    def save(fn, name, *a):
        p = out / name
        fn(*a, p)
        done.append(p)
    save(fig_bars, "bars.png", record)
    save(fig_exam_accuracy, "exam_accuracy.png", ana, record)
    for m in ana["metrics"]:
        save(fig_confusion, f"confusion__{m}.png", ana, m)
    save(fig_tag_accuracy, "tag_accuracy.png", ana)
    save(fig_margins, "r1_margins.png", ana)
    save(fig_r2_margins, "r2_margins.png", ana)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-dir", required=True)
    args = ap.parse_args()
    for p in save_all(pathlib.Path(args.cert_dir)):
        print(f"[figures] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
