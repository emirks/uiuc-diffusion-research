"""
Per-latent-frame round-trip error localization across exp_030/032/033.

Loads z0.pt and z0_recon.pt for each clip in each run, reshapes the packed
(1, 5632, 128) tensor to (1, F_lat=16, H_lat=22, W_lat=16, C=128), and
computes per-latent-frame sum-of-squared error between z0 and z0_recon.

Conditioning layout (from inv_meta clip_conditioning: num_clip_frames=25,
end_index=12):
  - Start sub-clip occupies latent frames {0..3}
  - Middle (free under all recipes) latent frames {4..11}
  - End sub-clip occupies latent frames {12..15}

Per-recipe conditioned masks:
  - exp_030 (vanilla sub-clip anchors):   cond = {0..3, 12..15}, free = {4..11}
  - exp_032 (full z0 slices, LEAKY):      cond = {0..3, 12..15}, free = {4..11}
  - exp_033 (sub-clip anchors + drop1):   cond = {0..3, 13..15}, free = {4..12}

Output: scripts/anchor_error_localization.csv (per-clip per-frame error) and
prints a summary table to stdout.
"""

import csv
import pathlib
from typing import Dict, List, Tuple

import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

RUNS = {
    "exp_030": REPO_ROOT / "outputs/videos/exp_030_ltx2_rf_inv_real_clips/run_0001",
    "exp_032": REPO_ROOT / "outputs/videos/exp_032_ltx2_rf_inv_selfcond/run_0001",
    "exp_033": REPO_ROOT / "outputs/videos/exp_033_ltx2_rf_inv_drop1/run_0001",
}
CLIPS = [f"shadow_smoke_{i}" for i in range(10)]

F_LAT, C = 16, 128  # constant across runs (num_frames=121, channels)
SPATIAL_DIV = 32    # LTX-2 VAE spatial compression factor

# Conditioning layout (verified from inv_meta: end_index=12 across all runs)
START_FRAMES = list(range(0, 4))      # {0, 1, 2, 3}
MIDDLE_FRAMES = list(range(4, 12))    # {4, 5, ..., 11}
END_FRAMES = list(range(12, 16))      # {12, 13, 14, 15}

# Per-recipe free position masks (free = NOT conditioned, solver fills freely)
FREE_BY_RECIPE = {
    "exp_030": set(MIDDLE_FRAMES),
    "exp_032": set(MIDDLE_FRAMES),
    "exp_033": set(MIDDLE_FRAMES) | {12},
}


def read_latent_grid(clip_dir: pathlib.Path) -> Tuple[int, int]:
    """Read render_HxW from inv_meta and return (H_lat, W_lat)."""
    meta = yaml.safe_load((clip_dir / "inv_meta.yaml").read_text())
    H, W = meta["render_HxW"]
    return H // SPATIAL_DIV, W // SPATIAL_DIV


def load_packed(path: pathlib.Path, h_lat: int, w_lat: int) -> torch.Tensor:
    """Load packed latent and return float32 on CPU after shape check."""
    t = torch.load(path, map_location="cpu", weights_only=False)
    expected_tokens = F_LAT * h_lat * w_lat
    if t.shape != (1, expected_tokens, C):
        raise ValueError(
            f"shape {tuple(t.shape)} at {path}, expected (1, {expected_tokens}, {C})"
        )
    return t.float()


def per_frame_sqerr(
    z0: torch.Tensor, z0_recon: torch.Tensor, h_lat: int, w_lat: int
) -> torch.Tensor:
    """Return per-latent-frame sum-of-squared error, shape (F_lat,)."""
    diff = (z0 - z0_recon).reshape(1, F_LAT, h_lat, w_lat, C)
    sq = diff.pow(2).sum(dim=(2, 3, 4)).squeeze(0)
    return sq


def main():
    rows: List[Dict] = []
    per_clip_total: Dict[str, Dict[str, float]] = {r: {} for r in RUNS}
    per_clip_endsplit: Dict[str, Dict[str, List[float]]] = {r: {} for r in RUNS}

    for recipe, run_dir in RUNS.items():
        for clip in CLIPS:
            z0p = run_dir / clip / "z0.pt"
            zrp = run_dir / clip / "z0_recon.pt"
            if not z0p.exists() or not zrp.exists():
                print(f"  [warn] {recipe}/{clip}: missing tensors, skip")
                continue
            h_lat, w_lat = read_latent_grid(run_dir / clip)
            z0 = load_packed(z0p, h_lat, w_lat)
            zr = load_packed(zrp, h_lat, w_lat)
            per_frame = per_frame_sqerr(z0, zr, h_lat, w_lat).tolist()
            total = sum(per_frame)
            per_clip_total[recipe][clip] = total

            # Per-frame row
            for f_idx, sqerr in enumerate(per_frame):
                rows.append(
                    {
                        "recipe": recipe,
                        "clip": clip,
                        "latent_frame": f_idx,
                        "sqerr": sqerr,
                        "frac_of_clip_total": sqerr / total if total > 0 else 0.0,
                        "is_conditioned": int(f_idx not in FREE_BY_RECIPE[recipe]),
                        "region": (
                            "start" if f_idx in START_FRAMES
                            else "middle" if f_idx in MIDDLE_FRAMES
                            else "end"
                        ),
                    }
                )

            # End-region split for hypothesis test
            end_sq = [per_frame[f] for f in END_FRAMES]
            per_clip_endsplit[recipe][clip] = end_sq

    # Write CSV
    out_csv = REPO_ROOT / "scripts/anchor_error_localization.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "recipe", "clip", "latent_frame", "sqerr",
                "frac_of_clip_total", "is_conditioned", "region",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {out_csv}")

    # Summary: aggregate per-frame stats across clips, per recipe
    print()
    print("=" * 78)
    print("PER-RECIPE per-latent-frame mean sqerr (sum across clips ÷ #clips):")
    print("=" * 78)
    print(f"  {'frame':>5}  " + "  ".join(f"{r:>14}" for r in RUNS))
    for f_idx in range(F_LAT):
        vals = []
        for recipe in RUNS:
            xs = [
                row["sqerr"]
                for row in rows
                if row["recipe"] == recipe and row["latent_frame"] == f_idx
            ]
            vals.append(sum(xs) / max(len(xs), 1))
        marker = "  S" if f_idx in START_FRAMES else "  M" if f_idx in MIDDLE_FRAMES else "  E"
        print(f"  {f_idx:>5}{marker}  " + "  ".join(f"{v:>12.1f}" for v in vals))

    print()
    print("=" * 78)
    print("END-region (frames 12..15) sqerr split — fraction of end-total at each frame:")
    print("=" * 78)
    print(f"  {'recipe':>8}  {'clip':>16}  " + "  ".join(f"f={f}" for f in END_FRAMES))
    for recipe, clips in per_clip_endsplit.items():
        for clip, end_sq in clips.items():
            end_tot = sum(end_sq)
            fracs = [s / end_tot if end_tot > 0 else 0.0 for s in end_sq]
            print(
                f"  {recipe:>8}  {clip:>16}  "
                + "  ".join(f"{x:5.2f}" for x in fracs)
            )

    # Decision-rule numbers for the gate
    print()
    print("=" * 78)
    print("DECISION-RULE METRICS")
    print("=" * 78)
    for recipe in RUNS:
        # For exp_033, R1 says ≥70% of end-region cost should sit at frame 13
        # (because 12 was freed). For exp_030/032, the analogous "first
        # end-sub-clip cond frame" is 12.
        first_end = 13 if recipe == "exp_033" else 12
        # Where the actual mass lives
        per_clip_frac = []
        for clip, end_sq in per_clip_endsplit[recipe].items():
            end_tot = sum(end_sq)
            if end_tot <= 0:
                continue
            f_idx_in_end = END_FRAMES.index(first_end)
            per_clip_frac.append(end_sq[f_idx_in_end] / end_tot)
        if per_clip_frac:
            mean_frac = sum(per_clip_frac) / len(per_clip_frac)
            med_frac = sorted(per_clip_frac)[len(per_clip_frac) // 2]
            print(
                f"  {recipe}: median fraction of end-region sqerr at frame {first_end} "
                f"= {med_frac:.3f}  (mean {mean_frac:.3f}, N={len(per_clip_frac)})"
            )

    # Total mass split: start / middle / end per recipe
    print()
    print("=" * 78)
    print("TOTAL MASS SPLIT (start cond 0..3 / middle 4..11 / end cond 12..15):")
    print("=" * 78)
    for recipe in RUNS:
        s = m = e = 0.0
        for row in rows:
            if row["recipe"] != recipe:
                continue
            if row["region"] == "start":
                s += row["sqerr"]
            elif row["region"] == "middle":
                m += row["sqerr"]
            else:
                e += row["sqerr"]
        tot = s + m + e
        if tot > 0:
            print(
                f"  {recipe}: start={s/tot:5.2%}  middle={m/tot:5.2%}  end={e/tot:5.2%}"
                f"  (total sqerr = {tot:.1f})"
            )


if __name__ == "__main__":
    main()
