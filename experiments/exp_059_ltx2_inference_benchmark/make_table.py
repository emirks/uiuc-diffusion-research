"""Build the granular markdown table from exp_059 timings.json files (CPU-only)."""

import argparse
import json
import statistics
from pathlib import Path

ARM_ORDER = [
    "dev_720p_eager", "dev_1080p_eager", "dist_720p_eager", "dist_1080p_eager",
    "dev_720p_compile", "dev_1080p_compile", "dist_720p_compile", "dist_1080p_compile",
    "dist_720p_registry", "dist_1080p_registry",
]
# section keys in display order; diffusion sections get stage suffixes by order
ROWS = [
    "prompt_encode", "s1_transformer_build", "s1_denoise", "s1_transformer_free",
    "upsample", "s2_transformer_build", "s2_denoise", "s2_transformer_free",
    "audio_decode", "vae_decode", "mux", "TOTAL",
]


def flatten(call: dict) -> dict:
    out = {}
    diff_idx = 0
    for s in call["sections"]:
        n, v = s["name"], s["seconds"]
        if n == "transformer_build":
            out[f"s{diff_idx + 1}_transformer_build"] = out.get(f"s{diff_idx + 1}_transformer_build", 0) + v
        elif n == "denoise":
            diff_idx += 1
            out[f"s{diff_idx}_denoise"] = v
            out[f"s{diff_idx}_steps"] = s.get("steps")
            out[f"s{diff_idx}_res"] = f"{s.get('width')}x{s.get('height')}"
        elif n == "transformer_free":
            out[f"s{diff_idx}_transformer_free"] = v
        else:
            out[n] = out.get(n, 0) + v
    out["TOTAL"] = call["total_s"]
    return out


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    results = {}
    for arm in ARM_ORDER:
        p = run_dir / arm / "timings.json"
        pp = run_dir / arm / "timings.partial.json"
        if p.exists() or pp.exists():
            results[arm] = json.loads((p if p.exists() else pp).read_text())

    if not results:
        print("no results found")
        return

    meta = next(iter(results.values()))["meta"]
    print(f"GPU: {meta['gpu']}  |  node(s): "
          + ", ".join(sorted({r['meta']['node'] for r in results.values()}))
          + f"  |  torch {meta['torch']} cu{meta['cuda']}\n")

    header = ["section (s)"] + [
        f"{a} {'COLD' if k == 'cold' else 'WARM'}"
        for a in results
        for k in ("cold", "warm")
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    cells = {}
    for arm, res in results.items():
        cold = [flatten(c) for c in res["calls"] if c["kind"] == "cold"]
        warm = [flatten(c) for c in res["calls"] if c["kind"] == "warm"]
        for row in ROWS:
            cv = cold[0].get(row) if cold else None
            wvs = [w[row] for w in warm if row in w]
            wv = statistics.mean(wvs) if wvs else None
            cells[(arm, "cold", row)] = cv
            cells[(arm, "warm", row)] = wv

    for row in ROWS:
        line = [row]
        for arm in results:
            line.append(fmt(cells[(arm, "cold", row)]))
            line.append(fmt(cells[(arm, "warm", row)]))
        lines.append("| " + " | ".join(line) + " |")

    print("\n".join(lines))

    print("\nPer-call detail:")
    for arm, res in results.items():
        print(f"\n== {arm} (init {res['pipeline_init_s']}s) ==")
        for c in res["calls"]:
            f = flatten(c)
            stages = f" s1={f.get('s1_steps')}st@{f.get('s1_res')} s2={f.get('s2_steps')}st@{f.get('s2_res')}"
            print(f"  {c['kind']:4s} seed={c['seed']} total={c['total_s']:8.1f}s "
                  f"peakVRAM={c['peak_vram_alloc_gb']}G/{c['peak_vram_reserved_gb']}G{stages}")


if __name__ == "__main__":
    main()
