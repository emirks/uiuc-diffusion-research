"""exp_081 — isolate the cost of the one-way reference attention mask.

The full 30-step training probes are the authoritative measurement, but they need an H100 and
are stuck behind fairshare. This isolates the ONLY thing that differs between the two arms:
the self-attention call. Everything else in the step -- patchify, AdaLN, cross-attention to
text, FFN, optimizer -- is bitwise identical whether or not a mask is installed, so the
end-to-end slowdown is bounded above by the slowdown measured here.

Shapes are the real ones for the ctt_v2 retrain:
    T = 9,600 tokens  (4,800 target + 4,800 reference, from latents (128,16,20,15) -> 16*20*15)
    32 heads x 128 head_dim  (LTX-2 19B, model_configurator defaults)
    bf16, batch 1

Three configurations, forward AND backward (training is what matters):
    none      - no mask. What the bidirectional incumbent does.
    dense     - the (1,1,T,T) additive bias the one-way mask produces. What ships if <=30% slower.
    split     - two unmasked calls: target rows over all T keys, reference rows over the 4,800
                reference keys only. Semantically identical to `dense`, 25% FEWER attention
                pairs than `none`, and keeps the flash path. The fallback if dense is too slow.

The `split` timing is indicative only: shipping it would require the numerical-equivalence gate
the advisor pre-registered, which this script does NOT substitute for.

Run:  python bench_attention_mask.py
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

N_TGT = 4800
N_REF = 4800
HEADS = 32
HEAD_DIM = 128


def make_qkv(total: int, device, dtype):
    g = torch.Generator(device=device).manual_seed(0)
    shape = (1, HEADS, total, HEAD_DIM)
    return [
        torch.randn(shape, device=device, dtype=dtype, generator=g).requires_grad_(True)
        for _ in range(3)
    ]


def build_bias(total: int, ref_first: bool, device, dtype) -> torch.Tensor:
    """The additive log-space bias TransformerArgs._prepare_self_attention_mask produces."""
    mask = torch.ones((1, total, total), device=device, dtype=dtype)
    if ref_first:
        ref, noisy = slice(0, N_REF), slice(N_REF, total)
    else:
        noisy, ref = slice(0, N_TGT), slice(N_TGT, total)
    mask[:, ref, noisy] = 0.0
    finfo = torch.finfo(dtype)
    bias = torch.full_like(mask, finfo.min, dtype=dtype)
    bias[mask > 0] = 0.0
    return bias.unsqueeze(1)  # (1, 1, T, T)


def run_none(q, k, v, bias):
    return F.scaled_dot_product_attention(q, k, v)


def run_dense(q, k, v, bias):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


def run_split(q, k, v, bias):
    """[ref | noisy] layout: ref rows see only ref keys; noisy rows see everything."""
    ref_out = F.scaled_dot_product_attention(
        q[:, :, :N_REF], k[:, :, :N_REF], v[:, :, :N_REF]
    )
    tgt_out = F.scaled_dot_product_attention(q[:, :, N_REF:], k, v)
    return torch.cat([ref_out, tgt_out], dim=2)


def bench(fn, q, k, v, bias, iters: int, backward: bool) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # warmup
    for _ in range(3):
        out = fn(q, k, v, bias)
        if backward:
            out.sum().backward()
            for t in (q, k, v):
                t.grad = None
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(q, k, v, bias)
        if backward:
            out.sum().backward()
            for t in (q, k, v):
                t.grad = None
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak_gb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    total = N_TGT + N_REF

    print(f"[bench] {torch.cuda.get_device_name(0)}")
    print(f"[bench] T={total} ({N_REF} ref + {N_TGT} target)  heads={HEADS} dim={HEAD_DIM} {dtype}")

    q, k, v = make_qkv(total, device, dtype)
    bias = build_bias(total, ref_first=True, device=device, dtype=dtype)
    print(f"[bench] dense bias tensor: {tuple(bias.shape)} = "
          f"{bias.numel() * bias.element_size() / 1024**3:.3f} GiB")

    results = {}
    for phase, backward in (("forward", False), ("fwd+bwd", True)):
        print(f"\n--- {phase} ---")
        base = None
        for name, fn in (("none", run_none), ("dense", run_dense), ("split", run_split)):
            try:
                ms, peak = bench(fn, q, k, v, bias, args.iters, backward)
            except torch.OutOfMemoryError:
                print(f"  {name:6s}  OOM")
                results[f"{phase}/{name}"] = {"oom": True}
                torch.cuda.empty_cache()
                continue
            if base is None:
                base = ms
            print(f"  {name:6s}  {ms:8.2f} ms   peak {peak:6.2f} GiB   x{ms / base:.2f} vs none")
            results[f"{phase}/{name}"] = {"ms": round(ms, 3), "peak_gb": round(peak, 3),
                                          "ratio_vs_none": round(ms / base, 4)}
            torch.cuda.empty_cache()

    # The pre-registered decision rule (advisor round 6).
    d = results.get("fwd+bwd/dense", {})
    n = results.get("fwd+bwd/none", {})
    if "ms" in d and "ms" in n:
        slow = d["ms"] / n["ms"] - 1.0
        print(f"\n[bench] attention-only fwd+bwd slowdown of dense mask: {slow * 100:+.1f}%")
        print(f"[bench] this BOUNDS the end-to-end step slowdown from above "
              f"(the rest of the step is identical)")
        print(f"[bench] pre-registered rule: <=30% end-to-end -> ship dense")
    print("\n" + json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
