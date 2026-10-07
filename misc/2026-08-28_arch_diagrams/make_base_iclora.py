#!/usr/bin/env python
"""Generate base_iclora.svg (+ .png): the DEFAULT LTX-2 IC-LoRA training graph —
frozen base vs LoRA-adapted linears, one-way reference attention, per-token σ.

Facts verified 2026-08-28 against src/LTX-2-dualforce (ltx-core model.py /
transformer.py / attention.py / feed_forward.py / adaln.py / text_projection.py;
ltx-trainer flexible.py / trainer.py) and store/runs/012 (960 LoRA tensors).

Arm variants: copy this file, change the boxes — keep the grid and palette so
the figures read as one family.

Run:  $LAB/envs-aarch64/ltx2/bin/python make_base_iclora.py
"""
from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent
NAME = "base_iclora"
W, H = 1600, 870
FONT = "Helvetica Neue, Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
PNG_FONT = "DejaVu Sans"  # cairo's toy text API has no per-glyph fallback; DejaVu covers σ ε ⊙ ‖ →

C = dict(
    ink="#1F2937", muted="#6B7280", line="#374151", panel="#B4BAC6",
    frozen_f="#EEF1F5", frozen_s="#8B95A5",
    lora_f="#FDE7C8", lora_s="#D97706",
    ref_f="#DBEAFE", ref_s="#2563EB",
    end_f="#DCFCE7", end_s="#16A34A",
    mid_f="#FEE2E2", mid_s="#DC2626",
    text_f="#EDE9FE", text_s="#7C3AED",
    sig_f="#FEF3C7", sig_s="#CA8A04",
    plain_f="#FFFFFF", plain_s="#6B7280",
    hi_f="#FFFFFF", hi_s="#111827",
    unused_f="#FAFAFA", unused_s="#C4C9D2",
)
KIND = {  # fill, stroke, text
    "frozen": (C["frozen_f"], C["frozen_s"], C["ink"]),
    "lora": (C["lora_f"], C["lora_s"], "#7C2D12"),
    "ref": (C["ref_f"], C["ref_s"], "#1E3A8A"),
    "end": (C["end_f"], C["end_s"], "#14532D"),
    "mid": (C["mid_f"], C["mid_s"], "#7F1D1D"),
    "text": (C["text_f"], C["text_s"], "#3B0764"),
    "sig": (C["sig_f"], C["sig_s"], "#713F12"),
    "plain": (C["plain_f"], C["plain_s"], C["ink"]),
    "hi": (C["hi_f"], C["hi_s"], C["ink"]),
    "unused": (C["unused_f"], C["unused_s"], C["muted"]),
}

E: list[str] = []


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def est(size: float, s: str) -> float:  # conservative text width (DejaVu-wide)
    return 0.6 * size * len(s)


def text(x, y, s, size=12, anchor="start", weight="normal", fill=None, italic=False, rotate=None):
    fill = fill or C["ink"]
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    st = ' font-style="italic"' if italic else ""
    E.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}"'
        f' fill="{fill}"{st}{tr}>{esc(s)}</text>'
    )


def box(x, y, w, h, lines, kind="frozen", size=11.5, rx=6, dashed=False, stroke_w=1.4, bold_first=False):
    fill, stroke, tcol = KIND[kind]
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    E.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"'
        f' stroke-width="{stroke_w}"{dash}/>'
    )
    if isinstance(lines, str):
        lines = [lines]
    # per-line font size shrinks to fit the box width
    sizes = [min(size, (w - 12) / (0.6 * max(len(s), 1))) for s in lines]
    lh = [s * 1.28 for s in sizes]
    total = sum(lh)
    cy = y + h / 2
    yy = cy - total / 2
    for i, (s, fs) in enumerate(zip(lines, sizes)):
        baseline = yy + lh[i] * 0.78
        wt = "bold" if (bold_first and i == 0) else "normal"
        text(x + w / 2, baseline, s, size=fs, anchor="middle", weight=wt, fill=tcol)
        yy += lh[i]


def line(pts, color=None, width=1.4, dashed=False, arrow=True):
    color = color or C["line"]
    d = " ".join(f"{p[0]},{p[1]}" for p in pts)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    mk = ' marker-end="url(#arr)"' if arrow else ""
    E.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{mk}/>')


def plus(cx, cy, r=9):
    E.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{C["line"]}" stroke-width="1.6"/>')
    E.append(
        f'<line x1="{cx - 5}" y1="{cy}" x2="{cx + 5}" y2="{cy}" stroke="{C["line"]}" stroke-width="1.8"/>'
        f'<line x1="{cx}" y1="{cy - 5}" x2="{cx}" y2="{cy + 5}" stroke="{C["line"]}" stroke-width="1.8"/>'
    )


def badge(cx, cy, label, kind="sig", r=8.5, size=9.5):
    fill, stroke, tcol = KIND[kind]
    E.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    text(cx, cy + size * 0.36, label, size=size, anchor="middle", weight="bold", fill=tcol)


def strip(x, y, w, h, segs, size=11):
    """segs = [(label, kind, width)]"""
    xx = x
    for label, kind, sw in segs:
        fill, stroke, tcol = KIND[kind]
        E.append(f'<rect x="{xx}" y="{y}" width="{sw}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        if kind == "mid":
            E.append(f'<rect x="{xx}" y="{y}" width="{sw}" height="{h}" fill="url(#hatch)" stroke="none"/>')
        fs = min(size, (sw - 8) / (0.6 * max(len(label), 1)))
        text(xx + sw / 2, y + h / 2 + fs * 0.36, label, size=fs, anchor="middle", fill=tcol, weight="bold")
        xx += sw
    return x + w


def panel(x, y, w, h, title):
    E.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" stroke="{C["panel"]}"'
        f' stroke-width="1.2"/>'
    )
    text(x + 14, y + 22, title, size=13.5, weight="bold")


def mask_matrix(x, y, cell=17):
    """2x2 one-way mask: rows = queries, cols = keys."""
    text(x + cell * 0.5, y - 3, "q\\k", size=8, anchor="middle", fill=C["muted"])
    text(x + cell * 1.5, y - 3, "ref", size=8, anchor="middle", fill=C["muted"])
    text(x + cell * 2.5, y - 3, "tgt", size=8, anchor="middle", fill=C["muted"])
    rows = [("ref", [True, False]), ("tgt", [True, True])]
    for i, (rl, cells) in enumerate(rows):
        yy = y + i * cell
        text(x + cell * 0.5, yy + cell * 0.7, rl, size=8, anchor="middle", fill=C["muted"])
        for j, ok in enumerate(cells):
            xx = x + cell * (1 + j)
            fill = C["end_f"] if ok else "#E5E7EB"
            E.append(f'<rect x="{xx}" y="{yy}" width="{cell}" height="{cell}" fill="{fill}" stroke="#9CA3AF" stroke-width="0.8"/>')
            text(xx + cell / 2, yy + cell * 0.72, "✓" if ok else "✗", size=10, anchor="middle",
                 fill=C["end_s"] if ok else C["mid_s"], weight="bold")


# ----------------------------------------------------------------------------- header
text(40, 42, "LTX-2 IC-LoRA — the default training graph: what is frozen, where LoRA trains", size=21, weight="bold")
text(40, 62, "Recipe of ctt_v2 → ctt_v3 → dualforce_control (store/runs/002, 008, 012) · LTX-2 19B dev · "
     "one-way reference attention · LoRA r=128 α=128 on attn1 + attn2 + FF in every block", size=12, fill=C["muted"])

# ----------------------------------------------------------------------------- ① inputs → sequence
panel(30, 76, 758, 384, "①  Inputs → one token sequence   (training layout  [ reference | target ])")

box(48, 116, 168, 44, ["Reference demo video", "(the A→B example, full clip)"], kind="ref")
box(48, 170, 168, 44, ["Target clip", "start · middle · end frames"], kind="plain")
box(246, 116, 108, 98, ["VAE encoder", "(frozen)", "→ 128-ch latents", "1 token / latent cell"], kind="frozen")
line([(216, 138), (246, 138)])
line([(216, 192), (246, 192)])
line([(354, 146), (392, 146)])
text(373, 134, "concat", size=9, anchor="middle", fill=C["muted"])

SX, SY, SW, SH = 392, 124, 380, 44
strip(SX, SY, SW, SH, [("reference · M tokens", "ref", 140), ("start", "end", 50),
                       ("middle · generated", "mid", 140), ("end", "end", 50)])
for cx, lab in ((462, "σ = 0 (clean)"), (557, "σ = 0"), (652, "σ = σ_t (noised)"), (747, "σ = 0")):
    text(cx, 116, lab, size=10, anchor="middle", fill=C["muted"])
for cx, lab, col in ((462, "no loss", C["muted"]), (557, "no loss", C["muted"]),
                     (652, "loss ✓", C["mid_s"]), (747, "no loss", C["muted"])):
    text(cx, 182, lab, size=10, anchor="middle", fill=col, weight="bold" if "✓" in lab else "normal")
text(392, 202, "reference = demo latents, in front · same 16×20×15 grid → M = N = 4800", size=9.5, fill=C["muted"])
text(392, 215, "start / end = clean target latents pinned by the mask (cond_clean_latents)", size=9.5, fill=C["muted"])
text(392, 228, "middle = noised, x = (1−σ)·x₀ + σ·ε — the only supervised tokens", size=9.5, fill=C["muted"])

line([(582, 168), (582, 244)], arrow=True)
box(392, 244, 380, 40, ["patchify_proj · Linear 128 → 4096  (frozen)",
                        "one shared projection for reference · start · middle · end tokens"], kind="frozen", bold_first=True)
line([(582, 284), (582, 300)])
box(392, 300, 380, 40, ["+ RoPE (t, h, w) · reference tokens reuse the target's own grid",
                        "token role is carried by σ and the attention split — not by position"], kind="plain")
# the stream leaves the panel and enters the block panel at its top
line([(772, 320), (806, 320), (806, 100), (868, 100)], width=2)

box(48, 340, 168, 44, ["Text prompt", "caption + trigger 'sksz.'"], kind="text")
box(246, 340, 108, 44, ["Gemma-3-12B", "(frozen) · 3840-d"], kind="frozen")
box(392, 340, 300, 44, ["caption_projection (frozen)", "Lin 3840→4096 · GELU · Lin 4096→4096"], kind="frozen")
line([(216, 362), (246, 362)])
line([(354, 362), (392, 362)])
line([(692, 362), (729, 362)])
badge(741, 362, "T", kind="text", r=11, size=11)
text(741, 386, "→ K/V of attn2", size=9, anchor="middle", fill=C["muted"])

box(48, 400, 168, 44, ["σ per token", "0 on ref/start/end · σ_t on middle"], kind="sig")
box(246, 400, 446, 44, ["adaln_single (frozen): sinusoidal(σ) → MLP → 6 × 4096 modulation",
                        "shift / scale / gate for attn1 & FF, computed PER TOKEN (σ=0 tokens modulate as clean)"],
    kind="frozen")
line([(216, 422), (246, 422)])
line([(692, 422), (729, 422)])
badge(741, 422, "σ", kind="sig", r=11, size=11)
text(741, 446, "→ adaLN + gates", size=9, anchor="middle", fill=C["muted"])

# ----------------------------------------------------------------------------- ② one block
panel(830, 76, 740, 740, "②  One transformer block (video stream) · × 48 · d 4096 · 32 heads × 128 · FF 16384")

XS = 880  # residual stream
line([(XS, 100), (XS, 230)], width=2, arrow=False)
text(888, 112, "x : (M+N) tokens × 4096", size=9.5, fill=C["muted"])
# ×48 bracket
E.append(f'<line x1="858" y1="124" x2="858" y2="556" stroke="{C["muted"]}" stroke-width="1.2"/>'
         f'<line x1="858" y1="124" x2="866" y2="124" stroke="{C["muted"]}" stroke-width="1.2"/>'
         f'<line x1="858" y1="556" x2="866" y2="556" stroke="{C["muted"]}" stroke-width="1.2"/>')
text(851, 340, "× 48 blocks", size=11, anchor="middle", weight="bold", fill=C["muted"], rotate=-90)

# --- self-attention (attn1)
text(920, 122, "self-attention · attn1", size=11, weight="bold")
line([(XS, 150), (920, 150)])
box(920, 130, 78, 40, ["RMSNorm", "(no affine)"], kind="frozen")
line([(998, 150), (1012, 150)])
box(1012, 130, 126, 40, ["adaLN", "⊙(1+scale_msa)+shift_msa"], kind="frozen")
badge(1016, 128, "σ")
line([(1138, 150), (1147, 150)], arrow=False)
E.append(f'<line x1="1147" y1="131" x2="1147" y2="195" stroke="{C["line"]}" stroke-width="1.4"/>')
for yy in (131, 163, 195):
    line([(1147, yy), (1160, yy)])
box(1160, 118, 90, 26, "to_q (LoRA)", kind="lora", size=11)
box(1160, 150, 90, 26, "to_k (LoRA)", kind="lora", size=11)
box(1160, 182, 90, 26, "to_v (LoRA)", kind="lora", size=11)
line([(1250, 131), (1266, 131)])
line([(1250, 163), (1266, 163)])
box(1266, 118, 76, 58, ["q_norm ·", "k_norm", "(RMSNorm)"], kind="frozen", size=10.5)
line([(1342, 131), (1352, 131)])
line([(1342, 163), (1352, 163)])
box(1352, 118, 48, 58, ["RoPE", "(t,h,w)"], kind="plain", size=10.5)
line([(1400, 131), (1408, 131)])
line([(1400, 163), (1408, 163)])
line([(1250, 195), (1408, 195)])
box(1408, 112, 150, 100, "", kind="hi", stroke_w=1.8)
text(1483, 128, "SDPA · one-way", size=11, anchor="middle", weight="bold")
mask_matrix(1418, 146, cell=17)
text(1478, 152, "tgt → all keys", size=9, fill=C["end_s"])
text(1478, 165, "ref → ref only", size=9, fill=C["mid_s"])
text(1478, 178, "(no ref→tgt", size=8.5, fill=C["muted"])
text(1478, 189, " appearance leak)", size=8.5, fill=C["muted"])
text(1483, 205, "only place ref & tgt meet", size=8, anchor="middle", fill=C["muted"])
line([(1483, 212), (1483, 226)])
box(1408, 226, 150, 26, "to_out (LoRA)  4096→4096", kind="lora", size=10.5)
line([(1408, 239), (1390, 239)])
box(1300, 226, 90, 26, "× gate_msa", kind="plain", size=10.5)
badge(1304, 224, "σ")
line([(1300, 239), (892, 239)])
plus(XS, 239)

# --- cross-attention (attn2)
line([(XS, 248), (XS, 408)], width=2, arrow=False)
text(920, 282, "cross-attention  attn2  — text only (no adaLN, no gate: cross_attention_adaln = off)", size=11, weight="bold")
line([(XS, 310), (920, 310)])
box(920, 290, 78, 40, ["RMSNorm", "(no affine)"], kind="frozen")
line([(998, 310), (1160, 310)])
box(1160, 297, 90, 26, "to_q (LoRA)", kind="lora", size=11)
badge(1118, 361, "T", kind="text", r=11, size=11)
text(1118, 383, "text ctx", size=9, anchor="middle", fill=C["muted"])
line([(1129, 361), (1147, 361)], arrow=False)
E.append(f'<line x1="1147" y1="344" x2="1147" y2="378" stroke="{C["line"]}" stroke-width="1.4"/>')
line([(1147, 344), (1160, 344)])
line([(1147, 378), (1160, 378)])
box(1160, 331, 90, 26, "to_k (LoRA) ← T", kind="lora", size=10.5)
box(1160, 365, 90, 26, "to_v (LoRA) ← T", kind="lora", size=10.5)
line([(1250, 310), (1266, 310)])
line([(1250, 344), (1266, 344)])
box(1266, 297, 76, 60, ["q_norm ·", "k_norm", "(RMSNorm)"], kind="frozen", size=10.5)
line([(1342, 310), (1408, 310)])
line([(1342, 344), (1408, 344)])
line([(1250, 378), (1408, 378)])
box(1408, 297, 150, 94, ["SDPA (full)", "video queries over", "prompt tokens", "(context mask, no RoPE)"], kind="hi", stroke_w=1.8, size=10.5)
line([(1483, 391), (1483, 404)])
box(1408, 404, 150, 26, "to_out (LoRA)  4096→4096", kind="lora", size=10.5)
line([(1408, 417), (892, 417)])
plus(XS, 417)

# --- feed-forward
line([(XS, 426), (XS, 534)], width=2, arrow=False)
text(920, 460, "feed-forward  ff", size=11, weight="bold")
line([(XS, 490), (920, 490)])
box(920, 470, 78, 40, ["RMSNorm", "(no affine)"], kind="frozen")
line([(998, 490), (1012, 490)])
box(1012, 470, 126, 40, ["adaLN", "⊙(1+scale_mlp)+shift_mlp"], kind="frozen")
badge(1016, 468, "σ")
line([(1138, 490), (1160, 490)])
box(1160, 470, 130, 40, ["ff.net.0.proj (LoRA)", "4096 → 16384"], kind="lora")
line([(1290, 490), (1306, 490)])
box(1306, 470, 62, 40, ["GELU", "(tanh)"], kind="plain", size=10.5)
line([(1368, 490), (1384, 490)])
box(1384, 470, 130, 40, ["ff.net.2 (LoRA)", "16384 → 4096"], kind="lora")
line([(1449, 510), (1449, 530)])
box(1384, 530, 130, 26, "× gate_mlp", kind="plain", size=10.5)
badge(1388, 528, "σ")
line([(1384, 543), (892, 543)])
plus(XS, 543)

# --- σ explainer + audio note
box(920, 600, 290, 60, ["σ badge = per-token modulation (frozen):",
                        "adaln_single(σ) + scale_shift_table[6×4096]",
                        "→ shift / scale / gate for attn1 and FF"], kind="sig", size=10.5)
box(1230, 600, 328, 60, ["audio stream: audio_attn1/2 + a2v / v2a cross-attn",
                         "present in the 19B checkpoint, disabled here",
                         "(audio: null → never runs, never adapted)"], kind="unused", dashed=True, size=10.5)

# --- stream to the output head
line([(XS, 552), (XS, 702), (920, 702)], width=2)
text(890, 582, "→ next block; after block 47:", size=9.5, fill=C["muted"], italic=True)
box(920, 680, 230, 44, ["norm_out LayerNorm + scale_shift_table[2]", "(frozen; final modulation by σ)"], kind="frozen")
line([(1150, 702), (1166, 702)])
box(1166, 680, 120, 44, ["proj_out (frozen)", "Linear 4096 → 128"], kind="frozen")
line([(1286, 702), (1302, 702)])
text(1430, 674, "v̂ per token", size=10, anchor="middle", fill=C["muted"])
strip(1302, 680, 256, 44, [("discard", "ref", 92), ("–", "end", 32), ("loss ✓", "mid", 92), ("–", "end", 40)], size=10.5)
text(920, 746, "Loss: flow-matching MSE  ‖ v̂ − (ε − x₀) ‖²  averaged over loss_mask = the middle tokens only.", size=10.5)
text(920, 762, "Reference / start / end predictions are computed but discarded — no gradient flows from them.", size=10.5, fill=C["muted"])
text(920, 788, "LoRA sits on 10 linears per block: attn1 q·k·v·out · attn2 q·k·v·out · ff.net.0.proj · ff.net.2.", size=9.5, fill=C["muted"])
text(920, 802, "Everything else in the block is frozen (norms, adaLN tables) or parameter-free (RoPE, SDPA, GELU, gates).", size=9.5, fill=C["muted"])

# ----------------------------------------------------------------------------- ③ LoRA + ④ accounting + legend
panel(30, 480, 758, 326, "③  LoRA on one linear (PEFT)        ④  What trains, what never does")

# LoRA inset
text(58, 592, "x", size=13, anchor="middle", weight="bold")
line([(66, 588), (90, 588)])
E.append(f'<line x1="78" y1="588" x2="78" y2="535" stroke="{C["line"]}" stroke-width="1.4"/>')
line([(78, 535), (96, 535)])
box(90, 566, 110, 44, ["W (base, frozen)", "d_out × d_in"], kind="frozen")
box(96, 520, 76, 30, "A  128 × d_in", kind="lora", size=10.5)
line([(172, 535), (186, 535)])
box(186, 520, 130, 30, "B  d_out × 128  (init 0)", kind="lora", size=10.5)
line([(316, 535), (330, 535), (330, 579)])
line([(200, 588), (321, 588)])
plus(330, 588)
line([(339, 588), (366, 588)])
text(376, 592, "y", size=13, anchor="middle", weight="bold")
text(48, 632, "y = W x + (α/r) · B A x", size=11)
text(48, 647, "α/r = 128/128 = 1 · dropout 0", size=10.5, fill=C["muted"])
text(48, 662, "init: A random, B = 0 → starts as the base model", size=10.5, fill=C["muted"])
text(48, 677, "only A and B receive gradients", size=10.5, fill=C["muted"])

# accounting
text(420, 530, "LoRA targets — identical in all 48 blocks", size=11.5, weight="bold")
text(420, 547, "attn1.to_q · to_k · to_v · to_out.0        attn2.to_q · to_k · to_v · to_out.0", size=10.5)
text(420, 562, "ff.net.0.proj · ff.net.2", size=10.5)
text(420, 584, "10 linears × 48 blocks = 480 adapted linears · 960 tensors (A, B)", size=11, weight="bold")
text(420, 600, "654.3 M trainable params → 1.31 GB bf16 checkpoint ≈ 3.4 % of the 19B base", size=10.5)
text(420, 615, "r = 128 · α = 128 · dropout 0 · lr 1e-4 · WSD · effective batch 8", size=10.5)
text(420, 638, "Never adapted (frozen)", size=11.5, weight="bold")
text(420, 655, "VAE · Gemma-3 · caption_projection · patchify_proj · adaln_single", size=10.5)
text(420, 670, "scale_shift_table (every block + output) · all norms · proj_out · audio stream", size=10.5)

# legend
text(48, 700, "Legend", size=11.5, weight="bold")
def sw(x, y, kind, label, dashed=False, hatch=False):
    fill, stroke, _ = KIND[kind]
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    E.append(f'<rect x="{x}" y="{y - 10}" width="22" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>')
    if hatch:
        E.append(f'<rect x="{x}" y="{y - 10}" width="22" height="14" rx="3" fill="url(#hatch)" stroke="none"/>')
    text(x + 28, y + 1, label, size=10.5)
sw(48, 722, "frozen", "frozen base weight")
sw(218, 722, "lora", "LoRA-adapted linear (W frozen + B·A trainable)")
sw(520, 722, "plain", "parameter-free op")
sw(650, 722, "unused", "present, unused", dashed=True)
sw(48, 748, "ref", "reference (demo) tokens · σ = 0")
sw(262, 748, "end", "start / end tokens · clean · σ = 0")
sw(490, 748, "mid", "middle tokens · noised · supervised", hatch=True)
sw(48, 774, "text", "text context (Gemma → caption_projection)")
badge(319, 770, "σ")
text(334, 775, "per-token σ modulation (frozen adaLN)")
badge(610, 770, "T", kind="text")
text(625, 775, "text-context port")

# footer
text(40, 840, "Verified 2026-08-28 against src/LTX-2-dualforce (ltx-core: model.py, transformer.py, attention.py, feed_forward.py, adaln.py; "
     "ltx-trainer: flexible.py, trainer.py) and the store/runs/012 checkpoint (960 LoRA tensors, 48 blocks).", size=9.5, fill=C["muted"])
text(40, 854, "Inference concatenates [target | reference] instead of [reference | target]; the one-way rule is identical. "
     "Token counts (M = N = 4800) are the ctt_v2 training geometry (16 latent frames × 20 × 15).", size=9.5, fill=C["muted"])

# ----------------------------------------------------------------------------- assemble
DEFS = f"""<defs>
  <marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">
    <polygon points="0,0 8,4 0,8" fill="{C['line']}"/>
  </marker>
  <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="7" stroke="{C['mid_s']}" stroke-width="1" stroke-opacity="0.35"/>
  </pattern>
</defs>"""


def svg(font: str) -> str:
    body = "\n".join(E)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{font}" role="img" '
        f'aria-label="LTX-2 IC-LoRA training graph: frozen base, LoRA on attention and feed-forward linears in all 48 blocks, one-way reference attention">\n'
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n{DEFS}\n{body}\n</svg>\n'
    )


if __name__ == "__main__":
    (OUT / f"{NAME}.svg").write_text(svg(FONT), encoding="utf-8")
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=svg(PNG_FONT).encode("utf-8"), write_to=str(OUT / f"{NAME}.png"), output_width=W * 2)
        print("wrote", OUT / f"{NAME}.svg", "and .png")
    except ImportError:
        print("wrote", OUT / f"{NAME}.svg", "(cairosvg missing: no png)")
