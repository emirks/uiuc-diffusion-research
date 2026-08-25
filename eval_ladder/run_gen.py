"""ladder2 — the generator. Consumes `registry.jsonl` rows for ONE arm.

    row  x  seed  ==  exactly one video

Nothing is decided here. The row already carries the prompt, the conditioning, the reference
and the arm; this script only loads that arm's adapter once and renders its rows. Selection is
by `--arm` (which model) and optionally `--priority` (which wave), so a generation job is
always "this model, these rows".

    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <repo>/eval_ladder/run_gen.py \
        --arm spec_color_rain --seed 42 [--priority P0] [--chunk 0 --num-chunks 4]

Skip-if-exists on the output path, so preemption + requeue simply continues.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig,
    ReferenceConditionConfig,
    SuffixConditionConfig,
    ValidationConfig,
    ValidationSample,
)
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

# LAB was hardcoded, which silently breaks this script on any machine that is not the campus
# cluster: MODEL/GEMMA resolved to CC paths and model loading failed at run time. The defaults are
# unchanged, so CC behaviour is byte-identical; LTX_MODEL / LTX_GEMMA additionally allow pointing at
# weights that do not sit under `cache/huggingface/...` (on the eps box they live in $MODELS).
LAB = Path(os.environ.get("LAB", "/projects/illinois/eng/cs/jrehg/users/emirkisa"))
MODEL = Path(os.environ.get("LTX_MODEL", LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"))
GEMMA = Path(os.environ.get("LTX_GEMMA", LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"))
STD = REPO_ROOT / "data/processed/transitions_std121"
REGISTRY = HERE / "registry.jsonl"
ARMS = HERE / "arms.yaml"
# exp_078: campaign-private output root. The default writes into the SHARED ladder2 video
# tree, which the parallel dataset-expansion campaign reads; the bottleneck campaign points
# this elsewhere so it can never add files under paths another campaign is scoring.
OUT_ROOT = Path(os.environ.get("LADDER_OUT_ROOT", REPO_ROOT / "outputs/videos/ladder2"))


def build_sample(row: dict) -> ValidationSample:
    # ref_downscale set per-arm in main() (m1lite uses 5 for the coarse 128x96 reference)
    """Conditioning is a pure function of the row — the same rule the mask uses at eval."""
    conds = []
    if row.get("conditioning") != "none" and row["endpoint"] is not None:
        paths = ec.cond_paths(row["endpoint"], row["sided"])
        conds.append(PrefixConditionConfig(video=str(paths["prefix"]), num_frames=ec.PX_PREFIX))
        if row["sided"] == "two":
            conds.append(SuffixConditionConfig(video=str(paths["suffix"]),
                                               num_frames=ec.SUFFIX_GEN_FRAMES))
    # `use_reference: false` (default true, so every pre-existing row is unaffected) keeps the
    # reference OFF the model while leaving it ON the row. The baseline arms need exactly that:
    # run_eval.pool_refs() bans the row's reference from the GT pool, so dropping the field would
    # hand them a different pool than the arms they are compared against — a silently unfair join.
    if row.get("reference") and row.get("use_reference", True):
        # `code_source_reference` (campaign `bneck_coupling`) names a DIFFERENT clip to feed the
        # model, while `reference` — the row's identity — is left untouched. That separation is
        # load-bearing, not cosmetic: run_eval.pool_refs() excludes the row's own `reference` from
        # the GT pool, so swapping `reference` to build the shuffled-code corpse would change pool
        # COMPOSITION between the matched and shuffled twins and contaminate every paired Δ — the
        # same class of keyed-join defect that flipped two cells in the ladder audit. With this
        # field the two arms are scored against byte-identical pools and differ only in what the
        # encoder saw. The eval side never reads it.
        # Calibrator A (campaign `bneck_coupling`, advisor A10): force EVERY row's code source to
        # one fixed clip. The encoder is frozen and deterministic, so a constant input clip yields a
        # constant 72-token code — which makes the matched and shuffled registries carry IDENTICAL
        # effective conditioning. Any distance measured between them is then nondeterminism floor
        # plus hidden leak, i.e. a channel that MUST read DEAD. Default None keeps every other
        # campaign's behaviour byte-identical.
        model_ref = (build_sample.const_code_clip
                     or row.get("code_source_reference") or row["reference"])
        # `reference_path` (campaign 2026-08-14_timing_relay): an explicit clip path fed to the model,
        # bypassing the STD/class/name resolution — for retimed reference demos that live OUTSIDE the
        # std121 tree and are not in split_v1.2. Absent on every pre-existing row → behaviour
        # byte-identical. The row's `reference` field is left as the clip identity (pool_refs still
        # excludes it); only the fed video differs.
        if row.get("reference_path"):
            ref_path = Path(row["reference_path"])
            if not ref_path.is_absolute():
                ref_path = REPO_ROOT / ref_path
        else:
            # authoritative clip -> class (clip names do NOT reliably encode the class)
            ref_path = STD / prompts.clip_class(model_ref) / f"{model_ref}.mp4"
        assert ref_path.exists(), f"reference clip not found: {ref_path}"
        # `attention` MUST match what the adapter was TRAINED with. ReferenceConditionConfig
        # defaults to "bidirectional", so omitting it silently runs a one-way-trained adapter
        # bidirectionally — a train/inference mismatch the config's own docstring warns about.
        # Every arm through b1/m1lite was trained bidirectionally, so the default keeps them
        # byte-identical; ctt_v2 sets `attention: one_way` in arms.yaml.
        conds.append(ReferenceConditionConfig(video=str(ref_path),
                                              downscale_factor=build_sample.ref_downscale,
                                              attention=build_sample.ref_attention,
                                              temporal_scale_factor=1, include_in_output=False))
    return ValidationSample(prompt=row["prompt"], conditions=conds)


def load_rows(arm: str, priority: str | None, cells: str | None = None,
              extra_registry: str | None = None) -> list[dict]:
    text = REGISTRY.read_text()
    if extra_registry:
        # ADDITIVE side registries (exp_077's d2_gen): registry.jsonl is frozen, so a new arm's
        # rows arrive in their own file and are appended, never merged into the frozen one.
        text += Path(extra_registry).read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = [r for r in rows if r["arm"] == arm]
    if priority:
        keep = set(priority.split(","))
        rows = [r for r in rows if r["priority"] in keep]
    if cells:
        # NOTE: Slurm's --export is itself comma-separated, so a comma-joined list silently
        # truncates to its first element. Accept "|" (what submit.py sends) as well as ",".
        want = {c for c in re.split(r"[,|]", cells) if c}
        rows = [r for r in rows if r["cell"] in want]
    return rows


def resolve_adapter(arm: str, arms_cfg: dict, step: int | None = None) -> tuple[Path | None, list[str]]:
    """`step` overrides the arm's pinned checkpoint — used ONLY by the convergence diagnostic
    (ckpt-4500 vs ckpt-5000). Claim-bearing generation always uses the pinned step from arms.yaml."""
    spec = arms_cfg["arms"][arm]
    if spec.get("adapter", "unset") is None or spec["kind"] in ("base", "text_floor"):
        return None, []
    if isinstance(spec.get("adapter"), str):
        # explicit checkpoint path — for arms trained outside adapter_template's tree.
        # Still PINNED in arms.yaml before any score is seen; --step may not override it.
        assert step is None, f"{arm} pins an explicit adapter; --step cannot override it"
        path = REPO_ROOT / spec["adapter"]
        assert path.exists(), f"adapter missing for {arm}: {path}"
        return path, arms_cfg["targets"][spec["targets"]]
    # per-arm adapter_template override, falling back to the global template
    template = spec.get("adapter_template", arms_cfg["adapter_template"])
    path = REPO_ROOT / template.format(arm=arm, step=step or spec["step"])
    assert path.exists(), f"adapter missing for {arm}: {path}"
    return path, arms_cfg["targets"][spec["targets"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--priority", default=None, help="e.g. P0 or P0,P1")
    ap.add_argument("--cells", default=None, help="comma-separated cell names")
    ap.add_argument("--step", type=int, default=None,
                    help="override the pinned checkpoint (convergence diagnostic only)")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--extra-registry", default=None,
                    help="additional jsonl of registry rows, appended to the frozen registry")
    ap.add_argument("--out-root", default=None,
                    help="override outputs/videos/ladder2 (side lanes write their own tree)")
    ap.add_argument("--videos-dir", action="store_true",
                    help="contract v2: --out-root is a store gen subentry; write clips flat "
                         "into <out-root>/videos/ (no <arm>__ck<step> dir — step lives in meta)")
    ap.add_argument("--const-code-clip", default=None,
                    help="Calibrator A (bneck_coupling/A10): force every row's code source to this "
                         "ONE clip, making matched and shuffled registries identical by "
                         "construction — the must-DEAD control. Use with --out-root.")
    ap.add_argument("--posshuf-seed", type=int, default=None,
                    help="D3 rider (bneck_redesign, DESIGN_LOCK §3): for a bottleneck arm, apply ONE "
                         "fixed permutation (torch.randperm with this seed) of the K operator "
                         "tokens' grid-cell assignment BEFORE RoPE positions are stamped. Content "
                         "otherwise identical. Diagnostic ONLY — use with --out-root.")
    ap.add_argument("--zero-context-adapter", action="store_true",
                    help="Dead-channel calibrator (bneck_redesign Idea 1): for an inject=context arm, "
                         "zero the ContextAdapter output head AFTER loading, so it emits all-zero "
                         "context tokens. The code's ONLY route to the output is then severed, so a "
                         "matched arm and its deranged twin must produce BITWISE-identical video "
                         "(proves no hidden leak / the reference never enters the sequence). "
                         "Diagnostic ONLY — use with --out-root.")
    args = ap.parse_args()

    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)

    arms_cfg = yaml.safe_load(ARMS.read_text())
    assert args.seed in arms_cfg["seeds"], f"seed {args.seed} is not a registered seed"
    build_sample.ref_downscale = arms_cfg['arms'][args.arm].get('ref_downscale', 1)
    build_sample.ref_attention = arms_cfg['arms'][args.arm].get('attention', 'bidirectional')
    build_sample.const_code_clip = args.const_code_clip
    if args.const_code_clip:
        assert args.out_root, "--const-code-clip must write to its own --out-root (it is a " \
                              "calibrator, not a scored arm) or it would overwrite real videos"
        print(f"[gen] CALIBRATOR A: code source forced to '{args.const_code_clip}' for EVERY row")
    # flowsig hook (campaign 2026-08-24_flow_signal_conditioning, Step-2 EVAL). Read here because
    # the conditioning MODE decides whether the pixel reference is attached, and that decision has
    # to land before build_sample() runs. Unset FLOWSIG_CONFIG => byte-identical to every other
    # campaign; the reference field itself is never touched, only `use_reference` (which run_eval
    # ignores), so the GT pool composition is identical across the four modes.
    _flowsig_cfg = None
    if os.environ.get("FLOWSIG_CONFIG"):
        _flowsig_cfg = json.loads(Path(os.environ["FLOWSIG_CONFIG"]).read_text())
        assert args.out_root, "FLOWSIG_CONFIG must write to its own --out-root"
    rows = load_rows(args.arm, args.priority, args.cells, args.extra_registry)
    if _flowsig_cfg is not None and _flowsig_cfg["mode"] not in ("both", "ref_only"):
        for _r in rows:
            _r["use_reference"] = False
        print(f"[gen] flowsig mode={_flowsig_cfg['mode']}: pixel reference DROPPED from the model "
              f"on all {len(rows)} rows (row identity `reference` untouched)")

    def out_path(r: dict) -> Path:
        # Baseline rows share one canonical video per (endpoint, sided) through
        # video_key = "<dir>/<name>" — the arm never sees the donor, so per-task outputs would
        # be byte-duplicates. Rows without video_key keep row x seed = one video.
        vk = r.get("video_key")
        d, name = vk.split("/", 1) if vk else (r["arm"], r["item_id"])
        suffix = f"__ck{args.step}" if args.step else ""
        if args.videos_dir:
            # contract v2: --out-root IS a store gen subentry — clips land flat in
            # videos/, the step lives in meta.yaml, never in a path
            return OUT_ROOT / "videos" / f"{name}__s{args.seed}.mp4"
        return OUT_ROOT / (d + suffix) / f"{name}__s{args.seed}.mp4"

    # dedup by canonical path BEFORE chunk slicing, so two array tasks can never race on the
    # same shared video; then slice. (No-op for arms without video_key.)
    seen_paths: set[Path] = set()
    rows = [r for r in rows
            if (p := out_path(r)) not in seen_paths and not seen_paths.add(p)]
    rows = rows[args.chunk::args.num_chunks]
    assert rows, f"no registry rows for arm={args.arm} priority={args.priority} cells={args.cells}"

    todo = [r for r in rows if not out_path(r).exists()]
    print(f"[gen] arm={args.arm} seed={args.seed} chunk={args.chunk}/{args.num_chunks}: "
          f"{len(todo)}/{len(rows)} to generate")
    if not todo:
        print("[gen] nothing to do")
        return
    for r in todo:
        out_path(r).parent.mkdir(parents=True, exist_ok=True)

    adapter, target_modules = resolve_adapter(args.arm, arms_cfg, args.step)
    inf = arms_cfg["inference"]
    # env-gated inference-steps override (no-op when unset → byte-identical for all existing
    # campaigns; mirrors the RELAY_CONFIG env gate below). Used by the lerp-collapse x0-hat probe:
    # a 1-step Euler generation from noise IS the model's x0-hat = x_1 - v_pred (conditional mean).
    if os.environ.get("GEN_INFERENCE_STEPS"):
        inf = {**inf, "steps": int(os.environ["GEN_INFERENCE_STEPS"])}
        print(f"[gen] GEN_INFERENCE_STEPS override → inference_steps={inf['steps']}")
    # ValidationSample.video_dims is (WIDTH, HEIGHT, FRAMES) — the corpus is portrait
    # 480x640, so this reads 480 wide by 640 high (same tuple exp_074 generated with).
    vw, vh, vf = arms_cfg["resolution"]
    val_cfg = ValidationConfig(
        samples=[build_sample(r) for r in todo],
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(vw, vh, vf), frame_rate=24.0, seed=args.seed,
        inference_steps=inf["steps"], interval=1, guidance_scale=inf["guidance_scale"],
        stg_scale=inf["stg_scale"], stg_blocks=inf["stg_blocks"], stg_mode=inf["stg_mode"],
        generate_audio=False,
    )

    device = torch.device("cuda")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)
    # relay hook (campaign misc/2026-08-14_timing_relay, Wave-2): install a Prompt-Relay cross-attention
    # bias when RELAY_CONFIG points to a config json. Monkeypatches ltx_core Attention.forward at class
    # level (affects the transformer built next). No-op when the env var is unset → byte-identical.
    if os.environ.get("RELAY_CONFIG"):
        _rc = Path(os.environ["RELAY_CONFIG"])
        _relay_cfg = json.loads(_rc.read_text())
        sys.path.insert(0, str(_rc.parent))
        import relay_hook
        relay_hook.install(_relay_cfg)
    # rope hook (campaign misc/2026-08-14_timing_relay, Wave-3): warp the TEMPORAL RoPE coordinate of the
    # target middle frames when ROPE_CONFIG points to a config json. Monkeypatches ltx_core
    # TransformerArgsPreprocessor._prepare_positional_embeddings at class level. No-op when unset.
    if os.environ.get("ROPE_CONFIG"):
        _pc = Path(os.environ["ROPE_CONFIG"])
        _rope_cfg = json.loads(_pc.read_text())
        sys.path.insert(0, str(_pc.parent))
        import rope_hook
        rope_hook.install(_rope_cfg)
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    if adapter is not None:
        print(f"[gen] adapter {adapter.name} ({len(target_modules)} target modules)")
        transformer = get_peft_model(transformer, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, target_modules=target_modules,
            lora_dropout=0.0, init_lora_weights=True))
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(str(adapter)).items()}
        set_peft_model_state_dict(transformer.get_base_model(), sd)
    else:
        print(f"[gen] {args.arm}: no adapter (base weights)")
    transformer = transformer.to(device).eval()

    # exp_078: if this arm is a bottleneck arm, reconstruct the operator-token encoder from the
    # SAME checkpoint file (saved under its own prefix) and attach it, so inference mirrors
    # training: the demo is compressed to K tokens before it ever enters the sequence.
    bspec = arms_cfg["arms"][args.arm].get("bottleneck")
    if bspec is not None and bspec.get("encoder_class") == "vjepa":
        # Campaign `bneck_redesign`, Idea 3: vjepa gen is backbone-free — the trained projector
        # (loaded from the coupling checkpoint's operator_encoder.projector.*) runs on PRECOMPUTED
        # eval-reference V-JEPA residual features (bspec['vjepa_gen_feats'], keyed by the reference
        # mp4 path validation_runner passes as cond.video). Mirrors the training/validation vjepa path.
        from ltx_trainer.vjepa_projector_encoder import VJEPAProjectorEncoder  # noqa: PLC0415
        from ltx_trainer.bneck_contract import compute_spanning_factors  # noqa: PLC0415
        f, h, w = bspec["token_shape"]
        gf = bspec["vjepa_gen_feats"]
        gen_feats = Path(gf) if os.path.isabs(gf) else REPO_ROOT / gf
        encoder = VJEPAProjectorEncoder(val_feats_path=str(gen_feats))  # gen mode: no training bank
        raw = load_file(str(adapter))
        enc_sd = {k[len("operator_encoder."):]: v for k, v in raw.items() if k.startswith("operator_encoder.")}
        if "projector.fc1.weight" not in enc_sd:
            raise RuntimeError(f"arm '{args.arm}' declares a vjepa bottleneck but checkpoint "
                               f"{adapter} has no operator_encoder.projector.* tensors — refusing "
                               "to generate without the trained projector")
        encoder.load_state_dict(enc_sd, strict=False)  # loads the trained projector.*
        encoder = encoder.to(device=device, dtype=torch.float32).eval()
        th, tw, tf = vh // 32, vw // 32, (vf - 1) // 8 + 1
        dsf, tsf = compute_spanning_factors((f, h, w), tf, th, tw)  # 5.7735 / 1.0 on the 121f grid
        if not encoder._val_feats:
            raise RuntimeError(f"vjepa_gen_feats {gen_feats} loaded 0 features — precompute them first")
        runner.attach_operator_encoder(encoder, downscale_factor=dsf, temporal_scale_factor=tsf)
        print(f"[gen] vjepa projector bottleneck ATTACHED: K={f*h*w} shape=({f},{h},{w}) "
              f"scale=(spatial {dsf:.4f}, temporal {tsf:.4f}), {len(enc_sd)} projector tensors, "
              f"{len(encoder._val_feats)} precomputed gen-ref features", flush=True)
    elif bspec is not None:
        # Campaign `bneck_redesign`, Idea 2: 'hrc' reconstructs the HRCEncoder; default the certified
        # OperatorTokenEncoder. Same kwargs, same encode_reference contract.
        if bspec.get("encoder_class", "operator") == "hrc":
            from ltx_trainer.hrc_encoder import HRCEncoder as _EncoderClass
        else:
            from ltx_trainer.operator_encoder import OperatorTokenEncoder as _EncoderClass
        f, h, w = bspec["token_shape"]
        encoder = _EncoderClass(
            token_shape=(f, h, w), width=bspec.get("width", 512), depth=bspec.get("depth", 2),
            num_heads=bspec.get("num_heads", 8),
            prefix_latent_frames=bspec.get("prefix_latent_frames", 2),
            suffix_latent_frames=bspec.get("suffix_latent_frames", 1),
            skip_scale=bspec.get("skip_scale", 0.0),
        )
        raw = load_file(str(adapter))
        enc_sd = {k[len("operator_encoder."):]: v for k, v in raw.items() if k.startswith("operator_encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=True), None
        # fp32, NOT bf16 (campaign `bneck_coupling`, design lock A5 §1): every certified capability
        # number came from the fp32 featurization path, and `bneck_contract.encode_reference`
        # disables autocast around the call so the arithmetic actually stays fp32. Casting to bf16
        # here — as this line used to — silently makes the coupled artifact a different one from
        # the certificate's, and the G1 contract gate rejects bf16 for exactly that reason.
        encoder = encoder.to(device=device, dtype=torch.float32).eval()
        # Scale factors MUST match what training derived from the token shape, or the K tokens
        # land on the wrong positional grid (silently — nothing raises).
        th, tw, tf = vh // 32, vw // 32, (vf - 1) // 8 + 1
        # `positional_mode` MUST match what the adapter was trained with (bneck_contract / the
        # trainer's flexible.py). 'fixed' pins both factors to 1 so the code's placement is
        # target-independent; 'commensurate' infers them and needs a uniform scale. A mismatch here
        # is silent and would corrupt every generation — gate PP-1 exists to catch it.
        pos_mode = bspec.get("positional_mode", "commensurate")
        if bspec.get("spanning_positions", False):
            # Campaign `bneck_redesign`, Idea 2 §3: FLOAT spanning factors, identical to the training
            # path (bneck_contract.compute_spanning_factors). Valid for any grid.
            from ltx_trainer.bneck_contract import compute_spanning_factors
            dsf, tsf = compute_spanning_factors((f, h, w), tf, th, tw)
        elif pos_mode == "fixed":
            dsf, tsf = 1, 1
        else:
            dsf, tsf = th // h, (tf - 1) // (f - 1)
            assert th // h == tw // w, f"non-uniform spatial scale: {th}//{h} vs {tw}//{w}"
        if not enc_sd:
            raise RuntimeError(f"arm '{args.arm}' declares a bottleneck but the checkpoint "
                               f"{adapter} has no operator_encoder.* tensors — refusing to generate "
                               "raw-reference videos under a bottleneck arm name")
        runner.attach_operator_encoder(encoder, downscale_factor=dsf, temporal_scale_factor=tsf)
        # flush=True: Slurm block-buffers stdout, so without it this provenance line is invisible
        # until process exit — and it is the record that proves the videos are bottleneck-generated.
        print(f"[gen] operator-token bottleneck ATTACHED: K={f*h*w} shape=({f},{h},{w}) "
              f"scale=(spatial {dsf}, temporal {tsf}), {len(enc_sd)} encoder tensors", flush=True)

        # D3 rider (campaign `bneck_redesign`): one fixed permutation of the K=f*h*w token->cell
        # assignment before RoPE stamping. Seeded here so the permutation is identical across every
        # array task and every seed of the sweep; the runner reorders ref_latent's cells (see
        # ValidationRunner.attach_position_shuffle). Diagnostic only — refuses to write into a
        # scored tree by requiring --out-root, exactly like the calibrator flags.
        if args.posshuf_seed is not None:
            assert args.out_root, "--posshuf-seed must write to its own --out-root (it is a " \
                                  "positional-diagnostic arm, not a scored one)"
            k_tokens = f * h * w
            g = torch.Generator().manual_seed(args.posshuf_seed)
            perm = torch.randperm(k_tokens, generator=g)
            runner.attach_position_shuffle(perm)
            n_fixed = int((perm == torch.arange(k_tokens)).sum())
            print(f"[gen] D3 POSITION-SHUFFLE armed: seed={args.posshuf_seed} K={k_tokens} "
                  f"fixed_points={n_fixed} perm={perm.tolist()}", flush=True)

        # Idea 1 (campaign `bneck_redesign`): inject='context' routes the code through a trained
        # ContextAdapter into the CROSS-attention context stream instead of the latent sequence. The
        # encoder above still runs (it produces the code); the adapter, loaded from the SAME
        # checkpoint under its own prefix, maps that code to K' context tokens. d_ctx / width / K' are
        # READ from the saved tensors, so no model dim is hardcoded and a shape drift fails loudly.
        if bspec.get("inject", "reference") == "context":
            from ltx_trainer.context_adapter import ContextAdapter
            adapter_sd = {k[len("context_adapter."):]: v
                          for k, v in raw.items() if k.startswith("context_adapter.")}
            if not adapter_sd:
                raise RuntimeError(f"arm '{args.arm}' declares inject='context' but the checkpoint "
                                   f"{adapter} has no context_adapter.* tensors — refusing to "
                                   "generate with an empty context port")
            d_ctx = adapter_sd["output_proj.weight"].shape[0]
            adapter_width = adapter_sd["output_proj.weight"].shape[1]
            context_tokens = adapter_sd["queries"].shape[0]
            adapter_mod = ContextAdapter(
                d_ctx=d_ctx, token_shape=(f, h, w), context_tokens=context_tokens,
                width=adapter_width, num_heads=bspec.get("num_heads", 8),
                zero_init_output=False,  # weights are loaded next.
            )
            adapter_mod.load_state_dict(adapter_sd, strict=True)
            # Dead-channel calibrator (bneck_redesign Idea 1): zero the output head so the adapter
            # emits all-zero context tokens regardless of the code. With the reference off the latent
            # sequence in context mode, this severs the code's ONLY route to the output — matched and
            # deranged twins must then be BITWISE identical. Refuses to write a scored tree.
            if args.zero_context_adapter:
                assert args.out_root, "--zero-context-adapter must write to its own --out-root " \
                                      "(it is a calibrator, not a scored arm)"
                adapter_mod.zero_init_output_head()
                print("[gen] DEAD-CHANNEL calibrator: ContextAdapter output head ZEROED "
                      "(A_out == 0 for every code; twins must be bitwise identical)", flush=True)
            # fp32 island (as at training): bneck_contract keeps autocast off around encode_reference,
            # and the adapter forces fp32 internally, so the code path stays fp32 end to end.
            adapter_mod = adapter_mod.to(device=device, dtype=torch.float32).eval()
            runner.attach_context_adapter(adapter_mod)
            print(f"[gen] context port ATTACHED (Idea 1): K'={context_tokens} width={adapter_width} "
                  f"d_ctx={d_ctx}, {len(adapter_sd)} adapter tensors", flush=True)

    # Scratch dir MUST be unique per running task: the runner writes step_000000_N.mp4 here and we
    # rename them to <item_id>. Two concurrent SUBMISSIONS sharing (arm, seed, chunk, out-root) — e.g.
    # a campaign fanned across rope configs — would otherwise clobber each other's samples mid-rename
    # (FileNotFoundError). Slurm job/array ids disambiguate; empty (bare arm_s_c) for local single runs.
    if _flowsig_cfg is not None:
        _fc = Path(os.environ["FLOWSIG_CONFIG"])
        # the hook module normally sits beside the config (RELAY_CONFIG/ROPE_CONFIG precedent);
        # `hook_dir` lets a campaign keep configs in gen/cfg/ and code in build/ without a symlink.
        for _d in (_flowsig_cfg.get("hook_dir"), str(_fc.parent)):
            if _d:
                sys.path.insert(0, str(Path(_d) if os.path.isabs(str(_d)) else REPO_ROOT / str(_d)))
        import flowsig_hook  # noqa: PLC0415
        _flowsig_cfg.setdefault("checkpoint", str(adapter))
        flowsig_hook.install(_flowsig_cfg, runner, transformer,
                             samples=val_cfg.samples, rows=todo)

    _jid = os.environ.get("SLURM_JOB_ID", "")
    _suffix = f"_{_jid}" if _jid else ""
    tmp = OUT_ROOT / "_runner" / f"{args.arm}_s{args.seed}_c{args.chunk}{_suffix}"
    saved = runner.run(transformer=transformer, step=0, output_dir=tmp, device=device,
                       progress=TrainingProgress(enabled=True, total_steps=1))
    for idx, path in saved:
        dst = out_path(todo[idx])
        Path(path).rename(dst)
        # dst may live outside the repo (LADDER_OUT_ROOT/--out-root); relative_to would raise
        # ValueError here and kill the job after the first rename, orphaning the rest in _runner/.
        rel = dst.relative_to(REPO_ROOT) if dst.is_relative_to(REPO_ROOT) else dst
        print(f"[done] {todo[idx]['item_id']} s{args.seed} -> {rel}")


if __name__ == "__main__":
    main()
