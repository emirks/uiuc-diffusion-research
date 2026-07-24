"""exp_079 build-order 2: pixel-space temporal manipulations -> frozen LTX-VAE latents.

For each split clip, load its std121 pixels once, apply each required manipulation IN PIXEL SPACE,
and VAE-encode -> latent [128,16,20,15]. Reuses eval_ladder/encode_conditioning's exact VAE path
(same load/preprocess/encode as every other latent in this campaign), so the identity latents match
the on-disk geometry and the manipulated latents are drawn from the same distribution.

Required manipulations per split:
  * train / heldout_instance clips (held-in classes): the 4 TRAIN manipulations.
  * heldout_class clips (zero-shot):                   all 8 (TRAIN + HELDOUT gammas) — the
                                                        temporal-generalization probe needs them.

Output: dataset/manip_latents/<split>/<class>/<clip>__<manip>.pt
        = {latents[128,16,20,15] bf16, clip, cls, manip, split}

    # GPU (L40S fine — no generation):
    python experiments/exp_079_standalone_operator_encoder/encode_manipulations.py
"""

import json
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))
sys.path.insert(0, str(EXP))
import encode_conditioning as ec          # noqa: E402
from manip_utils import HELDOUT_MANIPS, TRAIN_MANIPS, manipulate  # noqa: E402

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
SPLIT = EXP / "split.json"
OUT = EXP / "dataset" / "manip_latents"

ALL_MANIPS = TRAIN_MANIPS + HELDOUT_MANIPS


def clips_with_manips(split: dict):
    """Yield (split_name, clip, cls, mp4, [manips]) for every latent we must produce."""
    for name in ("train", "heldout_instance"):
        for c in split[name]:
            yield name, c["clip"], c["cls"], c["mp4"], TRAIN_MANIPS
    for c in split["heldout_class"]:
        yield "heldout_class", c["clip"], c["cls"], c["mp4"], ALL_MANIPS


def main() -> None:
    device = "cuda"
    split = json.loads(SPLIT.read_text())
    work = list(clips_with_manips(split))
    n_lat = sum(len(m) for *_, m in work)
    print(f"[encode] {len(work)} clips -> {n_lat} manipulated latents "
          f"(train_manips={TRAIN_MANIPS}, heldout_manips={HELDOUT_MANIPS})")

    vae = ec.load_vae(str(MODEL), device=device)
    done = roundtrip = 0
    for i, (sp, clip, cls, mp4, manips) in enumerate(work):
        px = None  # lazy: only read the mp4 if some manip is missing
        for manip in manips:
            dst = OUT / sp / cls / f"{clip}__{manip}.pt"
            if dst.exists():
                continue
            if px is None:
                px = ec.preprocess(REPO_ROOT / mp4)         # [121,C,H,W] in [0,1]
                assert px.shape[0] == ec.STD_FRAMES, f"{clip}: {px.shape[0]} frames"
            pm = manipulate(px, manip)                       # pixel-space manipulation
            lat = ec.encode(pm, vae, device, torch.bfloat16)[0].to(torch.bfloat16).cpu()  # [128,16,20,15]
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"latents": lat, "clip": clip, "cls": cls, "manip": manip, "split": sp}, dst)
            done += 1

            if roundtrip == 0 and manip == "reverse":
                # pre-registered sanity: reverse latent should differ substantially from identity
                idp = OUT / sp / cls / f"{clip}__identity.pt"
                if idp.exists():
                    idl = torch.load(idp, weights_only=False)["latents"]
                    r = ec.rel_l2(lat, idl)
                    print(f"[roundtrip] {clip}: reverse-vs-identity rel_L2={r:.3f} "
                          f"(want >0 — a real temporal difference in latent space) shape={tuple(lat.shape)}")
                    roundtrip = 1
        if (i + 1) % 25 == 0:
            print(f"[encode] {i + 1}/{len(work)} clips  ({done} latents written)")

    total = len(list(OUT.rglob("*.pt")))
    print(f"[encode] done: {done} new, {total} total latents in {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
