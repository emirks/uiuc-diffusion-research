"""Build SPEC §2 scoring manifests for exp_067 Stage-2 videos.

Reuses exp_066's row/gt/ic_rows/chunk_by_class/training_manifest_ic3 verbatim (identical
harness contract) — the ONLY change is out_root -> exp_067's residual-generation videos.
Emits eval_d_abc_c{0..2}.json (3 class-chunks, like ic3_abc) + training_manifest_068.json
(same pairs as ic3; adapter_id relabeled). Scored via the certified eval/v3.0.0 worktree.
"""
import importlib.util
import json
import pathlib
import sys

R = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
EXP66 = R / "experiments/exp_066_ladder_v3_scoring/build_eval_manifests.py"
OUT_ROOT = R / "outputs/videos/exp_068_s3a"
MANIFEST = R / "experiments/exp_068_s3a/eval/manifest_068.json"
DS = R / "experiments/exp_068_s3a/eval"

spec = importlib.util.spec_from_file_location("bem66", EXP66)
bem = importlib.util.module_from_spec(spec)
sys.modules["bem66"] = bem
spec.loader.exec_module(bem)

rows = bem.ic_rows(MANIFEST, OUT_ROOT, lambda r: f"d_{r['tier'].lower()}", "s2")
parts = bem.chunk_by_class(rows, 3)
for i, part in enumerate(parts):
    (DS / f"eval_d_abc_c{i}.json").write_text(json.dumps(part, indent=1))
    print(f"[write] eval_d_abc_c{i}.json  {len(part)} rows")

tm = bem.training_manifest_ic3()
tm["adapter_id"] = "exp068_s3a_step_05000"
(DS / "training_manifest_068.json").write_text(json.dumps(tm, indent=1))
print(f"[done] {len(rows)} eval rows (arms s2_a/s2_b/s2_c) + training_manifest_068.json")
