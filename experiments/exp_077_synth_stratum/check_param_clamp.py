"""exp_077 D2-FULL — offline verification of the parameter clamp (no GL, no render).

Draws N operator parameter sets per keep-shader with the clamp ACTIVE, then hard-asserts the
ruling's invariants on every delivered value and reports which shaders/params clamp most.

    python check_param_clamp.py [N]
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

import param_clamp as pc  # noqa: E402
from engine import shaders  # noqa: E402

OUT = HERE / "D2_CLAMP_CHECK.json"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cfg = load_config(HERE / "config_d2full.yaml")
    plan = json.loads((HERE / "d2full_plan.json").read_text())
    keep = plan["keep_shaders"]
    bank = shaders.load_bank(Path(cfg["model"]["shader_bank"]))
    fil = pc.make_filter(True)

    ev_by_rule: Counter = Counter()
    ev_by_param: Counter = Counter()
    ev_by_shader: Counter = Counter()
    ev_default: Counter = Counter()
    n_draw = n_with_event = 0
    viol: list = []
    worst: dict = {}
    vec3_seen: dict = {}

    for name in keep:
        sh = bank[name]
        defs = pc.defaults_of(sh)
        for u in sh.tunable:
            if u.gtype in ("vec3",) or (u.gtype == "vec4" and "colo" in u.name.lower()):
                vec3_seen[f"{name}.{u.name}"] = [u.gtype, u.default]
        for i in range(n):
            rng = random.Random(f"clampcheck-{name}-{i}")
            raw = shaders.sample_params(sh, rng, p_vary=cfg["sampling"]["p_vary"])
            got, events = fil(sh, dict(raw), rng)
            n_draw += 1
            if events:
                n_with_event += 1
            for e in events:
                ev_by_rule[e["rule"]] += 1
                ev_by_param[f"{e['shader']}.{e['param']}"] += 1
                ev_by_shader[e["shader"]] += 1
                if e["was_default"]:
                    ev_default[f"{e['shader']}.{e['param']}"] += 1

            # ---- invariants on the DELIVERED values ----
            for k, v in got.items():
                d = defs.get(k)
                if d is None or isinstance(v, bool):
                    continue
                if pc._is_color(k, v, d):
                    L = pc.luma(v[:3])
                    ok = pc.LUMA_LO - 1e-6 <= L <= pc.LUMA_HI + 1e-6
                    if not ok and tuple(v) != tuple(d):
                        viol.append({"shader": name, "param": k, "val": list(v), "luma": L,
                                     "why": "colour luma out of band and not the default"})
                    continue
                comps = v if isinstance(v, (tuple, list)) else [v]
                dcomps = d if isinstance(d, (tuple, list)) else [d]
                for j, x in enumerate(comps):
                    if isinstance(x, bool) or not isinstance(x, (int, float)):
                        continue
                    dx = float(dcomps[j] if j < len(dcomps) else dcomps[0])
                    lo, hi = pc._band(dx, pc.REL_LO, pc.REL_HI)
                    if pc._has(k, pc.POWER_HINTS):
                        plo, phi = pc._band(dx, pc.POW_LO, pc.POW_HI)
                        lo, hi = max(lo, plo), min(hi, phi)
                        hi = min(hi, pc.POW_ABS)
                        lo = max(min(lo, hi), -pc.POW_ABS)
                        if abs(float(x)) > pc.POW_ABS + 1e-6:
                            viol.append({"shader": name, "param": k, "val": x,
                                         "why": f"power/brightness |v| > {pc.POW_ABS}"})
                    tol = 0.51 if isinstance(x, int) else 1e-3   # ints round to nearest
                    if not (lo - tol <= float(x) <= hi + tol):
                        viol.append({"shader": name, "param": k, "val": x, "default": dx,
                                     "band": [lo, hi], "why": "outside relative band"})
                    if pc._has(k, pc.POSITION_HINTS) and not (-1e-6 <= float(x) <= 1 + 1e-6):
                        viol.append({"shader": name, "param": k, "val": x,
                                     "why": "position/progress outside [0,1]"})
                    key = f"{name}.{k}"
                    worst[key] = max(worst.get(key, 0.0), abs(float(x)))

    top_param = ev_by_param.most_common(25)
    res = {
        "n_shaders": len(keep), "draws_per_shader": n, "n_draws": n_draw,
        "draws_with_at_least_one_clamp": n_with_event,
        "clamp_rate": round(n_with_event / n_draw, 4),
        "events_by_rule": dict(ev_by_rule.most_common()),
        "events_total": sum(ev_by_rule.values()),
        "events_per_draw": round(sum(ev_by_rule.values()) / n_draw, 3),
        "top_25_clamped_params": [{"param": k, "n": v} for k, v in top_param],
        "top_15_clamped_shaders": [{"shader": k, "n": v} for k, v in ev_by_shader.most_common(15)],
        "clamps_that_moved_a_DEFAULT_value": dict(ev_default.most_common()),
        "max_abs_delivered_value_top20": dict(sorted(worst.items(), key=lambda kv: -kv[1])[:20]),
        "colour_like_uniforms_in_keep_bank": vec3_seen,
        "n_violations": len(viol), "violations": viol[:20],
        "verdict": "PASS" if not viol else "FAIL",
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("top_25_clamped_params", "max_abs_delivered_value_top20",
                                   "colour_like_uniforms_in_keep_bank", "violations",
                                   "clamps_that_moved_a_DEFAULT_value")}, indent=2))
    print("top clamped params:", json.dumps(top_param[:12]))
    print("default-moving clamps:", json.dumps(dict(ev_default.most_common(10))))
    print(f"-> {OUT}")
    if viol:
        print(json.dumps(viol[:10], indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
