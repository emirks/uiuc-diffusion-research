#!/usr/bin/env python
"""Tag S2 clips belonging to the full-occlusion shader family.

Advisor ruling 2026-07-28 (DOSSIER §10.7): the family is KEPT in both S2a and S2b,
unchanged -- no clip dropped, moved or reweighted. It is only TAGGED, so that the
pre-registered post-hoc diagnostic can split on it if the candidate fails its bars.

The family is defined BY SHADER NAME, exactly these six. The raters' looser
"noise/glitch/wave/mosaic/blur" wording is deliberately NOT the definition: the six
names are the replicated intersection of two independent blind raters and are
operationally crisp.
"""
import json, glob, sys, pathlib

FAMILY = {"ButterflyWaveScrawler", "StaticFade", "squeeze", "GridFlip", "flyeye", "CrossZoom"}
MAIN = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
WT = pathlib.Path(__file__).resolve().parents[3]
HALVES = {
    "S2a": MAIN / "outputs/videos/ctt_v2_s2/full/meta",
    "S2b": WT / "outputs/videos/ctt_v2_s2_humanvid/full/meta",
}
OUT = WT / "data/processed/ctt_v2_strata/s2_full_occlusion_tags.json"

def main():
    tags, summary = {}, {}
    for half, meta in HALVES.items():
        shards = sorted(glob.glob(str(meta / "clips_shard*.jsonl")))
        if not shards:
            sys.exit(f"[tag] no shards under {meta}")
        n = fam = 0
        per_shader = {}
        for f in shards:
            for line in open(f):
                r = json.loads(line)
                n += 1
                is_fam = r["shader"] in FAMILY
                fam += is_fam
                if is_fam:
                    per_shader[r["shader"]] = per_shader.get(r["shader"], 0) + 1
                tags[f"{half}/{r['stem']}"] = {"shader": r["shader"], "full_occlusion_family": is_fam}
        summary[half] = {"n_clips": n, "n_family": fam, "pct": round(100 * fam / n, 2),
                         "per_shader": dict(sorted(per_shader.items()))}
        print(f"[tag] {half}: {fam}/{n} = {summary[half]['pct']}% in family")
    # every family shader must actually be present in both halves, else the definition is stale
    for half, s in summary.items():
        missing = FAMILY - set(s["per_shader"])
        assert not missing, f"{half}: family shaders absent from manifest: {missing}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ruling": "DOSSIER 10.7 -- KEEP, tag only; no clip dropped or reweighted",
               "family_shaders": sorted(FAMILY), "summary": summary, "tags": tags},
              open(OUT, "w"), indent=1)
    tot_n = sum(s["n_clips"] for s in summary.values())
    tot_f = sum(s["n_family"] for s in summary.values())
    print(f"[tag] TOTAL S2: {tot_f}/{tot_n} = {100*tot_f/tot_n:.2f}%  ->  {OUT}")

if __name__ == "__main__":
    main()
