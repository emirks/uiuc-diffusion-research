import json, csv, re, os
BASE="/taiga/illinois/eng/cs/jrehg/users/emirkisa/data_external/EffectData"
ann=json.load(open(f"{BASE}/annotations.json"))

# en -> zh from csv
zh={}
with open(f"{BASE}/effect_names_list.csv") as f:
    for row in csv.DictReader(f):
        zh[row["effect_name_en"]]=row["effect_name_zh"]

# group videos by effect class (video_path prefix)
from collections import defaultdict
groups=defaultdict(list)
for fn,v in ann.items():
    cls=v.get("video_path","").split("/")[0]
    groups[cls].append(v)

TRANS=re.compile(r"transform|become|turn(?:ing|s)?[ _]into|morph|melt|shatter|dissolv|explod|erupt|transition|reveal|emerg|sprout|grow|bloom|freez|petrif|crystalliz|disintegrat|shift|split|burst|unfold|manifest", re.I)

preview_dir=f"{BASE}/example_preview"
def preview_name(en):
    z=zh.get(en,"")
    cand=f"{en},{z}.mp4"
    return cand

out=[]
n_trans=0
for cls in sorted(groups):
    vids=groups[cls]
    rep=vids[0]
    text_blob=" ".join([cls, rep.get("vfx_en",""), rep.get("abstract_en",""),
                         rep.get("instruction_en",""), rep.get("prompt_en","")])
    is_trans=bool(TRANS.search(text_blob))
    if is_trans: n_trans+=1
    pv=preview_name(cls)
    has_pv=os.path.exists(f"{preview_dir}/{pv}")
    out.append({
        "cls": cls,
        "zh": zh.get(cls,""),
        "n": len(vids),
        "vfx_en": rep.get("vfx_en",""),
        "abstract_en": rep.get("abstract_en","") or cls.replace("_"," "),
        "prompt_en": rep.get("prompt_en",""),
        "instruction_en": rep.get("instruction_en",""),
        "trans": is_trans,
        "pv": pv,          # preview filename (may or may not exist yet)
    })

meta={"n_classes":len(out), "n_videos":len(ann),
      "n_transition_shaped":n_trans,
      "vids_per_class":{"min":min(len(g) for g in groups.values()),
                        "median":sorted(len(g) for g in groups.values())[len(groups)//2],
                        "max":max(len(g) for g in groups.values())}}
json.dump({"meta":meta,"effects":out}, open(f"{BASE}/effects_index.json","w"), ensure_ascii=False)
print(json.dumps(meta, ensure_ascii=False, indent=2))
print("transition-shaped classes: %d / %d (%.1f%%)" % (n_trans, len(out), 100*n_trans/len(out)))
print("index written: effects_index.json  (%.1f KB)" % (os.path.getsize(f"{BASE}/effects_index.json")/1e3))
