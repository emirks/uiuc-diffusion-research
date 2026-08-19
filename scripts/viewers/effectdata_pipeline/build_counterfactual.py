"""Build the counterfactual galleries by extracting targeted clips from remote zips.
  Page A (same operator, different endpoints): operators x their subjects.
  Page B (same endpoints, different operator): subjects x their operators.
Writes cf_media/*.mp4 and cf_data.json."""
import json, os, re, concurrent.futures as cf
from collections import defaultdict
import remote_zip as rz

BASE="/taiga/illinois/eng/cs/jrehg/users/emirkisa/data_external/EffectData"
ZIPURL="https://huggingface.co/datasets/ysy31415926/EffectData/resolve/main/Videos/%s.zip"
ann=json.load(open(f"{BASE}/annotations.json"))
idx={e["cls"]:e for e in json.load(open(f"{BASE}/effects_index.json"))["effects"]}

# maps
id_ops=defaultdict(dict)      # id -> {effect: (member, prompt, tag)}
op_subj=defaultdict(list)     # effect -> [(id, member, prompt, tag)]
for k,v in ann.items():
    eff=v["video_path"].split("/")[0]
    parts=k[:-4].split(",")
    if len(parts)<3: continue
    cid=parts[-2].strip(); tag=parts[-1].strip()
    id_ops[cid].setdefault(eff,(v["video_path"], v.get("prompt_en",""), tag))
    op_subj[eff].append((cid, v["video_path"], v.get("prompt_en",""), tag))

def fam(eff):  # keyword family for diversity
    return re.split(r"[_,]", eff.lower())[0]

def subj_label(cid, tag):
    m=re.match(r"^([a-zA-Z]+)-", cid)
    if m: return m.group(1), cid.split("-",1)[1][:8], "animal"
    kind={"F":"person ♀","M":"person ♂","Z":"subject"}.get(tag,"subject")
    return kind, cid[:10], ("female" if tag=="F" else "male" if tag=="M" else "subject")

def diversify(items, keyfn, transfn, n):
    # prefer transition-shaped, one-per-family round robin, then fill
    items=sorted(items, key=lambda x:(0 if transfn(x) else 1, keyfn(x)))
    seen=set(); out=[]
    for it in items:
        f=keyfn(it)
        if f in seen: continue
        seen.add(f); out.append(it)
        if len(out)>=n: return out
    for it in items:
        if it in out: continue
        out.append(it)
        if len(out)>=n: return out
    return out

# ---- SAME-ENDPOINT page (subjects) ----
HERO=["wolf-97860e78","octopus-d25305d8","bird-00246b4f","elephant-7473f4b",
      "baboon-74bf78ab","cat-d476d3ed","zebra-73782f6a","dog-b54a1706"]
N_OPS=9
subjects=[]
tasks={}   # out_path -> (effect, member)
def safe(s): return re.sub(r"[^A-Za-z0-9._-]","_",s)
for cid in HERO:
    ops=list(id_ops[cid].items())     # [(eff,(member,prompt,tag))]
    chosen=diversify(ops, keyfn=lambda x:fam(x[0]),
                     transfn=lambda x:idx.get(x[0],{}).get("trans",False), n=N_OPS)
    lab,sub,kind=subj_label(cid, chosen[0][1][2])
    clips=[]
    for eff,(member,prompt,tag) in chosen:
        out=f"cf_media/{safe(cid)}__{safe(eff)}.mp4"; tasks[out]=(eff,member)
        clips.append({"op":eff,"op_label":idx.get(eff,{}).get("abstract_en",eff.replace('_',' ')),
                      "trans":idx.get(eff,{}).get("trans",False),"file":out,"prompt":prompt})
    subjects.append({"id":cid,"label":lab,"sub":sub,"kind":kind,"n_ops":len(id_ops[cid]),"clips":clips})

# ---- SAME-OPERATOR page (operators) ----
OPS=["Chest_Fire_Burst","Ice_Crystal_Wings","Shard_Wings","Code_wings_on_back",
     "Ribbons_from_eyes","Laser_Hair_Tendrils","Back_unfurling_wings","Blossom_trail"]
N_SUBJ=9
operators=[]
for eff in OPS:
    rows=op_subj.get(eff,[])
    # prefer hero animals first, then diversify by subject prefix, mix tags
    def skey(r):
        cid,tag=r[0],r[3]
        anim=re.match(r"^([a-zA-Z]+)-",cid)
        return (anim.group(1) if anim else "zz_"+tag)
    rows_sorted=sorted(rows, key=lambda r:(0 if r[0] in set(HERO) else 1, skey(r)))
    seen=set(); chosen=[]
    for r in rows_sorted:
        f=skey(r)
        if f in seen: continue
        seen.add(f); chosen.append(r)
        if len(chosen)>=N_SUBJ: break
    clips=[]
    for cid,member,prompt,tag in chosen:
        out=f"cf_media/{safe(cid)}__{safe(eff)}.mp4"; tasks[out]=(eff,member)
        lab,sub,kind=subj_label(cid,tag)
        clips.append({"id":cid,"subj_label":lab,"subj_sub":sub,"kind":kind,"file":out,"prompt":prompt})
    e=idx.get(eff,{})
    operators.append({"op":eff,"label":e.get("abstract_en",eff.replace('_',' ')),
                      "abstract":e.get("abstract_en",""),"instruction":e.get("instruction_en",""),
                      "vfx":e.get("vfx_en",""),"trans":e.get("trans",False),
                      "n_subj":len(rows),"clips":clips})

# ---- extract, grouped by zip, parallel ----
os.makedirs(f"{BASE}/cf_media",exist_ok=True)
by_zip=defaultdict(list)
for out,(eff,member) in tasks.items():
    full=os.path.join(BASE,out)
    if os.path.exists(full) and os.path.getsize(full)>1024: continue
    by_zip[eff].append((member,full))
print("clips needed:", len(tasks), "| to fetch:", sum(len(v) for v in by_zip.values()), "| zips:", len(by_zip))

def do_zip(eff, members):
    url=ZIPURL % eff
    try:
        zf,hf=rz.open_zip(url)
        got=0
        for member,full in members:
            try:
                data=zf.read(member)
                os.makedirs(os.path.dirname(full),exist_ok=True)
                open(full,"wb").write(data); got+=1
            except Exception as e:
                print("  member fail",eff,member,e)
        return eff,got,hf.bytes_fetched
    except Exception as e:
        print("  ZIP FAIL",eff,e); return eff,0,0

with cf.ThreadPoolExecutor(max_workers=8) as ex:
    futs=[ex.submit(do_zip,eff,mem) for eff,mem in by_zip.items()]
    tot=0
    for f in cf.as_completed(futs):
        eff,got,by=f.result(); tot+=by
        print("  %-34s %d clips  %.1f MB" % (eff,got,by/1e6))
print("total fetched: %.1f MB" % (tot/1e6))

json.dump({"subjects":subjects,"operators":operators,
           "validation":{"subject":"elephant-7473f4b","first_frame_diff":0.99,"last_frame_diff":68.33,
                         "note":"same subject-id => identical first frame (0.99/255); operators diverge by end (68/255)"}},
          open(f"{BASE}/cf_data.json","w"), ensure_ascii=False)
print("wrote cf_data.json:", len(subjects),"subjects,",len(operators),"operators")
