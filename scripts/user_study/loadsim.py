#!/usr/bin/env python3
"""Simulate N raters rating full sets concurrently against a study server. Media GETs included to mimic real load.
usage: loadsim.py BASE_URL N_RATERS [abandon_frac] [media 0/1]"""
import sys, json, time, random, threading, urllib.request, collections
BASE=sys.argv[1].rstrip('/'); N=int(sys.argv[2]); ABANDON=float(sys.argv[3]) if len(sys.argv)>3 else 0.0; MEDIA=int(sys.argv[4]) if len(sys.argv)>4 else 1
def req(path, body=None, timeout=60):
    data=json.dumps(body).encode() if body is not None else None
    r=urllib.request.Request(BASE+path, data=data, headers={"Content-Type":"application/json"} if data else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp: return resp.status, json.loads(resp.read())
lock=threading.Lock(); log=[]; errors=[]; lat=collections.defaultdict(list); sessions={}
def rater(i):
    pid=f"sim{i:03d}"; rnd=random.Random(1000+i)
    try:
        t=time.time(); st,s=req("/api/session",{"rater_id":"prolific:"+pid,"nickname":pid,"prolific":{"pid":pid,"study_id":"S","session_id":f"s{i}"}}); lat["session"].append(time.time()-t)
        items=s["items"]; sessions[pid]=(s["session_id"],[it["pair_id"] for it in items])
        n_do=len(items) if rnd.random()>=ABANDON else rnd.randint(1,len(items)-1)
        for k,it in enumerate(items[:n_do]):
            if MEDIA:
                for u in (it["a"],it["b"],it["ref"]):
                    t=time.time()
                    with urllib.request.urlopen(BASE+"/"+u, timeout=120) as resp: resp.read()
                    lat["media"].append(time.time()-t)
            fl=lambda: {"no_effect":rnd.random()<0.1,"endpoint":rnd.random()<0.15,"leak":rnd.random()<0.1}
            body={"session_id":s["session_id"],"pair_id":it["pair_id"],"answers":{"transition":rnd.choice("AB"),"overall":rnd.choice("AB")},"flags":{"A":fl(),"B":fl()},"seconds":rnd.randint(8,40),"replays":0}
            t=time.time(); st,r=req("/api/rating",body); lat["rating"].append(time.time()-t)
            with lock: log.append((pid,it["pair_id"],st))
        if n_do==len(items):
            st,c=req("/api/complete",{"session_id":s["session_id"]})
            with lock: log.append((pid,"complete",c.get("complete")))
    except Exception as e:
        with lock: errors.append((pid,repr(e)[:200]))
t0=time.time(); th=[threading.Thread(target=rater,args=(i,)) for i in range(N)]
[t.start() for t in th]; [t.join() for t in th]
el=time.time()-t0
n_rat=sum(1 for l in log if l[1]!="complete"); n_bad=sum(1 for l in log if l[1]!="complete" and l[2]!=200)
n_comp=sum(1 for l in log if l[1]=="complete" and l[2] is True)
print(f"raters={N} abandon={ABANDON} elapsed={el:.1f}s ratings={n_rat} non200={n_bad} completes={n_comp} errors={len(errors)}")
for k,v in lat.items(): v.sort(); print(f"  {k}: n={len(v)} median={v[len(v)//2]*1000:.0f}ms p95={v[int(len(v)*.95)]*1000:.0f}ms max={v[-1]*1000:.0f}ms")
for e in errors[:5]: print("  ERR", e)
# per-session sanity: unique pairs, unique rows
from collections import Counter
dup=[p for p,(sid,pids) in sessions.items() if len(set(pids))!=len(pids)]
print(f"  sessions={len(sessions)} with duplicate pair ids: {len(dup)}")
json.dump({"sessions":sessions}, open("/taiga/illinois/eng/cs/jrehg/users/emirkisa/tmp/loadsim_sessions.json","w"))
