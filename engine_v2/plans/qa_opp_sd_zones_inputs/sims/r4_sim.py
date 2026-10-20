"""Round-4 rules simulated on S_Q's H1 candles + S_Q's subs (ILLUSTRATIVE: the new rules create new subs that this
does not model). Zones are (label, inner, outer, avail_h1, live_until_h1, final, cycle_group)."""
import pandas as pd, math, sys
D="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"
df=pd.read_csv(D+"NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
PIP=0.0001; PROX=8; NARROW=40
av=lambda m: m//4+1          # an M15 moment m is known before H1 candle m//4+1 opens
def rnd(x): return int(math.floor(x+0.5))
def run(name, sd_dir, lo, hi, sd_z, opp_z, lock_default=False, prox=PROX, show=True):
    # sd_dir=+1: parent bullish (sd zones below, buy). closeness: sd -> max inner, opp -> min inner
    passed=set(); lock={'sd':None,'opp':None}; cnt={'sd':0,'opp':0}; prev=None; turn='sd'; out=[]
    def closer(side,a,b):  # is inner a closer to the gap centre than b
        if side=='sd': return a>b if sd_dir==1 else a<b
        return a<b if sd_dir==1 else a>b
    def pick(side,h,zones):
        if lock[side] is not None: return lock[side]
        avail=[z for z in zones if z[3]<=h<=z[4] and z[0] not in passed]
        # rule 2a: per sub keep only its most recent cycle (cycle_group = (sub, cycle)); parent zones always
        latest={}
        for z in avail:
            if z[6] is None: continue
            s,c=z[6]; latest[s]=max(latest.get(s,-1),c)
        avail=[z for z in avail if z[6] is None or z[6][1]==latest[z[6][0]]]
        best=None
        for z in avail:
            if best is None or closer(side,z[1],best[1]): best=z
        if best[5] and (lock_default or any(p in passed for p in [x[0] for x in zones])): lock[side]=best
        return best
    for h in range(lo,hi+1):
        r=df.iloc[h]
        s=pick('sd',h,sd_z); o=pick('opp',h,opp_z)
        gap=rnd(abs(s[1]-o[1])/PIP); narrow=gap<=NARROW
        if prev is not None:
            ps,po,pg,pn=prev
            if narrow and not pn: cnt={'sd':0,'opp':0}
            elif pn and gap>pg:
                if s[0]!=ps[0]: cnt['sd']=0
                if o[0]!=po[0]: cnt['opp']=0
            if (s[0],o[0])!=(ps[0],po[0]) and show:
                print(f"  {h}: refs sd={s[0]}({s[1]}) opp={o[0]}({o[1]}) gap={gap}{' NARROW' if narrow else ''} cnt={cnt}")
        elif show: print(f"  {h}: refs sd={s[0]}({s[1]}) opp={o[0]}({o[1]}) gap={gap}{' NARROW' if narrow else ''}")
        prev=(s,o,gap,narrow)
        z=s if turn=='sd' else o
        if turn=='sd': hit = r.l<=z[1]+prox*PIP if sd_dir==1 else r.h>=z[1]-prox*PIP
        else:          hit = r.h>=z[1]-prox*PIP if sd_dir==1 else r.l<=z[1]+prox*PIP
        if hit:
            if narrow and cnt[turn]>=1:
                if show: print(f"  {h}: {turn} BLOCKED on {z[0]} (narrow, cnt={cnt})")
            else:
                cnt[turn]+=1 if narrow else 0
                out.append((h,turn,z[0])); 
                if show: print(f"  {h}: TRIGGER {turn}:{z[0]} (l={r.l} h={r.h} c={r.c}) gap={gap}")
                turn='opp' if turn=='sd' else 'sd'
        # passes on this close (effective next candle)
        for side,zones in (('sd',sd_z),('opp',opp_z)):
            for zz in zones:
                if zz[5] or zz[0] in passed or not (zz[3]<=h<=zz[4]): continue
                beyond = (r.c<zz[2]) if ((side=='sd')==(sd_dir==1)) else (r.c>zz[2])
                if beyond:
                    passed.add(zz[0])
                    if show: print(f"  {h}: close {r.c} past {zz[0]} outer {zz[2]} -> passed")
    return out
INF=10**9
z01_sd=[("P.POI",0.57712,0.57676,683,INF,False,None),("P.BOS",0.57494,0.5736,683,INF,True,None)]
z01_opp=[("P.CTS",0.58404,0.58534,683,INF,True,None),
 ("s3c1.BOS",0.58366,0.58424,av(2829),av(3611)-1,False,(3,1)),("s3c1.POI",0.58279,0.58318,max(av(2844),av(2829)),av(3611)-1,False,(3,1)),
 ("s3c2.BOS",0.58171,0.58196,av(2992),av(3611)-1,False,(3,2)),("s3c2.POI",0.58096,0.58118,av(3086),av(3611)-1,False,(3,2)),
 ("s3c3.BOS",0.58055,0.58105,av(3589),av(3611)-1,False,(3,3))]
z12_sd=[("P.POI",0.57786,0.5785,935,INF,False,None),("P.BOS",0.57996,0.58105,935,INF,True,None),
 ("s4.POI",0.57885,0.5792,av(3741),av(3819)-1,False,(4,0)),("s4.BOS",0.58055,0.58105,av(3621),av(3819)-1,False,(4,0))]
z12_opp=[("P.CTS",0.57282,0.57112,935,INF,True,None),
 ("s5.BOS",0.57271,0.57112,av(3743),av(4143)-1,False,(5,0)),("s5.POI",0.57306,0.57276,av(3786),av(4143)-1,False,(5,0)),
 ("s7c0.BOS",0.57384,0.57362,av(4143),INF,False,(7,0)),("s7c0.POI",0.57484,0.57454,av(4143),INF,False,(7,0)),
 ("s7c1.BOS",0.57455,0.57413,av(4157),INF,False,(7,1)),("s7c1.POI",0.57462,0.57422,av(4159),INF,False,(7,1)),
 ("s7c2.BOS",0.57728,0.57691,av(4179),INF,False,(7,2))]
lockdef = len(sys.argv)>1 and sys.argv[1]=="lockdefault"
prox = int(sys.argv[2]) if len(sys.argv)>2 else PROX
print("(0,1)", "lock_default" if lockdef else "lock_on_pass", "prox",prox); print(run("01",1,683,901,z01_sd,z01_opp,lockdef,prox))
print("(1,2)", "lock_default" if lockdef else "lock_on_pass", "prox",prox); print(run("12",-1,935,1057,z12_sd,z12_opp,lockdef,prox))
