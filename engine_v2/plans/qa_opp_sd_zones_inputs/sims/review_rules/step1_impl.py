"""A faithful-as-possible implementation of PLAN_R step 1 (parent-only ladders + passes + lock + the cap, AND-ed with
today's Rules 2/3), run on the captured real inputs of check_zone_proximity. Variants via flags."""
import sys, pickle, math, itertools
from engine_v2.zones import zone_proximity as zp
from engine_v2.structure import event_fields as ef
from engine_v2.zones.structure_lifecycle import compute_reversal_idx_by_sid
C=pickle.load(open(sys.argv[1],'rb'))
df,events,kl,poi,pip=C['df'],C['events'],C['kl'],C['poi'],C['pip']
PROX=8; MIN_GAP=50; NARROW=40
def outer_at(z,i,inclusive):
    steps=z.meta.get('bounds_steps')
    side=z.side
    okey='bottom' if side=='buy' else 'top'
    o=steps[0][okey]
    for s in steps[1:]:
        e=s['expanded_idx']
        if (e<=i if inclusive else e<i): o=s[okey]
    return o
def run(last_rung_exempt=True, outer_incl=False, pass_on_blocked=True, use_cap=True, verbose=False):
    cts_conf={}; bos_conf={}
    for ev in events:
        k=(ev.meta.get('structure_id',0),ev.meta.get('cycle_id',0))
        if ev.type=='CTS_CONFIRMED': cts_conf[k]=ev
        if ev.type=='BOS_CONFIRMED': bos_conf[k]=ef.event_moment(ev)
    rev=compute_reversal_idx_by_sid(events)
    out={}; notes={}
    for key,cev in cts_conf.items():
        sid,cyc=key; sd=int(cev.meta.get('struct_direction',0))
        bz=zp._find_kl_zone_for_cycle(kl,sid,cyc,'BOS'); cz=zp._find_kl_zone_for_cycle(kl,sid,cyc,'CTS')
        bos_inner=float(bz.meta['inner']); cts_inner=float(cz.meta['inner'])
        pois=[p for p in poi if p.meta.get('structure_id')==sid and p.meta.get('cycle_id')==cyc]
        s0=ef.event_moment(cev); e0=len(df)-1
        if (sid,cyc+1) in bos_conf: e0=min(e0,bos_conf[(sid,cyc+1)]-1)
        if sid in rev: e0=min(e0,rev[sid]-1)
        if s0>e0: continue
        # ladders: (label, inner, outer_fn, last)
        sd_l=[]
        for p in pois:
            inner=float(p.top) if sd==1 else float(p.bottom); outer=float(p.bottom) if sd==1 else float(p.top)
            if (inner>bos_inner if sd==1 else inner<bos_inner):
                sd_l.append(('POI',inner,(lambda i,o=outer:o),False))
        sd_l.append(('BOS',bos_inner,(lambda i,z=bz:outer_at(z,i,outer_incl)),True))
        opp_l=[('CTS',cts_inner,(lambda i,z=cz:outer_at(z,i,outer_incl)),True)]
        closer={'sd':(lambda a,b:a>b) if sd==1 else (lambda a,b:a<b),'opp':(lambda a,b:a<b) if sd==1 else (lambda a,b:a>b)}
        passed=set(); lock={'sd':None,'opp':None}; prevref={'sd':None,'opp':None}
        def pick(side,lad):
            if lock[side] is not None: return lock[side]
            cand=[z for z in lad if z[0] not in passed]
            if not cand: return None
            best=cand[0]
            for z in cand[1:]:
                if closer[side](z[1],best[1]): best=z
            if best[3] and prevref[side] is not None and prevref[side][0] in passed: lock[side]=best
            return best
        tl=zp._build_cycle_threshold_timeline(events,sid,cyc); rc=rb=None; ptr=0
        while ptr<len(tl) and ef.event_moment(tl[ptr])<=s0:
            rc,rb=zp._apply_threshold_event(tl[ptr],rc,rb); ptr+=1
        pull=(cev.meta.get('confirmation_method','pullback')=='pullback')
        trig=[]; exp='sd'; nsd=nopp=False
        cnt=None; prev_gap=None; prev_narrow=None
        for i in range(s0,e0+1):
            while ptr<len(tl) and ef.event_moment(tl[ptr])<i:
                rc,rb=zp._apply_threshold_event(tl[ptr],rc,rb); ptr+=1
            s=pick('sd',sd_l); o=pick('opp',opp_l)
            # gap + counters
            if s is None or o is None:
                gap=None; narrow=False
            else:
                g=sd*(o[1]-s[1])/pip; gap=math.floor(g+0.5+1e-9); narrow=gap<=NARROW
            if prev_narrow is None:
                cnt={'sd':0,'opp':0} if narrow else None
            else:
                if narrow and not prev_narrow: cnt={'sd':0,'opp':0}
                elif narrow and prev_narrow:
                    for side,ref in (('sd',s),('opp',o)):
                        pr=prevref[side]
                        if pr is not None and ref is not None and pr[1]!=ref[1]:
                            outward = (ref[1]<pr[1]) if (side=='sd')==(sd==1) else (ref[1]>pr[1])
                            if outward: cnt[side]=0
                elif not narrow: cnt=None
            prevref['sd']=s; prevref['opp']=o; prev_narrow=narrow
            blocked=False
            isn=(rc is not None and rb is not None and abs(rc-rb)/pip<MIN_GAP)
            if isn:
                if not pull: blocked=True
                if exp=='sd' and nsd: blocked=True
                if exp=='opp_sd' and nopp: blocked=True
            side='sd' if exp=='sd' else 'opp'
            if use_cap and narrow and cnt[side]>=1: blocked=True
            if not blocked:
                z=s if exp=='sd' else o
                if z is None:
                    z=None
                lo_,hi_=df.l.iloc[i] if False else df.loc[i,'l'],df.loc[i,'h']
                if z is None: hit=False
                elif exp=='sd': hit=(lo_<=z[1]+PROX*pip) if sd==1 else (hi_>=z[1]-PROX*pip)
                elif True: hit=(hi_>=z[1]-PROX*pip) if sd==1 else (lo_<=z[1]+PROX*pip)
                if z is None: hit=False
                if hit:
                    trig.append((i,exp,z[0],z[1],gap))
                    if isn:
                        if exp=='sd': nsd=True
                        else: nopp=True
                    if narrow: cnt[side]+=1
                    exp='opp_sd' if exp=='sd' else 'sd'
            if blocked and not pass_on_blocked: continue
            c=df.loc[i,'c']
            for sname,lad in (('sd',sd_l),('opp',opp_l)):
                if lock[sname] is not None: continue
                for z in lad:
                    if z[0] in passed: continue
                    if z[3] and last_rung_exempt: continue
                    ou=z[2](i)
                    beyond=(c<ou) if (sname=='sd')==(sd==1) else (c>ou)
                    if beyond:
                        passed.add(z[0]); notes.setdefault(key,[]).append((i,sname,z[0],ou,c))
        if trig: out[key]=trig
    return out,notes
today={k:[(t.idx,t.direction,t.zone_kind,t.trigger_inner) for t in v] for k,v in C['out'].items()}
print("today", today)
for flags in [dict(),dict(last_rung_exempt=False),dict(outer_incl=True),dict(pass_on_blocked=False),dict(last_rung_exempt=False,outer_incl=True)]:
    o,n=run(**flags)
    oo={k:[(a,b,c,d) for a,b,c,d,_ in v] for k,v in o.items()}
    print(flags, "SAME" if oo==today else "DIFF", oo if oo!=today else "")
    print("   passes:", {k:v[:4] for k,v in n.items()})
    print("   gaps at triggers:", {k:[(a,e) for a,_,_,_,e in v] for k,v in o.items()})
