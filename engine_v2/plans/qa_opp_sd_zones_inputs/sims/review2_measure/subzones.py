import sys, csv, ast
sys.argv=[sys.argv[0]]
from engine_v2.tests import _plan_r_reference_window as W
import pandas as pd
P='artifacts/debug/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
S='artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
SQ={0:(1,454),1:(-1,1797),2:(1,2365),3:(-1,2639),4:(-1,3304),5:(1,3621),6:(-1,3760),7:(1,4027)}
NOW={0:(1,454),1:(-1,1797),2:(1,2365),3:(-1,2639),4:(1,3230),5:(-1,3304),6:(1,3621),7:(-1,3760),8:(1,4027),9:(-1,4179),10:(1,4183)}
def zones(p,m):
    out={}
    for l in ('counter','confluence'):
        for r in csv.DictReader(open(p+'_M15_%s_kl_zones.csv'%l,encoding='utf-8')):
            mm=ast.literal_eval(r['meta']); k=m[mm['sub_id']]
            steps=[(s.get('expanded_idx'),s['top'] if mm['struct_direction']==-1 else s['bottom']) for s in mm['bounds_steps']]
            out.setdefault(k,{})[(mm['cycle_id'],'BOS')]=(mm['inner'],steps)
        for r in csv.DictReader(open(p+'_M15_%s_poi_zones.csv'%l,encoding='utf-8')):
            mm=ast.literal_eval(r['meta']); k=m[mm['sub_id']]; sd=mm['struct_direction']
            inner=float(r['top']) if sd==1 else float(r['bottom']); outer=float(r['bottom']) if sd==1 else float(r['top'])
            out.setdefault(k,{})[(int(r['cycle_id']),'POI')]=(inner,[(None,outer)])
    return out
def events(p,m):
    out={}
    for l in ('counter','confluence'):
        for r in csv.DictReader(open(p+'_M15_%s_structure_events.csv'%l,encoding='utf-8')):
            if r['type'] not in ('BOS_CONFIRMED','CTS_CONFIRMED'): continue
            mm=ast.literal_eval(r['meta']); k=m[mm['sub_id']]
            out.setdefault(k,{})[(mm['cycle_id'],'BOS' if r['type']=='BOS_CONFIRMED' else 'POI')]=int(r['idx'])
    return out
zn,zs=zones(P,NOW),zones(S,SQ); en,es=events(P,NOW),events(S,SQ)
for key,zl in W.SUB_ZONES.items():
    for cyc,kind,inner,usable,steps in zl:
        src='NOW'; z=zn.get(key,{}).get((cyc,kind)); e=en.get(key,{}).get((cyc,kind))
        if z is None: src='SQ'; z=zs.get(key,{}).get((cyc,kind)); e=es.get(key,{}).get((cyc,kind)) if e is None else e
        if z is None:
            print(key,cyc,kind,'NOT IN ANY CSV', inner, usable, steps); continue
        ok_inner=abs(z[0]-inner)<1e-9; ok_use=(e==usable)
        csteps=z[1]; init_ok=abs(csteps[0][1]-steps[0][1])<1e-9
        if kind=='BOS':
            vs=[(a,b) for a,b in csteps[1:]]; es_=list(steps[1:])
            steps_ok = vs==[(a,b) for a,b in es_[:len(vs)]] 
            extra = es_[len(vs):]
        else: steps_ok=True; extra=[]
        flag='' if (ok_inner and ok_use and init_ok and steps_ok) else '  <<< MISMATCH'
        print(key,cyc,kind,src,'inner',ok_inner,'usable',usable,'evt',e,'initouter',init_ok,'steps',steps_ok,'view-extra-steps',extra,flag)
    # zones in CSV not in view
    vk={(c,k) for c,k,*_ in zl}
    for ck in zn.get(key,{}):
        if ck not in vk: print(key,ck,'IN NOW CSV, NOT IN VIEW')
