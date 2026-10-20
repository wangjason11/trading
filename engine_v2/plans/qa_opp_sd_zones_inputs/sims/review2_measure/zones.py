import pandas as pd, ast, sys
P='artifacts/debug/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
S='artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
def kl(p,l):
    e=pd.read_csv(p+'_M15_'+l+'_kl_zones.csv'); out=[]
    for _,r in e.iterrows():
        m=ast.literal_eval(r['meta'])
        out.append((m['sub_id'],m['cycle_id'],r.source_kind,m['confirmed_idx'],m['inner'],m['outer'],m.get('end_idx'),m.get('end_reason'),r.start_time,r.end_time,tuple((s.get('base_idx'),s['top'],s['bottom'],s['event'],s.get('idx'),s.get('expanded_idx')) for s in m['bounds_steps'])))
    return out
def poi(p,l):
    e=pd.read_csv(p+'_M15_'+l+'_poi_zones.csv'); out=[]
    for _,r in e.iterrows():
        m=ast.literal_eval(r['meta'])
        out.append((m['sub_id'],r.cycle_id,r.ic_idx,r.confirmed_idx,r.top,r.bottom,r.status,m.get('end_idx'),m.get('end_reason'),r.start_time,r.end_time))
    return out
which=sys.argv[1]
for l in ['counter','confluence']:
    for lab,p in (('SQ',S),('NOW',P)):
        print('=====',lab,l,which)
        for z in (kl(p,l) if which=='kl' else poi(p,l)): print('  ',z)
