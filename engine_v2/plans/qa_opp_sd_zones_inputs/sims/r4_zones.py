import pandas as pd, ast, sys
D="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"
S=D+"NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
def zones(path):
    z=pd.read_csv(path); z['m']=z['meta'].apply(ast.literal_eval); return z
# 1) inner stability over every KL zone (main + both lenses)
for tag,p in [("H1",S+"_kl_zones.csv"),("M15c",S+"_M15_counter_kl_zones.csv"),("M15f",S+"_M15_confluence_kl_zones.csv")]:
    z=zones(p); n_in=0; n_out=0; kinds={}
    for _,r in z.iterrows():
        m=r['m']; side=r['side']; steps=m.get('bounds_steps',[])
        inner_key='top' if side=='buy' else 'bottom'; outer_key='bottom' if side=='buy' else 'top'
        ins={s[inner_key] for s in steps}; outs={s[outer_key] for s in steps}
        if len(ins)>1: n_in+=1
        if len(outs)>1: n_out+=1; kinds[r['source_kind']]=kinds.get(r['source_kind'],0)+1
    print(tag, "zones",len(z),"inner changed",n_in,"outer changed",n_out,kinds)
# 2) main cycles: BOS / CTS / POI inners
k=zones(S+"_kl_zones.csv"); p=zones(S+"_poi_zones.csv")
for _,r in k.iterrows():
    m=r['m']; print("KL",m['structure_id'],m['cycle_id'],r['source_kind'],r['side'],"inner",m['inner'],"outer",m['outer'],"conf",m['confirmed_idx'],"end",m.get('end_idx'),m.get('end_reason'))
for _,r in p.iterrows():
    m=r['m']; inner=r['top'] if r['side']=='buy' else r['bottom']; outer=r['bottom'] if r['side']=='buy' else r['top']
    print("POI",r['structure_id'],r['cycle_id'],r['side'],"inner",inner,"outer",outer,"ic",r['ic_idx'],"conf",r['confirmed_idx'],"end",m.get('end_idx'))
