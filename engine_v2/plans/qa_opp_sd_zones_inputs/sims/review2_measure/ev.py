import pandas as pd, ast, sys
P='artifacts/debug/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
S='artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
def load(p):
    rows=[]
    for l in ['counter','confluence']:
        e=pd.read_csv(p+'_M15_'+l+'_structure_events.csv')
        for _,r in e.iterrows():
            m=ast.literal_eval(r['meta']) if isinstance(r['meta'],str) else {}
            rows.append(dict(lens=l,idx=r['idx'],type=r['type'],price=r['price'],sub=m.get('sub_id'),cyc=m.get('cycle_id'),dir=m.get('struct_direction'),conf=m.get('confirmed_at'),ps=m.get('parent_sid'),pc=m.get('parent_cycle_id'),m=m))
    return pd.DataFrame(rows)
now=load(P)
want=sys.argv[1:] if len(sys.argv)>1 else None
for (l,s),g in now.groupby(['lens','sub']):
    sel=g[g.type.isin(['BOS_CONFIRMED','CTS_CONFIRMED','CTS_ESTABLISHED'])]
    print(l,'sub',s,'n_events',len(g),'idx range',g.idx.min(),g.idx.max())
    for _,r in sel.iterrows():
        print('   ',r.idx,r.type,r.price,'cyc',r.cyc,'dir',r.dir,'parent',(r.ps,r.pc))
