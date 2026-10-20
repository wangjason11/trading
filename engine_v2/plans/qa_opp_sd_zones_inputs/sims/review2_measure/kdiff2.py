import csv, re, sys
from collections import defaultdict
sys.path.insert(0, r'C:\Users\wangj\AppData\Local\Temp\rv2_meas')
P='artifacts/debug/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
S='artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
SQ={0:(1,454),1:(-1,1797),2:(1,2365),3:(-1,2639),4:(-1,3304),5:(1,3621),6:(-1,3760),7:(1,4027)}
NOW={0:(1,454),1:(-1,1797),2:(1,2365),3:(-1,2639),4:(1,3230),5:(-1,3304),6:(1,3621),7:(-1,3760),8:(1,4027),9:(-1,4179),10:(1,4183)}
import ast
def norm(v):
    try:
        f=float(v)
        return repr(f) if f==f else 'nan'
    except Exception: return v
def load(p,m,idcols):
    out=defaultdict(dict)
    with open(p,newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            meta={}
            if row.get('meta'):
                meta=ast.literal_eval(row['meta']) if row['meta'].startswith('{') else {}
            sid=row.get('sub_id') or meta.get('sub_id')
            key=m[int(float(sid))]
            flat={k:norm(v) for k,v in row.items() if k!='meta'}
            for k,v in meta.items(): flat['meta.'+k]=norm(str(v)) if not isinstance(v,(list,dict,tuple)) else repr(v)
            for c in ('sub_id','meta.sub_id','ended_by_sub_id','meta.cross_fib_limit_from_sub_id','meta.ended_by_sub_id'):
                if c in flat and flat[c] not in ('','None','nan'): flat[c]=m[int(float(flat[c]))]
            flat.pop('seq',None)
            ident=tuple(flat.get(c) for c in idcols)
            out[key].setdefault(ident,[]).append(flat)
    return out
f=sys.argv[1]; idcols=sys.argv[2].split(','); keys=[ast.literal_eval(k) for k in sys.argv[3:]] if len(sys.argv)>3 else None
for l in ['counter','confluence']:
    a=load(S+'_M15_%s_%s.csv'%(l,f),SQ,idcols); b=load(P+'_M15_%s_%s.csv'%(l,f),NOW,idcols)
    for key in sorted(set(a)|set(b)):
        if keys and key not in keys: continue
        ra,rb=a.get(key,{}),b.get(key,{})
        for ident in sorted(set(ra)|set(rb),key=str):
            xa,xb=ra.get(ident),rb.get(ident)
            if xa is None: print(l,key,ident,'ONLY NOW'); continue
            if xb is None: print(l,key,ident,'ONLY SQ'); continue
            if len(xa)!=len(xb): print(l,key,ident,'count',len(xa),len(xb)); continue
            for u,v in zip(xa,xb):
                d={c:(u.get(c),v.get(c)) for c in set(u)|set(v) if u.get(c)!=v.get(c)}
                if d: print(l,key,ident,'DIFF',d)
