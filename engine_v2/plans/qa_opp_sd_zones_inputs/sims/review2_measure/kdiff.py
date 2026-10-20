import csv, re, ast, sys, io
from collections import Counter, defaultdict
P='artifacts/debug/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
S='artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5'
SQ={0:(1,454),1:(-1,1797),2:(1,2365),3:(-1,2639),4:(-1,3304),5:(1,3621),6:(-1,3760),7:(1,4027)}
NOW={0:(1,454),1:(-1,1797),2:(1,2365),3:(-1,2639),4:(1,3230),5:(-1,3304),6:(1,3621),7:(-1,3760),8:(1,4027),9:(-1,4179),10:(1,4183)}
def K(m,v):
    if v in ('',None,'None','nan'): return v
    try: return 'K%s'%(m[int(float(v))],)
    except Exception: return 'BAD%s'%v
def rows(p,m):
    out=defaultdict(Counter)
    with open(p,newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            key=None
            if 'sub_id' in row and row['sub_id'] not in ('',None):
                key=m[int(float(row['sub_id']))]
            meta=row.get('meta')
            if meta:
                mm=re.search(r"'sub_id': (\d+)",meta)
                if mm:
                    k2=m[int(mm.group(1))]
                    if key is None: key=k2
                    assert key==k2
                meta=re.sub(r"'sub_id': (\d+)",lambda g:"'sub_id': %s"%K(m,g.group(1)),meta)
                meta=re.sub(r"'cross_fib_limit_from_sub_id': (\d+)",lambda g:"'cross_fib_limit_from_sub_id': %s"%K(m,g.group(1)),meta)
                meta=re.sub(r"'ended_by_sub_id': (\d+)",lambda g:"'ended_by_sub_id': %s"%K(m,g.group(1)),meta)
                row['meta']=meta
            for c in ('sub_id','ended_by_sub_id','cross_fib_limit_from_sub_id'):
                if c in row: row[c]=K(m,row[c])
            if 'seq' in row: row['seq']='*'   # pool seq renumbering
            out[key][tuple(sorted(row.items()))]+=1
    return out
files=sys.argv[1:]
for l in ['counter','confluence']:
    for f in files:
        a=rows(S+'_M15_%s_%s.csv'%(l,f),SQ); b=rows(P+'_M15_%s_%s.csv'%(l,f),NOW)
        print('=== %s %s'%(l,f))
        for key in sorted(set(a)|set(b),key=lambda k:(k is None, k if k else (0,0))):
            ca,cb=a.get(key,Counter()),b.get(key,Counter())
            common=sum((ca&cb).values()); oa=sum((ca-cb).values()); ob=sum((cb-ca).values())
            print('   key',key,'SQ',sum(ca.values()),'NOW',sum(cb.values()),'common',common,'onlySQ',oa,'onlyNOW',ob)
