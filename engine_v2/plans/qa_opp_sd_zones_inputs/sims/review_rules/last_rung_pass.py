import pandas as pd, ast
D="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"
S=D+"NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
df=pd.read_csv(D+"NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
k=pd.read_csv(S+"_kl_zones.csv")
win={(0,0):(1,439,651),(0,1):(1,683,901),(1,0):(-1,727,747),(1,1):(-1,810,901),(1,2):(-1,935,1057)}
for _,r in k.iterrows():
    m=ast.literal_eval(r['meta']); key=(m['structure_id'],m['cycle_id']); sd,lo,hi=win[key]
    kind=r['source_kind']; steps=m['bounds_steps']
    side=r['side']; okey='bottom' if side=='buy' else 'top'
    def outer_at(i, inclusive):
        o=steps[0][okey]
        for s in steps[1:]:
            e=s['expanded_idx']
            if (e<=i if inclusive else e<i): o=s[okey]
        return o
    hits_s=[];hits_e=[]
    for i in range(lo,hi+1):
        c=df.c.iloc[i]
        for inc,h in ((False,hits_s),(True,hits_e)):
            o=outer_at(i,inc)
            beyond = c<o if side=='buy' else c>o
            if beyond: h.append(i)
    print(key,kind,side,"closes beyond outer: start-of-candle",hits_s[:12],"| incl own expansion",hits_e[:12])
