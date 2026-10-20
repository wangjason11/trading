import pandas as pd, ast
S="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
tot=0; uniq=set(); moved_in=[]; moved_out=[]
for tag,suf in [("H1","_kl_zones.csv"),("M15counter","_M15_counter_kl_zones.csv"),("M15confluence","_M15_confluence_kl_zones.csv")]:
    z=pd.read_csv(S+suf)
    for r in z.itertuples():
        m=ast.literal_eval(r.meta); tot+=1
        ik,ok=("top","bottom") if r.side=="buy" else ("bottom","top")
        ins={s[ik] for s in m["bounds_steps"]}; outs=[s[ok] for s in m["bounds_steps"]]
        key=(tag if tag=="H1" else "M15", m.get("sub_id"), m["structure_id"], m["cycle_id"], r.source_kind); uniq.add(key)
        assert m["inner"] in ins
        if len(ins)>1: moved_in.append((tag,key))
        if len(set(outs))>1: moved_out.append((tag,key,r.source_kind,outs[0],outs[-1]))
print("rows",tot,"unique zones",len(uniq),"inner moved",len(moved_in))
print("outer moved",len(moved_out)); [print("  ",x) for x in moved_out]
