import pandas as pd, ast
D="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"
S=D+"NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
rec=pd.concat([pd.read_csv(S+f"_M15_{l}_triggers.csv") for l in ("counter","confluence")])
subs=pd.concat([pd.read_csv(S+f"_M15_{l}_subs.csv") for l in ("counter","confluence")]).drop_duplicates("sub_id").sort_values("sub_id")
print(subs[["sub_id","direction","starting_idx","start_idx","end_idx","end_reason","lenses"]].to_string(index=False))
print(rec[["lens","parent_sid","parent_cycle_id","sub_id","trigger_type","trigger_idx","probe_finalize_idx","start_idx","end_idx","end_reason","ended_by_sub_id","is_zero_length"]].sort_values(["sub_id","start_idx"]).to_string(index=False))
# per sub cycle: BOS zone inner + moment, POI inner + CTS_CONFIRMED moment
kl=pd.concat([pd.read_csv(S+f"_M15_{l}_kl_zones.csv") for l in ("counter","confluence")])
kl['m']=kl['meta'].apply(ast.literal_eval)
po=pd.concat([pd.read_csv(S+f"_M15_{l}_poi_zones.csv") for l in ("counter","confluence")])
po['m']=po['meta'].apply(ast.literal_eval)
ev=pd.concat([pd.read_csv(S+f"_M15_{l}_structure_events.csv") for l in ("counter","confluence")])
ev['m']=ev['meta'].apply(ast.literal_eval)
ev['sub']=ev['m'].apply(lambda m:m.get('sub_id')); ev['cyc']=ev['m'].apply(lambda m:m.get('cycle_id'))
seen=set()
for _,r in kl.iterrows():
    m=r['m']; key=(m['sub_id'],m['cycle_id'],r['source_kind'])
    if key in seen: continue
    seen.add(key)
    print("KL sub",m['sub_id'],"c",m['cycle_id'],r['source_kind'],r['side'],"inner",m['inner'],"outer",m['outer'],"conf",m['confirmed_idx'],"H1",m['confirmed_idx']//4,"end",m.get('end_idx'),m.get('end_reason'))
seen=set()
for _,r in po.iterrows():
    m=r['m']; key=(m.get('sub_id'),r['cycle_id'])
    if key in seen: continue
    seen.add(key)
    inner=r['top'] if r['side']=='buy' else r['bottom']; outer=r['bottom'] if r['side']=='buy' else r['top']
    cc=ev[(ev['sub']==m.get('sub_id'))&(ev['cyc']==r['cycle_id'])&(ev['type']=='CTS_CONFIRMED')]['idx'].tolist()
    print("POI sub",m.get('sub_id'),"c",r['cycle_id'],r['side'],"inner",inner,"outer",outer,"ic",r['ic_idx'],"band",m.get('ic_zone_band'),"conf",r['confirmed_idx'],"CTS_CONF",sorted(set(cc)),"end",m.get('end_idx'))
for t in ("BOS_CONFIRMED","CTS_CONFIRMED","REVERSAL_CONFIRMED","CTS_ESTABLISHED"):
    e=ev[ev['type']==t].drop_duplicates(["sub","cyc","idx"])
    print(t, sorted(zip(e['sub'],e['cyc'],e['idx'])))
