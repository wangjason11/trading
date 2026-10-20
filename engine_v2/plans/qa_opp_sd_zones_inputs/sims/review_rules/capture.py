import sys, pickle, time
OUT=sys.argv[1]; sys.argv=[sys.argv[0]]
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
from engine_v2.zones.zone_proximity import check_zone_proximity as _orig
CAP={}
def patched(**kw):
    out=_orig(**kw)
    CAP['kw']=kw; CAP['out']=out
    return out
orch.check_zone_proximity=patched
df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"]=CONFIG.pair
t0=time.perf_counter()
res=orch.run_pipeline(df, lower_timeframes=())
print("pipeline", time.perf_counter()-t0)
kw=CAP['kw']
dfx=kw['df']
pickle.dump({'df':dfx[['o','h','l','c']].copy(),'events':kw['sorted_events'],'kl':kw['kl_zones'],'poi':kw['poi_zones'],
             'pip':kw['pip_size'],'tf':kw['timeframe'],'out':CAP['out']}, open(OUT,'wb'))
print({k:[(t.idx,t.direction,t.zone_kind,t.trigger_inner) for t in v] for k,v in CAP['out'].items()})
