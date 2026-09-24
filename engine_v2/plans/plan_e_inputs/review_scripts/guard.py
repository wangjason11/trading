import sys, io, contextlib, time, collections
sys.path.insert(0, r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2")
import engine_v2.tests.test_render_sub_projection as T
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.sub_structure_pool import SubStructurePool
t0=time.time()
with contextlib.redirect_stdout(io.StringIO()):
    m15 = T.m15_df.__wrapped__() if hasattr(T.m15_df,'__wrapped__') else T._prepare_df(T._make_flat_prefix()+T._make_reversing_data())
    pool = SubStructurePool()
    sub, _ = edm.build_or_get_geometry(pool, m15, parent_path=T._PARENT, sd=1, start_abs=T._STARTING_IDX, bos0_inner=None, timeframe=T._TF)
    T._two_lens_records(pool, sub, m15)
    T._set_lifecycle(sub, T._START, None, None)
    lens = T._lens_dfs(m15)
    T._render(sub, m15, lens)
keys=collections.Counter(); types=collections.Counter()
for ln, d in lens.items():
    for ev in d.attrs.get("events", []):
        types[ev.type]+=1
        for k in ev.meta:
            if k.endswith("_idx") or k.endswith("_at") or k in ("pb_start",):
                keys[(ev.type,k, k in edm._EVENT_META_IDX_KEYS)]+=1
    zk=collections.Counter()
    for z in d.attrs.get("kl_zones", [])+d.attrs.get("poi_zones", []):
        for k in z.meta:
            if k.endswith("_idx") or k.endswith("_at"): zk[(k, k in edm._ZONE_META_IDX_KEYS)]+=1
print("secs", round(time.time()-t0,2)); print(types)
for k,v in sorted(keys.items()): print(k,v)
print("zone keys", sorted(zk.items()))
