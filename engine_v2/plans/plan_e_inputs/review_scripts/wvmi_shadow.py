"""Shadow: per sub WVMI sweep -> frame span, window, records (slice-local + absolute). Writes WVMI_SHADOW_OUT (default: the temp dir, never the repo)."""
import json, os, atexit, tempfile
import engine_v2.multitf.sub_wvmi as sw
import engine_v2.zones.wvmi as wv
OUT = os.environ.get("WVMI_SHADOW_OUT") or os.path.join(tempfile.gettempdir(), "wvmi_shadow.json")
rows, main_rows = [], []
_orig = sw.compute_parent_driven_sub_wvmi
def wrapped(result, sub_path_id, parent_trigger):
    recs = _orig(result, sub_path_id, parent_trigger)
    m = result.meta; sb = int(m.get("slice_begin", 0))
    rows.append(dict(sub_id=m.get("sub_id"), lenses=sorted(m.get("lenses") or ()), sweep_path=sub_path_id,
                     trigger=parent_trigger.idx, slice_begin=sb, df_first=int(result.df.index.min()) + sb,
                     df_last=int(result.df.index.max()) + sb, start_idx=m.get("start_idx"), m15_end_idx=m.get("m15_end_idx"),
                     end_idx=m.get("end_idx"), n_cts_conf=sum(1 for e in result.events if e.type == "CTS_CONFIRMED"),
                     recs=[dict(cyc=r.bos_cycle_id, fp=None if r.fp_idx is None else r.fp_idx + sb,
                                lp=None if r.lp_idx is None else r.lp_idx + sb, locked=r.lp_locked) for r in recs]))
    return recs
sw.compute_parent_driven_sub_wvmi = wrapped
_orig_upd = wv.WVMITracker.update_temporary_lp
def upd(self, df, kl_zones):
    out = _orig_upd(self, df, kl_zones)
    if self._structure_path_id == "H1.main":
        main_rows.append(dict(df_last=int(df.index.max()), recs=[dict(sid=r.bos_structure_id, cyc=r.bos_cycle_id, fp=r.fp_idx, lp=r.lp_idx, locked=r.lp_locked) for r in self.get_records()]))
    return out
wv.WVMITracker.update_temporary_lp = upd
atexit.register(lambda: json.dump(dict(subs=rows, main=main_rows), open(OUT, "w"), indent=1, default=str))
