"""Dump the candles + sd imbalances around the tied ICs of each tied drawn record (read-only, no exports)."""
import engine_v2.zones.poi_zones as pz
_orig = pz.select_ip_records
SHIFT = {"2025-11-21 03:00:00+00:00": ("conf sub0", 404), "2025-12-11 02:45:00+00:00": ("conf sub1", 1747),
         "2026-01-15 08:15:00+00:00": ("sub7", 3977)}
def _first_fill(df, inst, thr, last):
    for k in range(inst.end_idx + 1, last + 1):
        if inst.is_filled(df, k, thr):
            return k
    return None
def _wrapped(*a, **k):
    recs = _orig(*a, **k)
    df = a[0] if a else k["df"]
    f0 = str(df["time"].iloc[0])
    if f0 not in SHIFT:
        return recs
    name, sh = SHIFT[f0]
    for rec in recs:
        sel = rec.selection
        if sel is None or sel.tiebreak is None:
            continue
        fs = rec.fib_state
        sd = int(fs.struct_direction)
        pats = [c.ip for c in sel.candidates if pz.is_ic_candidate(c.touches_zone, c.ic_ref_reason) and c.ip.rank is not None]
        best = min(r.order_key for r in pats)
        tied = sorted(r.ic_idx for r in pats if r.order_key == best)
        print(f"\n### {name} cyc {fs.cycle_id} sd {sd} bos {fs.bos_idx+sh} cts {fs.cts_idx+sh} tied {[t+sh for t in tied]} pick {sel.pick.ic_idx+sh} tb {sel.tiebreak}")
        for t in tied:
            print(f"  -- IC {t+sh}")
            for i in range(t - 1, t + 5):
                r = df.loc[i]
                print(f"    {i+sh:5d} {r['time']}  o {r['o']:.5f} h {r['h']:.5f} l {r['l']:.5f} c {r['c']:.5f} dir {int(r['direction']):+d} vol_dir {int(r['vol_dir']):+d} {r['candle_type']:7s} is_imb {int(r['is_imbalance'])}")
        lo, hi = tied[0], tied[-1] + 4
        for inst in df.attrs["imbalances"]:
            if inst.direction == sd and lo < inst.start_idx <= fs.cts_idx and (inst.start_idx <= hi or not inst.is_filled(df, fs.cts_idx, 0.70)):
                ff = _first_fill(df, inst, 0.70, fs.cts_idx)
                print(f"  imb c2 {inst.start_idx+sh}..{inst.end_idx+sh} gap {inst.gap_bottom:.5f}-{inst.gap_top:.5f}  filled by CTS: {ff is not None}"
                      + (f" (stroke 2 at {ff+sh}, {df.loc[ff,'time']})" if ff is not None else ""))
    return recs
pz.select_ip_records = _wrapped
import engine_v2.pipeline.orchestrator as _orc
_orc.select_ip_records = _wrapped
import engine_v2.run_replay as rr
class _Stop(Exception): pass
_o = rr.run_pipeline
def _r(*a, **k):
    _o(*a, **k); raise _Stop()
rr.run_pipeline = _r
rr.export_csv = lambda *a, **k: None
try:
    rr.main()
except _Stop:
    print("[dump] done")
