"""Measure the proposed new TB1 (an unfilled sd imbalance within 2 candles after the IC) on every tied
(rank, shape) group select_ip_ic sees in a full pipeline run — the drawn POIs AND MS's in-flight reads.
Read-only: wraps poi_zones.select_ip_ic, exports nothing."""
import inspect
import sys

import engine_v2.zones.poi_zones as pz
from engine_v2.patterns.imbalance import iter_unfilled_imbalances

_orig = pz.select_ip_ic
seen = {}


def _caller():
    for fr in inspect.stack()[2:8]:
        if fr.function in ("select_ip_records", "poi_inner_for_fib"):
            return fr.function
    return "?"


def _wrapped(df, fib_state, config, **kw):
    sel = _orig(df, fib_state, config, **kw)
    pats = [c.ip for c in sel.candidates
            if pz.is_ic_candidate(c.touches_zone, c.ic_ref_reason) and c.ip.rank is not None]
    if len(pats) < 2:
        return sel
    best = min(r.order_key for r in pats)
    tied = sorted((r for r in pats if r.order_key == best), key=lambda r: r.ic_idx)
    if len(tied) < 2:
        return sel
    sd = int(fib_state.struct_direction)
    cts = int(fib_state.cts_idx)
    t0 = df.index[0]
    rows = []
    for r in tied:
        i = r.ic_idx
        insts = [inst for inst in df.attrs.get("imbalances", [])
                 if inst.direction == sd and i < inst.start_idx <= i + 3]
        desc = []
        for inst in insts:
            filled = inst.is_filled(df, cts, config.fill_threshold)
            desc.append(f"c2 {inst.start_idx}..{inst.end_idx} (+{inst.start_idx - i}) {'FILLED' if filled else 'unfilled'}")
        def unf(lo, hi):
            return any(iter_unfilled_imbalances(df, lo, hi, cts, config.fill_threshold, direction=sd,
                                                evaluated_at=None))
        rows.append(dict(
            ic=i, ctype=str(df.loc[i, "candle_type"]),
            c1=any((not inst.is_filled(df, cts, config.fill_threshold)) and inst.start_idx == i + 1 for inst in insts),
            c2=unf(i + 1, min(i + 2, cts)),
            c2_noclamp=unf(i + 1, i + 2),
            c1w2=unf(i + 1, min(i + 3, cts)),
            near=desc))
    key = (df.attrs.get("timeframe"), str(df.loc[t0, "time"]) if "time" in df.columns else t0,
           fib_state.structure_id, getattr(fib_state, "cycle_id", None), int(fib_state.bos_idx), cts, _caller())
    if key in seen:
        return sel
    seen[key] = 1
    pick = sel.pick.ic_idx if sel.pick else None
    print(f"\n== {key[0]} frame0={key[1]} sid={key[2]} cyc={key[3]} bos={key[4]} cts={cts} sd={sd} via={key[6]}"
          f" rank={tied[0].rank}.{getattr(tied[0], 'shape', '')} pick_now={pick} tb={sel.tiebreak}")
    for row in rows:
        print(f"   IC {row['ic']} {row['ctype']:6s}  TB1[c1-only]={row['c1']!s:5s} TB1[c2<=IC+2]={row['c2']!s:5s}"
              f" (no CTS clamp {row['c2_noclamp']!s:5s}) TB1[c1<=IC+2]={row['c1w2']!s:5s}  near: {row['near']}")
    sys.stdout.flush()
    return sel


pz.select_ip_ic = _wrapped

from engine_v2.run_replay import main  # noqa: E402

if __name__ == "__main__":
    import engine_v2.run_replay as rr
    # no chart / CSV export: stop after the pipeline
    _orig_run = rr.run_pipeline

    class _Stop(Exception):
        pass

    def _run_and_stop(*a, **k):
        _orig_run(*a, **k)
        raise _Stop()

    rr.run_pipeline = _run_and_stop
    rr.export_csv = lambda *a, **k: None
    try:
        main()
    except _Stop:
        print("\n[measure] pipeline done; groups:", len(seen))
