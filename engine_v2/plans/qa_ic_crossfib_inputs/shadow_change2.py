"""Read-only SHADOW of change 2 ALONE (2026-10-13; the user's write-up: the CTS reference = CTS_{T−1}, T = the fib's
target cycle, for a cross too) on HEAD — no engine edit; the run's outputs are today's. Every caller of
`FibTracker.ic_reference`, today (V0) vs change 2 (V2 = `ref_v2`, the code of `trial_p1_change2.patch`):

  [STAMP]  `stamp_ic_references` — the fib meta `ic_ref_*` (the fib CSV `meta`, the fib hover's IC ref line)
  [DRAWN]  `select_ic_records` — the drawn POI's pick (+ its IC CSV rows' `ic_ref_reason`)
  [MSREAD] MS's in-flight read (`poi_inner_for_fib`) — the POI inner of the cycle's proximity snapshot
  [MSPROX] MS's per-candle sd-proximity check (`_check_proximity_at_candle`) under the V0 vs the V2 snapshot — the
           FIRST candle per (MS run, cycle) where the outcome (hit / inner / zone kind) differs = a predicted CTS
           confirmation (moment or `trigger_inner` meta) change; none = MS's events cannot move (its only use of the
           inner)

    python <this> [OUT_LOG]      (repo root; ~1 min)
"""
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
import engine_v2.structure.market_structure as msm  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402
from engine_v2.patterns.cross_limit_watch import find_cross_limit_break  # noqa: E402
from engine_v2.zones.cross_cycle_fib import ICReference  # noqa: E402
from engine_v2.zones.fib_tracker import FibTracker  # noqa: E402

LINES = []
CTX = []
STATS = Counter()
SEEN = set()
FIRST_DIV = set()


def out(s):
    LINES.append(s)


def tm(df, i):
    return str(df.loc[i, "time"])[5:16] if i is not None and "time" in df.columns and i in df.index else "?"


_real_rsp = edm.render_sub_projection


def _rsp(sub, *a, **k):
    CTX.append((int(sub.sub_id), int(sub.geometry[1])))
    try:
        return _real_rsp(sub, *a, **k)
    finally:
        CTX.pop()


edm.render_sub_projection = _rsp


def ref_v2(self, fib_state, df, *, evaluated_at):
    """`FibTracker.ic_reference` with change 2 (= `trial_p1_change2.patch`'s body)."""
    sid = int(fib_state.structure_id)
    target = int(fib_state.cycle_id)
    sd = int(fib_state.struct_direction)
    refs = []
    if target >= 1:
        cts = self._cts_by_cycle.get((sid, target - 1))
        assert cts is not None, (sid, target)
        anchor = self._est_pattern_anchor.get((sid, target))
        known = self._bos_moment_by_cycle.get((sid, target))
        assert (anchor is None) == (known is None), (sid, target, anchor, known)
        if anchor is None:
            STATS["v2_target_not_established"] += 1
        brk = (anchor, known) if (anchor is not None and (evaluated_at is None or known <= int(evaluated_at))) \
            else None
        refs.append((False, float(cts[1]), "prev_cycle_cts", brk))
    lim = self._structure_reference_limit(sid, target, evaluated_at)
    if lim is not None:
        if lim.source == "reversal_bos_frozen":
            brk = (int(lim.break_pattern_anchor_idx), int(lim.break_confirmed_idx))
            refs.append((True, float(lim.ref_anchor_price), "prev_structure_bos", brk))
        else:
            start = int(self._bos_by_cycle[(sid, 0)][0])
            brk = find_cross_limit_break(df, float(lim.price_at(evaluated_at)), sd, start, known_at=evaluated_at)
            refs.append((True, float(lim.ref_anchor_price), "cross_limit_anchor", brk))
    if not refs:
        return ICReference.NONE
    if sd == 1:
        chosen = min(refs, key=lambda r: (r[1], 0 if r[0] else 1))
    else:
        chosen = min(refs, key=lambda r: (-r[1], 0 if r[0] else 1))
    _s, price, source, brk = chosen
    if brk is None:
        return ICReference(price, source)
    return ICReference(price, source, int(brk[0]), int(brk[1]))


def rstr(df, r, b=0):
    if r.price is None:
        return "none"
    k = r.break_pattern_anchor_idx
    return (f"{r.price:.5f} {r.source} brk {None if k is None else int(k) + b}"
            f"{'' if k is None else ' (' + tm(df, k) + ')'}")


def kind(fs):
    m = fs.meta
    return (f"cross {m.get('cross_start_cycle')}->{fs.cycle_id} v{m.get('version')}" if m.get("cross_cycle")
            else f"single {fs.cycle_id}")


def ent():
    return ("H1" if not CTX else f"sub {CTX[-1][0]}"), (0 if not CTX else CTX[-1][1])


# --- [STAMP]
_real_stamp = FibTracker.stamp_ic_references


def _stamp(self, df, lifecycle_cap=None):
    e, b = ent()
    for key, fib in list(self._fibs.items()):
        r0 = self.ic_reference(fib, df, evaluated_at=lifecycle_cap)
        r2 = ref_v2(self, fib, df, evaluated_at=lifecycle_cap)
        STATS["stamps"] += 1
        k = ("STAMP", e, int(fib.structure_id), kind(fib), int(fib.bos_idx) + b)
        if r0 != r2 and k not in SEEN:
            SEEN.add(k)
            STATS["stamps_differ"] += 1
            out(f"[STAMP] {e} sid{fib.structure_id} {kind(fib)} bos {int(fib.bos_idx) + b} cts {int(fib.cts_idx) + b}: "
                f"{rstr(df, r0, b)}  ->  {rstr(df, r2, b)}")
    return _real_stamp(self, df, lifecycle_cap)


FibTracker.stamp_ic_references = _stamp

# --- [DRAWN]
_real_sir = orch.select_ic_records


def _sir(df, events, fib_tracker, config, *, lifecycle_cap=None):
    recs = _real_sir(df, events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    e, b = ent()
    for rec in recs:
        fs = rec.fib_state
        r0 = fib_tracker.ic_reference(fs, df, evaluated_at=lifecycle_cap)
        r2 = ref_v2(fib_tracker, fs, df, evaluated_at=lifecycle_cap)
        if r0 == r2:
            continue
        s2 = pz.select_ic(df, fs, config, ic_reference=r2)
        p0 = None if rec.selection.pick is None else int(rec.selection.pick.ic_idx) + b
        p2 = None if s2.pick is None else int(s2.pick.ic_idx) + b
        k = ("DRAWN", e, int(fs.structure_id), kind(fs), int(fs.bos_idx) + b)
        if k in SEEN:
            continue
        SEEN.add(k)
        STATS["drawn_ref_differs"] += 1
        STATS["drawn_pick_moves"] += p0 != p2
        n0 = sum(c.ic_ref_reason is None for c in rec.selection.checks)
        n2 = sum(c.ic_ref_reason is None for c in s2.checks)
        out(f"[DRAWN] {e} sid{fs.structure_id} {kind(fs)} bos {int(fs.bos_idx) + b} cts {int(fs.cts_idx) + b}: "
            f"pick {p0} ({tm(df, None if p0 is None else p0 - b)}) -> {p2} ({tm(df, None if p2 is None else p2 - b)}) "
            f"{'MOVED' if p0 != p2 else 'same'}; checks passing the reference {n0} -> {n2} of {len(s2.checks)}")
    return recs


orch.select_ic_records = _sir


# --- [MSREAD] + the V2 snapshot (an extra attribute on MS's state: deep-copied with it at every snapshot)
def _inner(df, fs, ref, ft):
    sel = pz.select_ic(df, fs, pz.POIConfig(fill_threshold=ft), ic_reference=ref)
    if sel.pick is None:
        return None
    i = int(sel.pick.ic_idx)
    return float(df.loc[i, "h"]) if int(fs.struct_direction) == 1 else float(df.loc[i, "l"])


_real_refresh = msm.MarketStructure._refresh_poi_inners_for_cycle


def _refresh(self, moment):
    _real_refresh(self, moment)
    st = self.state
    if self._fib_tracker is None:
        st._v2_inners = []
        return
    fib = self._fib_tracker.representative_fib(int(st.structure_id), int(st.cts_cycle_id))
    if fib is None:
        st._v2_inners = []
        return
    df = self._resolver_df()
    ft = float(self._fib_tracker.config.fill_threshold)
    r0 = self._fib_tracker.ic_reference(fib, df, evaluated_at=int(moment))
    r2 = ref_v2(self._fib_tracker, fib, df, evaluated_at=int(moment))
    i2 = _inner(df, fib, r2, ft) if r2 != r0 else (st.poi_inners_for_cycle[0] if st.poi_inners_for_cycle else None)
    st._v2_inners = [] if i2 is None else [float(i2)]
    tf = df.attrs.get("timeframe")
    STATS[f"msreads_{tf}"] += 1
    if r0 != r2:
        STATS[f"msreads_ref_differs_{tf}"] += 1
        i0 = st.poi_inners_for_cycle[0] if st.poi_inners_for_cycle else None
        if i0 != i2:
            STATS[f"msreads_inner_differs_{tf}"] += 1
            out(f"[MSREAD] {tf} sid{fib.structure_id} {kind(fib)} bos {fib.bos_idx} ({tm(df, fib.bos_idx)}) "
                f"at {moment} ({tm(df, moment)}): ref {rstr(df, r0)} -> {rstr(df, r2)}; inner {i0} -> {i2}")


msm.MarketStructure._refresh_poi_inners_for_cycle = _refresh

# --- [MSPROX]
_real_check = msm.MarketStructure._check_proximity_at_candle


def _check(self, i):
    h0 = _real_check(self, i)
    st = self.state
    if st.bos_inner_for_cycle is None:
        return h0
    v2 = getattr(st, "_v2_inners", st.poi_inners_for_cycle)
    h2 = msm._check_sd_proximity_at_candle(self.df, candle_idx=i, struct_direction=self.struct_direction,
                                           bos_inner=st.bos_inner_for_cycle, threshold=self._proximity_threshold,
                                           poi_inners=list(v2) if v2 else None)
    STATS["prox_checks"] += 1
    if h0 != h2:
        k = (id(self), int(st.structure_id), int(st.cts_cycle_id))
        STATS["prox_checks_differ"] += 1
        if k not in FIRST_DIV:
            FIRST_DIV.add(k)
            out(f"[MSPROX] {self.df.attrs.get('timeframe')} sid{st.structure_id} cycle {st.cts_cycle_id} candle {i} "
                f"({tm(self.df, i)}): V0 {h0} vs V2 {h2}")
    return h0


msm.MarketStructure._check_proximity_at_candle = _check

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
out(f"[STATS] {dict(sorted(STATS.items()))}")
out(f"[done] {time.perf_counter() - t0:.1f} s")
text = "\n".join(LINES)
if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
print(text)
