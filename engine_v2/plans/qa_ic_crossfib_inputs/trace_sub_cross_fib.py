"""Read-only trace: every cross-fib decision of ONE sub's DOWNSTREAM FibTracker (the drawn fib), entity-absolute.

    python <this> SUB_ID            (from the repo root; runs the replay pipeline in-process, no exports)

Wraps (no engine edit): render_sub_projection (context: sub id + slice_begin), FibTracker.__init__ (tag the
non-quiet tracker built inside that sub's projection), the four public handlers, `_on_cts_threshold_updated`,
`_cross_check` and fib_tracker's `resolve_cross_cycle_eligibility` (walk detail recomputed BEFORE the real call,
since the real call mutates the dead-cycle cache), `_log`.
"""
import sys
import time

sys.path.insert(0, ".")
TARGET = int(sys.argv[1])
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.zones.fib_tracker as ftm  # noqa: E402
from engine_v2.zones.fib_tracker import FibTracker  # noqa: E402
from engine_v2.structure import event_fields as ef  # noqa: E402

CTX = []
OFF = {"b": None}


def A(i):
    return None if i is None else int(i) + OFF["b"]


_real_rsp = edm.render_sub_projection


def _rsp(sub, *a, **k):
    CTX.append((int(sub.sub_id), int(sub.geometry[1])))
    try:
        return _real_rsp(sub, *a, **k)
    finally:
        CTX.pop()


edm.render_sub_projection = _rsp

_real_init = FibTracker.__init__


def _init(self, *a, **k):
    _real_init(self, *a, **k)
    c = CTX[-1] if CTX else None
    self._tr = (c is not None and c[0] == TARGET and not self._quiet)
    if self._tr:
        OFF["b"] = c[1]
        lim = self._limits.get(0)
        print(f"[TR] sub {TARGET} downstream tracker; slice_begin={c[1]}; limit P={lim.price} src={lim.source} "
              f"moves={lim.price_moves} ref_anchor={lim.ref_anchor_price}")


FibTracker.__init__ = _init

_real_log = FibTracker._log


def _log(self, msg):
    if getattr(self, "_tr", False):
        print("[TRlog local] " + msg)


FibTracker._log = _log


def _evline(ev):
    t = ev.type
    m = ev.meta
    s = f"{t} cyc={m.get('cycle_id')} idx={A(ev.idx)} price={ev.price}"
    if t in ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED"):
        try:
            s += f" anchor={A(ef.cts_anchor_idx(ev))}"
        except Exception:  # noqa: BLE001
            pass
    try:
        s += f" moment={A(ef.event_moment(ev))}"
    except Exception:  # noqa: BLE001
        pass
    if t == "CTS_ESTABLISHED" and m.get("pattern_anchor_idx") is not None:
        s += f" pattern_anchor={A(m.get('pattern_anchor_idx'))}"
    return s


def _state(self, cyc):
    out = []
    lc = self._get_latest_cross(0, cyc)
    if lc is not None:
        k, f = lc
        out.append(f"cross v{k[3]} x={f.meta.get('cross_start_cycle')} {A(f.bos_idx)}->{A(f.cts_idx)} "
                   f"active={f.active} locked={f.locked}")
    s = self._fibs.get((0, cyc))
    if s is not None:
        out.append(f"single {A(s.bos_idx)}->{A(s.cts_idx)} active={s.active} locked={s.locked} via={s.meta.get('via')}")
    return "; ".join(out) or "no fib"


for name in ("on_cts_established", "on_cts_updated", "on_cts_confirmed", "on_cts_threshold_updated"):
    real = getattr(FibTracker, name)

    def mk(real, name):
        def w(self, event, *a, **k):
            if not getattr(self, "_tr", False):
                return real(self, event, *a, **k)
            extra = ""
            if name == "on_cts_established":
                extra = f" bos_idx={A(a[1])} bos_price={a[2]}"
            print(f"[TR] >> {_evline(event)}{extra}")
            out = real(self, event, *a, **k)
            cyc = int(event.meta.get("cycle_id", 0))
            tgt = cyc + 1 if name == "on_cts_threshold_updated" else cyc
            print(f"[TR]    after: cycle {tgt}: {_state(self, tgt)} | phase={self._phase.get((0, tgt))} "
                  f"| dead={sorted(self._dead_cycles.get(0, set()))} | M={self.cross_fib_limit_cycle(0)} "
                  f"| perm={self._cross_permission.get((0, tgt))}")
            return out
        return w
    setattr(FibTracker, name, mk(real, name))

_real_thr = FibTracker._on_cts_threshold_updated


def _thr(self, event, df):
    if getattr(self, "_tr", False):
        src = int(event.meta.get("cycle_id", 0))
        tgt = src + 1
        ph = self._phase.get((0, tgt))
        allowed = self._cross_allowed(0, tgt) if ph == "pre_established" else "n/a"
        conf = self._cts_confirmed_idx.get((0, src))
        print(f"[TR]    threshold path: target {tgt} phase={ph} allowed={allowed} cts_{src}_confirmed={A(conf)}")
    return _real_thr(self, event, df)


FibTracker._on_cts_threshold_updated = _thr

CUR = {"t": None}
_real_cc = FibTracker._cross_check


def _cc(self, sid, target_cycle, df, sd, current_candle, anchor_idx, anchor_price, own_imb_start, *,
        own_window_end_idx):
    if getattr(self, "_tr", False):
        CUR["t"] = (self, df)
        print(f"[TR]    CROSS CHECK target={target_cycle} phase={self._phase.get((sid, target_cycle))} "
              f"own=[{A(own_imb_start)}, {A(own_window_end_idx)}] now={A(current_candle)} anchor={A(anchor_idx)} "
              f"@{anchor_price} allowed={self._cross_allowed(sid, target_cycle)} "
              f"bos_by_cycle={ {k[1]: (A(v[0]), v[1]) for k, v in self._bos_by_cycle.items()} } "
              f"cts_by_cycle={ {k[1]: (A(v[0]), v[1]) for k, v in self._cts_by_cycle.items()} }")
    try:
        return _real_cc(self, sid, target_cycle, df, sd, current_candle, anchor_idx, anchor_price, own_imb_start,
                        own_window_end_idx=own_window_end_idx)
    finally:
        if getattr(self, "_tr", False):
            print(f"[TR]    CROSS CHECK result: {_state(self, target_cycle)}")
        CUR["t"] = None


FibTracker._cross_check = _cc


def _fill(inst, df, horizon, thr=0.70):
    """(armed_idx, filled_idx) within (end_idx, horizon] — the is_filled state machine, positions."""
    h = df["h"].to_numpy(float)
    l_ = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)
    armed = filled = None
    for p in range(inst.end_idx + 1, min(horizon, len(df) - 1) + 1):
        if inst.direction == 1:
            if armed is None and l_[p] <= inst.gap_top - inst.gap_size * thr:
                armed = p
            if armed is not None and c[p] >= inst.gap_top:
                filled = p
                break
        else:
            if armed is None and h[p] >= inst.gap_bottom + inst.gap_size * thr:
                armed = p
            if armed is not None and c[p] <= inst.gap_bottom:
                filled = p
                break
    return armed, filled


def _insts(df, lo, hi, sd, horizon, at):
    rows = []
    for inst in df.attrs.get("imbalances", []):
        if inst.direction != sd:
            continue
        if not inst.overlaps(lo, hi):
            continue
        formed = inst.overlaps_formed_prefix(lo, hi, at) if at is not None else True
        armed, filled = _fill(inst, df, horizon)
        real = inst.is_filled(df, horizon, 0.70)
        assert real == (filled is not None), (inst, horizon)
        rows.append(f"c2 {A(inst.start_idx)}..{A(inst.end_idx)} gap [{inst.gap_bottom:.5f},{inst.gap_top:.5f}] "
                    f"formed_by_now={formed} armed={A(armed)} filled={A(filled)}")
    return rows


_real_res = ftm.resolve_cross_cycle_eligibility


def _res(**kw):
    if CUR["t"] is not None:
        self, _ = CUR["t"]
        df = kw["df"]
        sd = kw["sd"]
        hz = kw["fill_horizon_idx"]
        at = kw["evaluated_at"]
        lo = min(kw["own_imb_start"], kw["own_window_end_idx"])
        hi = max(kw["own_imb_start"], kw["own_window_end_idx"])
        print(f"[TR]      own window [{A(lo)}, {A(hi)}] (horizon {A(hz)}, moment {A(at)}), sd imbalances:")
        for r in _insts(df, lo, hi, sd, hz, at):
            print("[TR]        " + r)
        dead = set(kw["dead_cycles"])
        for k in range(kw["target_cycle"] - 1, -1, -1):
            if k in dead:
                print(f"[TR]      walk k={k}: ALREADY DEAD (cached)")
                break
            b = kw["bos_by_cycle"].get(k)
            c = kw["cts_by_cycle"].get(k)
            if b is None or c is None:
                print(f"[TR]      walk k={k}: missing bos/cts")
                break
            lo2, hi2 = min(b[0], c[0]), max(b[0], c[0])
            print(f"[TR]      walk k={k}: window [{A(lo2)}, {A(hi2)}] sd imbalances:")
            rows = _insts(df, lo2, hi2, sd, hz, at)
            for r in rows:
                print("[TR]        " + r)
            if not rows:
                print("[TR]        (none)")
    out = _real_res(**kw)
    if CUR["t"] is not None:
        print(f"[TR]      => own_has={out.own_has} earliest_x={out.earliest_x} crosses={out.crosses} "
              f"dead_after={sorted(kw['dead_cycles'])}")
    return out


ftm.resolve_cross_cycle_eligibility = _res

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
print(f"[trace] window {effective_start} -> {rr.CONFIG.end}: {len(df)} H1 candles")
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
print(f"[trace] done in {time.perf_counter() - t0:.1f} s")
