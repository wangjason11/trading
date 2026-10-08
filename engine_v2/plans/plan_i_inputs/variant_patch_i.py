"""PLAN_I session 2 — the minimal BEHAVIOUR patches of the variant replays (PLAN_H §8.4's pattern).

    python engine_v2/plans/plan_i_inputs/variant_patch_i.py TREE V2|V3|V3c|V4|V5

TREE = a `git archive` of HEAD e63c3ea (`TREE/engine_v2/...`). Each edit replaces an exact text that must occur
ONCE (else AssertionError — the tree is not HEAD). Cumulative:
  V2  — FORMAT ONLY: `cts_frozen` on CTS_ESTABLISHED (MS, c-1's `cts_threshold` before the reset; None on cycle 0)
        + the computed limit (`multitf/cross_fib_limit.py` = `cross_fib_limit_proto.py`) stamped on every fib
        record's meta at finalize (`cross_fib_limit_price/_source/_cycle/_from_sub_id`, main `started_by`); NOT
        applied. Logs: every CTS_ESTABLISHED's `cts_frozen` vs the breakout threshold its pattern was detected
        against; every sub's P; the cross_failed-single stale case in `_m15_cross_check`.
  V3  — V2 + the limit APPLIED to subs (drawn fib only; MS unchanged): a cycle T >= 1 not allowed (no P, or a
        cycle k < T whose latest CTS anchor is strictly beyond P) is a SINGLE with the sub's cycle-0 semantics
        (activate at EST if unfilled, else late on a CTS_UPDATED; every update re-checks); no pre-established cross.
  V3c — V2 + the limit applied with the CROSS_FAILED single instead (the walk forced empty -> today's
        `_activate_or_update_single_m15` fallback, which skips the single when the own window is filled).
  V4  — V3 + main onto the sub path: H1 FibTracker in `cross_cycle` mode with main's limits (sid 0 none, sid N
        the applied reversal's `bos_frozen`) — Q4 / Q5 / Q6 / I2 / I5 / Q8 / the P source together.
  V5  — V4 + MS reads the same fib decision (an embedded FibTracker fed MS's own events; added later).
"""
import shutil
import sys
from pathlib import Path

TREE, VARIANT = Path(sys.argv[1]), sys.argv[2]
ORDER = {"V2": ["stamps"], "V3": ["stamps", "apply_a"], "V3c": ["stamps", "apply_c"],
         "V4": ["stamps", "apply_a", "main"], "V5": ["stamps", "apply_a", "main", "ms"],
         # TEST-CENSUS trees (suite only, never replayed): the retired paths RAISE.
         "V3x": ["stamps", "apply_a", "strict_limits"],
         "V4x": ["stamps", "apply_a", "main", "strict_limits", "no_h1"],
         "V5x": ["stamps", "apply_a", "main", "ms", "strict_limits", "no_h1", "no_old_resolver"]}
assert VARIANT in ORDER, VARIANT
E = TREE / "engine_v2"


class F:
    def __init__(self, rel):
        self.p = E / rel
        self.src = self.p.read_text(encoding="utf-8")

    def sub(self, old, new):
        n = self.src.count(old)
        assert n == 1, (self.p.name, n, old[:160])
        self.src = self.src.replace(old, new)

    def save(self):
        self.p.write_text(self.src, encoding="utf-8")


def g_stamps():
    shutil.copy(Path(__file__).resolve().parent / "cross_fib_limit_proto.py", E / "multitf" / "cross_fib_limit.py")

    ms = F("structure/market_structure.py")
    ms.sub('''        if winner is not None:
            ev, apply_idx, kind = winner  # kind in {"breakout","pullback", "reversal"}
''', '''        if winner is not None:
            ev, apply_idx, kind = winner  # kind in {"breakout","pullback", "reversal"}
            self._pi_bth = breakout_th if kind == "breakout" else None
''')
    ms.sub('''                        "pattern_name": str(ev.name),
                        "confirmed_at": int(apply_idx),  # apply candle that established CTS
                    },
                )
''', '''                        "pattern_name": str(ev.name),
                        "confirmed_at": int(apply_idx),  # apply candle that established CTS
                        "cts_frozen": (float(st.cts_threshold) if st.cts_threshold is not None else None),
                    },
                )
                from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG
                _PI_LOG.append({"kind": "cts_frozen", "tf": self.timeframe, "ms_start": int(self.start_idx),
                                "ms_end": self.end_idx, "sid": int(st.structure_id), "cycle": int(st.cts_cycle_id),
                                "apply": int(apply_idx), "cts_frozen": self.events[-1].meta["cts_frozen"],
                                "breakout_th": getattr(self, "_pi_bth", None),
                                "scan_mode": bool(self.enforce_cts0_new_extreme)})
''')
    ms.save()

    ft = F("zones/fib_tracker.py")
    ft.sub('''    def __init__(self, config: Optional[FibTrackerConfig] = None, fib_mode: str = "h1"):''',
           '''    def __init__(self, config: Optional[FibTrackerConfig] = None, fib_mode: str = "h1", cross_fib_limits=None):''')
    ft.sub('''        self._evaluated_at: Optional[int] = None
        self._in_event = False
''', '''        self._evaluated_at: Optional[int] = None
        self._in_event = False
        # PLAN_I variant: the cross-fib limit per sid {sid: {"price", "source", "started_by", "from_sub"}} and
        # every cycle's LATEST CTS anchor (idx, price, sd) — EST / UPD / CONF.
        self._limits = dict(cross_fib_limits or {})
        self._latest_anchor: Dict[tuple, tuple] = {}
''')
    ft.sub('''    def _finalize_lifecycle_fields(
''', '''    def _pi_limit_cycle(self, sid: int):
        from engine_v2.multitf.cross_fib_limit import beyond
        lim = self._limits.get(sid)
        if not lim or lim["price"] is None:
            return None
        for c in sorted(c for (s, c) in self._latest_anchor if s == sid):
            _i, p, sd = self._latest_anchor[(sid, c)]
            if beyond(p, lim["price"], sd):
                return c
        return None

    def _pi_allowed(self, sid: int, target_cycle: int) -> bool:
        from engine_v2.multitf.cross_fib_limit import beyond
        lim = self._limits.get(sid)
        if lim is None:
            return True
        if lim["price"] is None:
            return False
        for (s, c), (_i, p, sd) in self._latest_anchor.items():
            if s == sid and c < target_cycle and beyond(p, lim["price"], sd):
                return False
        return True

    def _pi_limit_meta(self, sid: int) -> dict:
        lim = self._limits.get(sid)
        if lim is None:
            return {}
        fs = lim.get("from_sub")
        out = {"cross_fib_limit_price": lim["price"], "cross_fib_limit_source": lim["source"],
               "cross_fib_limit_cycle": self._pi_limit_cycle(sid),
               "cross_fib_limit_from_sub_id": None if fs is None else int(fs.sub_id)}
        if lim.get("started_by") is not None:
            out["started_by"] = lim["started_by"]
        return out

    def _finalize_lifecycle_fields(
''')
    ft.sub('''            for key in keys:
                fib = self._fibs[key]
                self._fibs[key] = replace(
                    fib,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    end_reason=end_reason,
                    status=status,
                )
''', '''            for key in keys:
                fib = self._fibs[key]
                self._fibs[key] = replace(
                    fib,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    end_reason=end_reason,
                    status=status,
                    meta={**fib.meta, **self._pi_limit_meta(int(sid))},
                )
''')
    ft.sub('''        # Check for unfilled imbalance between BOS and CTS, counting only gaps
''', '''        self._latest_anchor[(sid, cycle_id)] = (int(cts_idx), float(cts_price), int(sd))
        # Check for unfilled imbalance between BOS and CTS, counting only gaps
''')
    ft.sub('''        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_updated(sid, cycle_id, sd, cts_idx, cts_price, df)
''', '''        self._latest_anchor[(sid, cycle_id)] = (int(cts_idx), float(cts_price), int(sd))
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_updated(sid, cycle_id, sd, cts_idx, cts_price, df)
''')
    ft.sub('''        self._cts_by_cycle[(sid, cycle_id)] = (cts_anchor_idx, cts_price_val)
''', '''        self._cts_by_cycle[(sid, cycle_id)] = (cts_anchor_idx, cts_price_val)
        self._latest_anchor[(sid, cycle_id)] = (
            int(cts_anchor_idx), float(cts_price_val), int(event.meta.get("struct_direction", 0)))
''')
    ft.sub('''        if not elig.own_has:
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")
''', '''        if not elig.own_has:
            _pi_single = self._fibs.get((sid, target_cycle))
            if _pi_single is not None:
                from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG
                _PI_LOG.append({"kind": "stale_single", "sid": int(sid), "cycle": int(target_cycle),
                                "moment": int(current_candle), "phase": phase, "single_active": bool(_pi_single.active),
                                "single_cts_idx": int(_pi_single.cts_idx), "anchor_idx": int(anchor_idx),
                                "via": _pi_single.meta.get("via"), "locked": bool(_pi_single.locked)})
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")
''')
    ft.save()

    orc = F("pipeline/orchestrator.py")
    orc.sub('''    lifecycle_cap: Optional[int] = None,
    cap_reason: str = "lifecycle_end",
) -> Dict[str, Any]:
    """Run downstream pipeline (KL zones -> wave candles -> Fib -> POI -> WVMI).''',
            '''    lifecycle_cap: Optional[int] = None,
    cap_reason: str = "lifecycle_end",
    cross_fib_limits: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run downstream pipeline (KL zones -> wave candles -> Fib -> POI -> WVMI).''')
    orc.sub('''        fib_mode=fib_mode,
    )

    bos_anchor_by_cycle = {}''', '''        fib_mode=fib_mode,
        cross_fib_limits=cross_fib_limits,
    )

    bos_anchor_by_cycle = {}''')
    orc.sub('''    downstream = _run_downstream_pipeline(
        s_res.df,
        s_res.events,
        s_res.struct_direction,
    )''', '''    from engine_v2.multitf.cross_fib_limit import main_limits as _pi_main_limits
    downstream = _run_downstream_pipeline(
        s_res.df,
        s_res.events,
        s_res.struct_direction,
        cross_fib_limits=_pi_main_limits(s_res.events),  # PI_MAIN_MODE
    )''')
    orc.sub('''    sweep = run_lifecycle_sweep(
        sweep_triggers, pool=sub_pool, tables=tables,''', '''    import engine_v2.multitf.lifecycle_sweep as _pi_ls
    from engine_v2.multitf.cross_fib_limit import sub_limit as _pi_sub_limit
    _pi_ls.CROSS_FIB_LIMIT_HOOK = lambda sw, t, s, rs: _pi_sub_limit(sw, t, s, rs, sorted_events)
    sweep = run_lifecycle_sweep(
        sweep_triggers, pool=sub_pool, tables=tables,''')
    orc.save()

    sw = F("multitf/lifecycle_sweep.py")
    sw.sub('''_PHASE_FIRE, _PHASE_RECORD_START, _PHASE_SUB_START, _PHASE_RECORD_END, _PHASE_SUB_END = range(5)
''', '''_PHASE_FIRE, _PHASE_RECORD_START, _PHASE_SUB_START, _PHASE_RECORD_END, _PHASE_SUB_END = range(5)
CROSS_FIB_LIMIT_HOOK = None   # PLAN_I variant: (sweep, trigger, record start, reversing_sub) -> limit dict
''')
    sw.sub('''        key = StructureKey(self.parent_path, self.sub_tf, int(t.direction), int(res.starting_idx))
        built = self.build_geometry(key, res.bos0_inner)
''', '''        key = StructureKey(self.parent_path, self.sub_tf, int(t.direction), int(res.starting_idx))
        _pi_limit = None
        if self.pool.get(key) is None and CROSS_FIB_LIMIT_HOOK is not None:
            _pi_start = max(int(res.finalize_idx), int(t.trigger_idx), int(self.tables.floor(S, C)))
            _pi_limit = CROSS_FIB_LIMIT_HOOK(self, t, _pi_start, reversing_sub)
            import engine_v2.multitf.entity_df_mutation as _pi_edm
            _pi_edm._PI_PENDING_LIMIT = _pi_limit
        built = self.build_geometry(key, res.bos0_inner)
''')
    sw.sub('''        sub, created = built
        if created and sub.natural_reversal_idx is not None:''', '''        sub, created = built
        if created:
            sub.cross_fib_limit = _pi_limit
        if created and sub.natural_reversal_idx is not None:''')
    sw.save()

    edm = F("multitf/entity_df_mutation.py")
    edm.sub('''        log_prefix=f"M15_sub{sub.sub_id}_{first.trigger_type}",
        structure_path_id=first_path,
    )''', '''        log_prefix=f"M15_sub{sub.sub_id}_{first.trigger_type}",
        structure_path_id=first_path,
        cross_fib_limits={0: getattr(sub, "cross_fib_limit", None)},
    )''')
    edm.save()

    psb = F("multitf/pooled_structure_build.py")
    psb.sub('''    log_prefix: str = "",
) -> Dict[str, Any]:
    """Derive the render snapshots for ONE lifecycle window over a natural-end run.''', '''    log_prefix: str = "",
    cross_fib_limits=None,
) -> Dict[str, Any]:
    """Derive the render snapshots for ONE lifecycle window over a natural-end run.''')
    psb.sub('''        lifecycle_cap=cap,
        cap_reason=cap_reason,
    )
    downstream["events"] = events''', '''        lifecycle_cap=cap,
        cap_reason=cap_reason,
        cross_fib_limits={k: v for k, v in (cross_fib_limits or {}).items() if v is not None},
    )
    downstream["events"] = events''')
    psb.save()


def g_apply_a():
    ft = F("zones/fib_tracker.py")
    ft.sub('''        if cycle_id == 0:
            # Cycle 0: no cross possible. Simple single-fib activation.''',
           '''        if cycle_id == 0 or not self._pi_allowed(sid, cycle_id):
            # Cycle 0: no cross possible. Simple single-fib activation.''')
    ft.sub('''            return self._activate_fib(
                sid=sid,
                cycle_id=0,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_established_idx, "fib_mode": "cross_cycle"},
            )''', '''            return self._activate_fib(
                sid=sid,
                cycle_id=cycle_id,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_established_idx, "fib_mode": "cross_cycle"},
            )''')
    ft.sub('''        if cycle_id == 0:
            # Cycle 0: no cross logic. Single-fib update.
            key = (sid, 0)''', '''        if cycle_id == 0 or not self._pi_allowed(sid, cycle_id):
            # Cycle 0: no cross logic. Single-fib update.
            key = (sid, cycle_id)''')
    ft.sub('''            bos = self._bos_by_cycle.get((sid, 0))
            if bos is None:
                return None
            bos_idx, bos_price = bos
            start_idx = min(bos_idx, cts_idx)''', '''            bos = self._bos_by_cycle.get((sid, cycle_id))
            if bos is None:
                return None
            bos_idx, bos_price = bos
            start_idx = min(bos_idx, cts_idx)''')
    ft.sub('''            return self._activate_fib(
                sid=sid,
                cycle_id=0,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={
                    "activated_at": self._moment(),
                    "fib_mode": "cross_cycle",
                    "activated_on": "update",
                },
            )''', '''            return self._activate_fib(
                sid=sid,
                cycle_id=cycle_id,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={
                    "activated_at": self._moment(),
                    "fib_mode": "cross_cycle",
                    "activated_on": "update",
                },
            )''')
    _no_pre_established(ft)
    ft.save()


def _no_pre_established(ft):
    ft.sub('''        if phase != "pre_established":
            # Not in pre-established for the expected target: skip.
            return
''', '''        if phase != "pre_established":
            # Not in pre-established for the expected target: skip.
            return
        if not self._pi_allowed(sid, target_cycle):
            return
''')


def g_apply_c():
    ft = F("zones/fib_tracker.py")
    ft.sub('''            bos_by_cycle={
                k: self._bos_by_cycle[(sid, k)]
                for k in range(target_cycle)
                if (sid, k) in self._bos_by_cycle
            },''', '''            bos_by_cycle=({
                k: self._bos_by_cycle[(sid, k)]
                for k in range(target_cycle)
                if (sid, k) in self._bos_by_cycle
            } if self._pi_allowed(sid, target_cycle) else {}),''')
    ft.save()


def g_main():
    orc = F("pipeline/orchestrator.py")
    orc.sub('''        cross_fib_limits=_pi_main_limits(s_res.events),  # PI_MAIN_MODE''',
            '''        cross_fib_limits=_pi_main_limits(s_res.events),  # PI_MAIN_MODE
        fib_mode="cross_cycle",''')
    orc.save()


def g_ms():
    """V5: MS's in-flight POI read = the representative fib of an EMBEDDED FibTracker (`cross_cycle` mode + the
    structure's limit) fed MS's own CTS / BOS events as they are emitted (Q7). Main sid N gets P from sid N-1's
    applied reversal; a sub's geometry gets the pool's P (computed at the fire, before the build); a probe gets no P.
    Every tracker (MS shadow + downstream) logs its handled sequence + the representative fib after each event
    (`seq`), so the agreement can be checked offline."""
    ft = F("zones/fib_tracker.py")
    ft.sub('''    def _pi_limit_cycle(self, sid: int):''', '''    def _pi_rep(self, sid: int, cycle: int):
        latest = self._get_latest_cross(sid, cycle)
        if latest is not None and latest[1].active:
            f = latest[1]
            return ["cross", int(latest[0][3]), int(f.bos_idx), int(f.cts_idx), bool(f.active)]
        f = self._fibs.get((sid, cycle))
        if f is not None and f.active:
            return ["single", None, int(f.bos_idx), int(f.cts_idx), bool(f.active)]
        return None

    def _pi_seq(self, event):
        label = getattr(self, "_pi_label", None)
        if label is None:
            return
        from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG
        from engine_v2.structure import event_fields as _ef
        sid = int(event.meta.get("structure_id", 0))
        cyc = int(event.meta.get("cycle_id", 0))
        _PI_LOG.append({"kind": "seq", "label": label, "type": event.type, "sid": sid, "cycle": cyc,
                        "moment": int(_ef.event_moment(event)) if event.type != "CTS_CONFIRMED" else int(event.idx),
                        "rep": self._pi_rep(sid, cyc),
                        "rep_next": self._pi_rep(sid, cyc + 1) if event.type == "CTS_THRESHOLD_UPDATED" else None})

    def _pi_limit_cycle(self, sid: int):''')
    ft.sub('''        with self._evaluating(event):
            return self._on_cts_established(
                event, df, bos_idx, bos_price,
                reversal_confirmed_idx, prev_bos_outer, prev_sd,
            )''', '''        with self._evaluating(event):
            _r = self._on_cts_established(
                event, df, bos_idx, bos_price,
                reversal_confirmed_idx, prev_bos_outer, prev_sd,
            )
        self._pi_seq(event)
        return _r''')
    ft.sub('''        with self._evaluating(event):
            return self._on_cts_updated(event, df, reversal_confirmed_idx)''', '''        with self._evaluating(event):
            _r = self._on_cts_updated(event, df, reversal_confirmed_idx)
        self._pi_seq(event)
        return _r''')
    ft.sub('''        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_confirmed(sid, cycle_id, event)
''', '''        if self.fib_mode == "cross_cycle":
            _r = self._handle_cross_cycle_cts_confirmed(sid, cycle_id, event)
            self._pi_seq(event)
            return _r
''')
    ft.sub('''        with self._evaluating(event):
            self._on_cts_threshold_updated(event, df)''', '''        with self._evaluating(event):
            self._on_cts_threshold_updated(event, df)
        self._pi_seq(event)''')
    ft.save()

    orc = F("pipeline/orchestrator.py")
    orc.sub('''        cross_fib_limits=cross_fib_limits,
    )

    bos_anchor_by_cycle = {}''', '''        cross_fib_limits=cross_fib_limits,
    )
    fib_tracker._pi_label = f"ds|{structure_path_id}|{log_prefix}"

    bos_anchor_by_cycle = {}''')
    orc.save()

    ms = F("structure/market_structure.py")
    ms.sub('''        stop_after_cts_established: Optional[int] = None,
    ):''', '''        stop_after_cts_established: Optional[int] = None,
        fib_shadow=None,
    ):''')
    ms.sub('''        self._fill_threshold = float(fill_threshold)
''', '''        self._fill_threshold = float(fill_threshold)
        # PLAN_I V5: the embedded FibTracker (Q7) — fed every CTS / BOS event as MS emits it.
        self._fib_shadow = fib_shadow
''')
    ms.sub('''                # New cycle => reset CTS confirmation guard & threshold
                st.confirmed_cts_anchor_idx = None''', '''                if self._fib_shadow is not None:
                    _pi_est = next(e for e in reversed(self.events) if e.type == "CTS_ESTABLISHED")
                    self._fib_shadow.on_cts_established(
                        _pi_est, self._resolver_df(), int(st.bos.idx), float(st.bos.price))
                # New cycle => reset CTS confirmation guard & threshold
                st.confirmed_cts_anchor_idx = None''')
    ms.sub('''            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_UPDATED", price=price, meta=meta2)
        )
        self.state.cts_event = "CTS_UPDATED"''', '''            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_UPDATED", price=price, meta=meta2)
        )
        self.state.cts_event = "CTS_UPDATED"
        if self._fib_shadow is not None:
            self._fib_shadow.on_cts_updated(self.events[-1], self._resolver_df())''')
    ms.sub('''            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=cts_price_val, meta=meta2)
        )''', '''            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=cts_price_val, meta=meta2)
        )
        if self._fib_shadow is not None:
            self._fib_shadow.on_cts_confirmed(self.events[-1])''')
    ms.sub('''            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED", price=float(price), meta=meta2)
        )''', '''            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED", price=float(price), meta=meta2)
        )
        if self._fib_shadow is not None:
            self._fib_shadow.on_cts_threshold_updated(self.events[-1], self._resolver_df())''')
    ms.sub('''        st = self.state
        if st.cts is None or st.bos is None or self._poi_inners_resolver is None:
            st.poi_inners_for_cycle = []
            return
''', '''        st = self.state
        if self._fib_shadow is not None:
            # Q7: the SAME fib decision as the drawn POI — the shadow's representative fib of this cycle.
            st.poi_inners_for_cycle = []
            if st.cts is None or st.bos is None:
                return
            sid, cyc = int(st.structure_id), int(st.cts_cycle_id)
            fs = None
            latest = self._fib_shadow._get_latest_cross(sid, cyc)
            if latest is not None and latest[1].active:
                fs = latest[1]
            else:
                single = self._fib_shadow._fibs.get((sid, cyc))
                if single is not None and single.active:
                    fs = single
            if fs is not None:
                from engine_v2.zones.poi_zones import POIConfig, select_ip_ic
                df_r = self._resolver_df()
                sel = select_ip_ic(df_r, fs, POIConfig(fill_threshold=self._fill_threshold))
                if sel.pick is not None:
                    ic = int(sel.pick.ic_idx)
                    st.poi_inners_for_cycle = [float(df_r.loc[ic, "h"]) if self.struct_direction == 1
                                               else float(df_r.loc[ic, "l"])]
            # shadow log: today's resolver answer for the same refresh (behaviour-neutral)
            _lab = getattr(self._fib_shadow, "_pi_label", None)
            if _lab is not None and self._poi_inners_resolver is not None:
                if st.cts_cycle_id == 0:
                    self._update_cycle0_data(moment_idx)
                _old = self._poi_inners_resolver(
                    self._resolver_df(), int(st.bos.idx), float(st.bos.price), int(st.cts.idx),
                    float(st.cts.price), int(self.struct_direction), int(st.structure_id), int(st.cts_cycle_id),
                    float(self._fill_threshold), st.cycle0_data, fill_horizon_idx=int(moment_idx),
                    snapshot_horizon_idx=int(st.cts_established_moment_idx))
                from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG
                st._pi_old_inners = [float(x) for x in _old]
                _PI_LOG.append({"kind": "inner", "label": _lab, "sid": sid, "cycle": cyc, "moment": int(moment_idx),
                                "new": list(st.poi_inners_for_cycle), "old": [float(x) for x in _old],
                                "fib": None if fs is None else [int(fs.bos_idx), int(fs.cts_idx)],
                                "ms_bos_cts": [int(st.bos.idx), int(st.cts.idx)]})
            return
        if st.cts is None or st.bos is None or self._poi_inners_resolver is None:
            st.poi_inners_for_cycle = []
            return
''')
    ms.sub('''        hit = self._check_proximity_at_candle(i)
        if hit is None:
            return
        trigger_inner, zone_kind = hit
        self._fire_cts_confirmation_via_proximity(i, trigger_inner, zone_kind)
''', '''        hit = self._check_proximity_at_candle(i)
        _lab = getattr(self._fib_shadow, "_pi_label", None) if self._fib_shadow is not None else None
        if _lab is not None:
            _old_in = getattr(st, "_pi_old_inners", None)
            hit_old = _check_sd_proximity_at_candle(
                self.df, candle_idx=i, struct_direction=self.struct_direction, bos_inner=st.bos_inner_for_cycle,
                threshold=self._proximity_threshold, poi_inners=_old_in or None)
            if (hit is None) != (hit_old is None) or (hit is not None and hit_old is not None and hit != hit_old):
                from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG
                _PI_LOG.append({"kind": "prox_flip", "label": _lab, "sid": int(st.structure_id),
                                "cycle": int(st.cts_cycle_id), "i": int(i), "new": hit, "old": hit_old,
                                "new_inners": list(st.poi_inners_for_cycle), "old_inners": _old_in,
                                "bos_inner": st.bos_inner_for_cycle})
            from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG2
            _PI_LOG2.append({"kind": "prox_check", "label": _lab, "cycle": int(st.cts_cycle_id), "i": int(i),
                             "hit": hit is not None})
        if hit is None:
            return
        trigger_inner, zone_kind = hit
        self._fire_cts_confirmation_via_proximity(i, trigger_inner, zone_kind)
''')
    ms.save()

    se = F("structure/structure_engine.py")
    se.sub('''    kwargs.setdefault("poi_inners_resolver", compute_poi_inners_for_cycle)
    return MarketStructure(''', '''    kwargs.setdefault("poi_inners_resolver", compute_poi_inners_for_cycle)
    # PLAN_I V5: the embedded FibTracker with this structure's cross-fib limit (default: no P).
    _pi_lim = kwargs.pop("cross_fib_limit", None) or {"price": None, "source": "none", "started_by": None,
                                                       "from_sub": None}
    from engine_v2.zones.fib_tracker import FibTracker as _PIFT, FibTrackerConfig as _PIFC
    kwargs["fib_shadow"] = _PIFT(_PIFC(fib_levels=[30.0, 50.0, 61.8, 80.0], fill_threshold=0.70),
                                 fib_mode="cross_cycle",
                                 cross_fib_limits={int(kwargs.get("structure_id", 0)): _pi_lim})
    return MarketStructure(''')
    se.sub('''        ms = _make_market_structure(
            df2, struct_direction=struct_direction, start_idx=start_idx,
            structure_id=structure_id, timeframe=timeframe, pip_size=pip_size,
            enforce_cts0_new_extreme=_use_cts0_scan,
            bos0_inner=cur_bos0_inner if _use_cts0_scan else None,
        )
        ms.debug = True
        df2, ms_events, levels = ms.run()
''', '''        ms = _make_market_structure(
            df2, struct_direction=struct_direction, start_idx=start_idx,
            structure_id=structure_id, timeframe=timeframe, pip_size=pip_size,
            enforce_cts0_new_extreme=_use_cts0_scan,
            bos0_inner=cur_bos0_inner if _use_cts0_scan else None,
            cross_fib_limit=_pi_next_limit,
        )
        ms._fib_shadow._pi_label = f"ms|H1.main|sid{structure_id}"
        ms.debug = True
        df2, ms_events, levels = ms.run()
        _pi_sc = [e for e in ms_events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
        _pi_next_limit = ({"price": float(_pi_sc[0].meta["bos_frozen"]), "source": "reversal_bos_frozen",
                           "started_by": "reversal", "from_sub": None} if _pi_sc else None)
''')
    se.sub('''    # Run multiple structure segments until no more reversals (or we hit end)
    max_structures_guard = 20''', '''    _pi_next_limit = {"price": None, "source": "none", "started_by": "start_trading", "from_sub": None}
    # Run multiple structure segments until no more reversals (or we hit end)
    max_structures_guard = 20''')
    se.sub('''    enforce_cts0_new_extreme: bool = False,
    bos0_inner: Optional[float] = None,
) -> BoundedStructureResult:''', '''    enforce_cts0_new_extreme: bool = False,
    bos0_inner: Optional[float] = None,
    cross_fib_limit=None,
) -> BoundedStructureResult:''')
    se.sub('''        enforce_cts0_new_extreme=enforce_cts0_new_extreme,
        bos0_inner=bos0_inner,
    )
    ms.debug = True
    df2, events, levels = ms.run()''', '''        enforce_cts0_new_extreme=enforce_cts0_new_extreme,
        bos0_inner=bos0_inner,
        cross_fib_limit=cross_fib_limit,
    )
    global _PI_LAST_SHADOW
    _PI_LAST_SHADOW = ms._fib_shadow
    global _PI_GEOM_N
    _PI_GEOM_N += 1
    ms._fib_shadow._pi_label = f"ms|geom|{_PI_GEOM_N}"
    ms.debug = True
    df2, events, levels = ms.run()''')
    se.sub('''def compute_bounded_structure(''', '''_PI_LAST_SHADOW = None
_PI_GEOM_N = 0


def compute_bounded_structure(''')
    se.save()

    edm = F("multitf/entity_df_mutation.py")
    edm.sub('''    from engine_v2.structure.structure_engine import compute_bounded_structure
    try:
        bounded = compute_bounded_structure(
            trigger_df,
            start_idx=start_in_slice,
            struct_direction=int(sd),
            end_idx=run_cap_in_slice,
            timeframe=timeframe,
            enforce_cts0_new_extreme=(bos0_inner is not None),
            bos0_inner=bos0_inner,
        )''', '''    from engine_v2.structure.structure_engine import compute_bounded_structure
    global _PI_PENDING_LIMIT
    _pi_lim, _PI_PENDING_LIMIT = _PI_PENDING_LIMIT, None
    try:
        bounded = compute_bounded_structure(
            trigger_df,
            start_idx=start_in_slice,
            struct_direction=int(sd),
            end_idx=run_cap_in_slice,
            timeframe=timeframe,
            enforce_cts0_new_extreme=(bos0_inner is not None),
            bos0_inner=bos0_inner,
            cross_fib_limit=_pi_lim,
        )''')
    edm.sub('''    bounded, slice_begin = geom
    sub, created = pool.get_or_create(key)
    assert created''', '''    bounded, slice_begin = geom
    sub, created = pool.get_or_create(key)
    assert created
    import engine_v2.structure.structure_engine as _pi_se
    if _pi_se._PI_LAST_SHADOW is not None:
        from engine_v2.multitf.cross_fib_limit import LOG as _PI_LOG
        _PI_LOG.append({"kind": "geom_map", "label": _pi_se._PI_LAST_SHADOW._pi_label, "sub_id": int(sub.sub_id)})''')
    edm.sub('''def _build_geometry(
    m15_df: pd.DataFrame,''', '''_PI_PENDING_LIMIT = None


def _build_geometry(
    m15_df: pd.DataFrame,''')
    edm.save()


def g_strict_limits():
    """A cross_cycle tracker consulted for a structure with no limit entry raises (the build makes the limit
    REQUIRED)."""
    ft = F("zones/fib_tracker.py")
    ft.sub('''        lim = self._limits.get(sid)
        if lim is None:
            return True
''', '''        lim = self._limits.get(sid)
        if lim is None:
            raise RuntimeError(f"PI census: no cross_fib_limit for sid {sid}")
''')
    ft.save()


def g_no_h1():
    """FibTracker's h1 mode is retired (S4): constructing one raises."""
    ft = F("zones/fib_tracker.py")
    ft.sub('''        self.fib_mode = fib_mode  # "h1" (default) or "cross_cycle"
''', '''        self.fib_mode = fib_mode  # "h1" (default) or "cross_cycle"
        if fib_mode != "cross_cycle":
            raise RuntimeError("PI census: FibTracker h1 mode retired (S4)")
''')
    ft.save()


def g_no_old_resolver():
    """The MS in-flight resolver's anchor selection is retired (S5): `select_fib_anchor_for_cycle` and
    `compute_poi_inners_for_cycle` raise; MS's own cycle-0 snapshot too."""
    ft = F("zones/fib_tracker.py")
    ft.sub('''    if sid < 1 or cycle_id != 1 or c0_data is None:
        return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "intra")
''', '''    raise RuntimeError("PI census: select_fib_anchor_for_cycle retired (S5)")
''')
    ft.save()
    pz = F("zones/poi_zones.py")
    pz.sub('''    if cts_idx <= bos_idx:
        return []

    try:
        sd = int(struct_direction)
''', '''    raise RuntimeError("PI census: compute_poi_inners_for_cycle retired (S5)")

    try:
        sd = int(struct_direction)
''')
    pz.save()
    ms = F("structure/market_structure.py")
    ms.sub('''            if _lab is not None and self._poi_inners_resolver is not None:''',
           '''            if False:  # PI census: today's resolver is retired, so the side-by-side log is off''')
    ms.save()


for g in ORDER[VARIANT]:
    globals()[f"g_{g}"]()
print(f"[variant_patch_i] {VARIANT} applied to {TREE}: {ORDER[VARIANT]}")
