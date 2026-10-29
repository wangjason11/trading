"""PLAN_X X3 — the fib re-ask VARIANT (PLAN_X §X3; PLAN_F §7): patch a SCRATCH tree's engine copy. Never run on the
main checkout.

    python engine_v2/plans/plan_x_inputs/x3_variant_patch.py TREE

TREE holds `engine_v2/` (a `w9b_window_replay`-style copy). Every replacement is an exact string asserted to occur
exactly once (CRLF sources are normalised to LF, patched, and written back CRLF). The rule, as code:

- FibTracker registers a re-ask when a decision answers "no unfilled sd gap" over a window reaching its moment m
  (`hi >= m`): (a) the CTS_ESTABLISHED single (cycle 0 / past the limit), (b) the first activation on a CTS_UPDATED,
  (c) `_update_fib_cts`'s re- / de-activation, (d) `_cross_check`'s own-gap test (EST / UPD / THRESHOLD); one pending
  per (sid, cycle), due m + 1;
- `flush_reasks(upto, df)` runs the due ones as of their due candle (moment = fill horizon = due) before a fed event
  of moment >= due (both feeders) and at every MS candle step; a same-key event at an earlier moment cancels; a
  CTS_CONFIRMED of the key cancels; due past the sid's end (downstream) is dropped;
- MS refreshes its POI snapshot when a re-ask changed the current cycle's fib before confirmation;
- labels `activated_on` / `reactivated_on = "reask"`; `[fib][reask]` log lines + a per-feed summary; the walk-back
  counts any window reaching the moment (not re-asked).
"""
import os
import sys

tree = sys.argv[1]
ENG = os.path.join(tree, "engine_v2")


def patch(rel: str, pairs: list) -> None:
    p = os.path.join(ENG, rel)
    raw = open(p, encoding="utf-8", newline="").read()
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for old, new in pairs:
        n = text.count(old)
        assert n == 1, f"{rel}: expected 1 occurrence, found {n}:\n{old[:200]}"
        text = text.replace(old, new)
    if crlf:
        text = text.replace("\n", "\r\n")
    open(p, "w", encoding="utf-8", newline="").write(text)
    print(f"patched {rel}: {len(pairs)} replacements")


# ---------------------------------------------------------------- zones/cross_cycle_fib.py — the walk-back counter
patch("zones/cross_cycle_fib.py", [
    ("from dataclasses import dataclass, replace\n",
     "from collections import Counter\nfrom dataclasses import dataclass, replace\n"),
    ("from engine_v2.patterns.imbalance import has_unfilled_imbalance\n",
     "from engine_v2.patterns.imbalance import has_unfilled_imbalance\n\n"
     "# PLAN_X X3 VARIANT: the cross walk-back's questions (not re-asked) — examined / a window reaching the moment.\n"
     "REASK_WALK: Counter = Counter()\n"),
    ("""        range_start = min(bos_k[0], cts_k[0])
        range_end = max(bos_k[0], cts_k[0])
        live = has_unfilled_imbalance(""",
     """        range_start = min(bos_k[0], cts_k[0])
        range_end = max(bos_k[0], cts_k[0])
        REASK_WALK["examined"] += 1
        if evaluated_at is not None and range_end >= evaluated_at:
            REASK_WALK["window_reaches_moment"] += 1
            print(f"[fib][reask] WARNING walk-back window [{range_start}, {range_end}] reaches the moment "
                  f"{evaluated_at} (cycle {k}, target {target_cycle})")
        live = has_unfilled_imbalance("""),
])

# ---------------------------------------------------------------- zones/fib_tracker.py — the re-ask
patch("zones/fib_tracker.py", [
    ("from contextlib import contextmanager\n",
     "from collections import Counter\nfrom contextlib import contextmanager\n"),
    ("from engine_v2.zones.cross_cycle_fib import CrossFibLimit, ICReference, beyond, resolve_cross_cycle_eligibility\n",
     "from engine_v2.zones.cross_cycle_fib import (\n"
     "    REASK_WALK, CrossFibLimit, ICReference, beyond, resolve_cross_cycle_eligibility,\n)\n"),
    # state
    ("""        self._est_pattern_anchor: Dict[tuple, int] = {}

    def _log(self, msg: str) -> None:""",
     """        self._est_pattern_anchor: Dict[tuple, int] = {}
        # PLAN_X X3 VARIANT — the fib re-ask (PLAN_F §7): (sid, cycle) -> the pending re-ask of a decision answered
        # False at m over a window reaching m, due m + 1. Plain values (the tracker is deep-copied at MS snapshots).
        self._reasks: Dict[tuple, Dict[str, Any]] = {}
        self._reask_seq = 0
        self._reask_sid_end: Dict[int, int] = {}
        self._reask_fired: set = set()
        self._in_reask = False
        self._reask_suppress_single = False
        self.reask_stats: Counter = Counter()

    def _log(self, msg: str) -> None:"""),
    # the re-ask machinery, after `_moment`
    ("""        if not self._in_event or self._evaluated_at is None:
            raise RuntimeError("FibTracker._moment(): no handled event")
        return self._evaluated_at
""",
     """        if not self._in_event or self._evaluated_at is None:
            raise RuntimeError("FibTracker._moment(): no handled event")
        return self._evaluated_at

    # ------------------------------------------------------------------
    # PLAN_X X3 VARIANT — the fib re-ask (PLAN_F §7; PLAN_X §X3).
    # ------------------------------------------------------------------
    def set_reask_sid_ends(self, ends: Mapping[int, int]) -> None:
        \"\"\"The downstream feed's drop rule: a re-ask of sid `s` due after `ends[s]` (its reversal candle, where its
        MS run stopped) never runs.\"\"\"
        self._reask_sid_end = {int(k): int(v) for k, v in ends.items()}

    def _register_reask(self, key: tuple, kind: str, *, hi: int, **args: Any) -> None:
        \"\"\"Record a re-ask of the decision just answered False at the moment m, if its window reaches m (a gap with
        c2 = m forms at m + 1). One pending per key: a later registration replaces it.\"\"\"
        m = self._moment()
        if int(hi) < m:
            self.reask_stats[f"not_exposed_{kind}"] += 1
            return
        self._reask_seq += 1
        self._reasks[(int(key[0]), int(key[1]))] = dict(
            kind=kind, m=int(m), due=int(m) + 1, seq=self._reask_seq, hi=int(hi), **args)
        self.reask_stats[f"registered_{kind}"] += 1

    def _reask_cancel(self, event: StructureEvent) -> None:
        self._reask_cancel_key((int(event.meta.get("structure_id", 0)), int(event.meta.get("cycle_id", 0))))

    def _reask_cancel_key(self, key: tuple) -> None:
        if self._reasks.pop((int(key[0]), int(key[1])), None) is not None:
            self.reask_stats["cancelled_same_candle"] += 1

    @contextmanager
    def _at_moment(self, moment: int):
        assert not self._in_event, "FibTracker re-ask inside a handler"
        previous = self._evaluated_at
        self._in_event = True
        self._in_reask = True
        self._evaluated_at = int(moment)
        try:
            yield
        finally:
            self._evaluated_at = previous
            self._in_event = False
            self._in_reask = False

    def flush_reasks(self, upto: int, df: pd.DataFrame) -> None:
        \"\"\"Run every pending re-ask due by `upto`, in (due, registration) order, each as of its due candle.\"\"\"
        due = sorted((rec["due"], rec["seq"], key) for key, rec in self._reasks.items() if rec["due"] <= int(upto))
        for d, _, key in due:
            rec = self._reasks.pop(key)
            end = self._reask_sid_end.get(key[0])
            if end is not None and d > end:
                self.reask_stats["dropped_past_sid_end"] += 1
                continue
            before = {k: v for k, v in self._fibs.items() if k[0] == key[0] and k[1] == key[1]}
            with self._at_moment(d):
                self._run_reask(key, rec, df)
            after = {k: v for k, v in self._fibs.items() if k[0] == key[0] and k[1] == key[1]}
            self.reask_stats[f"ran_{rec['kind']}"] += 1
            if before.keys() != after.keys() or any(after[k] is not before[k] for k in after):
                self.reask_stats[f"CHANGED_{rec['kind']}"] += 1
                self._reask_fired.add(key)
                self._log(f"[fib][reask] sid={key[0]} cycle={key[1]} {rec['kind']}: False at {rec['m']} (window end "
                          f"{rec['hi']}) -> re-asked at {d}: CHANGED")

    def pop_reask_fired(self) -> set:
        fired, self._reask_fired = self._reask_fired, set()
        return fired

    def _run_reask(self, key: tuple, rec: Dict[str, Any], df: pd.DataFrame) -> None:
        sid, cycle_id = key
        d = self._moment()
        kind = rec["kind"]
        if kind == "single":
            if key in self._fibs:
                self.reask_stats["skip_single_exists"] += 1
                return
            if not self._has_unfilled(df, rec["lo"], rec["hi"], d, rec["sd"]):
                return
            meta = {"activated_at": d}
            if cycle_id >= 1:
                meta["via"] = "cross_fib_limit"
            self._activate_fib(sid=sid, cycle_id=cycle_id, sd=rec["sd"], bos_idx=rec["bos_idx"],
                               bos_price=rec["bos_price"], cts_idx=rec["cts_idx"], cts_price=rec["cts_price"],
                               meta=meta)
        elif kind == "update":
            state = self._fibs.get(key)
            if state is None or state.locked or state.active or int(state.cts_idx) != int(rec["cts_idx"]):
                self.reask_stats["skip_update_state"] += 1
                return
            if not self._has_unfilled(df, rec["lo"], rec["hi"], d, rec["sd"]):
                return
            self._assert_no_active_cross(sid, cycle_id)
            self._fibs[key] = replace(state, active=True,
                                      meta={**state.meta, "reactivated_at": d, "reactivated_on": "reask"})
        elif kind == "cross":
            if self._phase.get(key) == "confirmed":
                self.reask_stats["skip_cross_confirmed"] += 1
                return
            self._cross_check(sid=sid, target_cycle=cycle_id, df=df, sd=rec["sd"], current_candle=d,
                              anchor_idx=rec["anchor_idx"], anchor_price=rec["anchor_price"],
                              own_imb_start=rec["own_imb_start"], own_window_end_idx=rec["own_end"])
        else:
            raise AssertionError(f"unknown re-ask kind {kind}")

    def log_reask_summary(self) -> None:
        self.reask_stats["pending_past_data_edge"] += len(self._reasks)
        self._log(f"[fib][reask] summary: {dict(sorted(self.reask_stats.items()))}; walk-back (process total) "
                  f"{dict(sorted(REASK_WALK.items()))}")
"""),
    # cancels: a fed event of the key at a moment before the pending's due
    ("""        self._handled.append(event)
        with self._evaluating(event):
            return self._on_cts_established(event, df, bos_idx, bos_price)""",
     """        self._handled.append(event)
        self._reask_cancel(event)
        with self._evaluating(event):
            return self._on_cts_established(event, df, bos_idx, bos_price)"""),
    ("""        self._handled.append(event)
        with self._evaluating(event):
            return self._on_cts_updated(event, df)""",
     """        self._handled.append(event)
        self._reask_cancel(event)
        with self._evaluating(event):
            return self._on_cts_updated(event, df)"""),
    ("""        self._handled.append(event)
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))

        # Populate CTS lookup""",
     """        self._handled.append(event)
        self._reask_cancel(event)
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))

        # Populate CTS lookup"""),
    ("""        own_imb_start = prospective_bos[0]

        self._cross_check(""",
     """        own_imb_start = prospective_bos[0]

        self._reask_cancel_key((sid, target_cycle))
        self._cross_check("""),
    # (a) the CTS_ESTABLISHED single
    ("""            if not has_unfilled:
                self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} NO FIB: no unfilled imbalance "
                      f"(evaluated_at={self._evaluated_at})")
                return None""",
     """            if not has_unfilled:
                self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} NO FIB: no unfilled imbalance "
                      f"(evaluated_at={self._evaluated_at})")
                self._register_reask((sid, cycle_id), "single", hi=max(bos_idx, cts_idx), lo=min(bos_idx, cts_idx),
                                     sd=sd, bos_idx=bos_idx, bos_price=bos_price, cts_idx=cts_idx,
                                     cts_price=cts_price)
                return None"""),
    # (b) the first activation on a CTS_UPDATED
    ("""            if not self._has_unfilled(df, start_idx, end_idx, self._moment(), sd):
                return None""",
     """            if not self._has_unfilled(df, start_idx, end_idx, self._moment(), sd):
                self._register_reask(key, "single", hi=end_idx, lo=start_idx, sd=sd, bos_idx=bos_idx,
                                     bos_price=bos_price, cts_idx=cts_idx, cts_price=cts_price)
                return None"""),
    # (c) the update's re- / de-activation
    ("""        has_unfilled = self._has_unfilled(df, start_idx, end_idx, self._moment(), sd)

        if has_unfilled and not new_state.active:
            # Reactivate - unfilled imbalances now exist in expanded range
            self._assert_no_active_cross(sid, cycle_id)
            new_state = replace(new_state, active=True, meta={**new_state.meta, "reactivated_at": self._moment()})""",
     """        has_unfilled = self._has_unfilled(df, start_idx, end_idx, self._moment(), sd)
        if not has_unfilled and not self._reask_suppress_single:
            self._register_reask(key, "update", hi=end_idx, lo=start_idx, sd=sd, cts_idx=int(new_state.cts_idx))

        if has_unfilled and not new_state.active:
            # Reactivate - unfilled imbalances now exist in expanded range
            self._assert_no_active_cross(sid, cycle_id)
            new_state = replace(new_state, active=True, meta={**new_state.meta, "reactivated_at": self._moment(),
                                                               **({"reactivated_on": "reask"} if self._in_reask else {})})"""),
    # (d) the cross check's own-gap test
    ("""        if not elig.own_has:
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")""",
     """        if not elig.own_has:
            self._register_reask((sid, target_cycle), "cross", hi=max(own_imb_start, own_window_end_idx), sd=sd,
                                 anchor_idx=anchor_idx, anchor_price=anchor_price, own_imb_start=own_imb_start,
                                 own_end=own_window_end_idx)
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")"""),
    ("""            if phase != "pre_established" and (sid, target_cycle) in self._fibs:
                self._update_fib_cts((sid, target_cycle), anchor_idx, anchor_price, df)
            return""",
     """            if phase != "pre_established" and (sid, target_cycle) in self._fibs:
                self._reask_suppress_single = True        # the cross registration above covers the key
                try:
                    self._update_fib_cts((sid, target_cycle), anchor_idx, anchor_price, df)
                finally:
                    self._reask_suppress_single = False
            return"""),
    # labels
    ("""        self._assert_no_active_cross(sid, cycle_id)
        if sd == 1:  # Bullish
            anchor_high = cts_price
            anchor_low = bos_price""",
     """        self._assert_no_active_cross(sid, cycle_id)
        if self._in_reask:
            meta = {**(meta or {}), "activated_on": "reask"}
        if sd == 1:  # Bullish
            anchor_high = cts_price
            anchor_low = bos_price"""),
    ("""                "version": version,
                "activated_at": current_candle,
            },""",
     """                "version": version,
                "activated_at": current_candle,
                **({"activated_on": "reask"} if self._in_reask else {}),
            },"""),
    ("""            meta={**state.meta, "reactivated_at": current_candle},""",
     """            meta={**state.meta, "reactivated_at": current_candle,
                  **({"reactivated_on": "reask"} if self._in_reask else {})},"""),
])

# ---------------------------------------------------------------- pipeline/orchestrator.py — the downstream feed
patch("pipeline/orchestrator.py", [
    ("""    bos_anchor_by_cycle = {}  # {(sid, cycle_id): (bos_anchor_idx, bos_price)}

    for ev in sorted_events:
        sid = ev.meta.get("structure_id", 0)""",
     """    bos_anchor_by_cycle = {}  # {(sid, cycle_id): (bos_anchor_idx, bos_price)}
    # PLAN_X X3 VARIANT: a re-ask of sid s due after its reversal candle never runs (MS's run stopped there).
    fib_tracker.set_reask_sid_ends(compute_reversal_idx_by_sid(events))

    for ev in sorted_events:
        if ev.type in ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED"):
            fib_tracker.flush_reasks(ef.event_moment(ev), df)       # PLAN_X X3 VARIANT
        sid = ev.meta.get("structure_id", 0)"""),
    ("""    # Wire the reversal terminal onto the ended sid's cycles BEFORE finalize so""",
     """    fib_tracker.flush_reasks(len(df) - 1, df)                      # PLAN_X X3 VARIANT: up to the data edge
    fib_tracker.log_reask_summary()

    # Wire the reversal terminal onto the ended sid's cycles BEFORE finalize so"""),
])

# ---------------------------------------------------------------- structure/market_structure.py — MS's embedded tracker
patch("structure/market_structure.py", [
    ("""          - just persist current state (typically RANGE) and write the row
        \"\"\"
        st = self.state
        if not freeze_range:""",
     """          - just persist current state (typically RANGE) and write the row
        \"\"\"
        self._flush_fib_reasks(i)                                    # PLAN_X X3 VARIANT
        st = self.state
        if not freeze_range:"""),
    ("""    def _feed_fib_tracker(self, handler: str, *args: Any) -> None:""",
     """    def _flush_fib_reasks(self, i: int) -> None:
        \"\"\"PLAN_X X3 VARIANT — the fib re-ask at the start of candle `i`'s step (before its raw CTS update and its
        proximity check): the embedded tracker runs every re-ask due by `i`; if one changed the CURRENT cycle's fib
        before confirmation, the POI snapshot is refreshed at `i` (as a CTS_UPDATED at `i` would).\"\"\"
        if self._fib_tracker is None:
            return
        try:
            self._fib_tracker.flush_reasks(int(i), self._resolver_df())
        except RuntimeError:
            raise
        except Exception as exc:
            raise EmbeddedFibTrackerError(
                f"[market_structure] the embedded FibTracker re-ask failed at candle {int(i)}: {exc!r}") from exc
        fired = self._fib_tracker.pop_reask_fired()
        st = self.state
        if fired and st.cts_phase == "EST_OR_UPD" and (int(st.structure_id), int(st.cts_cycle_id)) in fired:
            self._refresh_poi_inners_for_cycle(int(i))

    def _feed_fib_tracker(self, handler: str, *args: Any) -> None:"""),
    ("""        try:
            getattr(self._fib_tracker, handler)(*args)""",
     """        try:
            self._fib_tracker.flush_reasks(ef.event_moment(args[0]), self._resolver_df())   # PLAN_X X3 VARIANT
            getattr(self._fib_tracker, handler)(*args)"""),
])
print("variant patch applied")
