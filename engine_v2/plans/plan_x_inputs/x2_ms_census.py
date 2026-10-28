"""PLAN_X X2 — the MS risk census (measurement only; PLAN_X §X2). One window per run, in-process, nothing written to
`artifacts/`:

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_x_inputs/x2_ms_census.py START END OUT.txt

START / END as YYYY-MM-DD (UTC; the config window — the replay auto-extends the start). The window's H1 history, its
floor seeds and its M15 data are fetched exactly as `run_replay` does; `run_pipeline` runs with the shadows below
installed. Every shadow calls the original and returns ITS value — no behaviour changes. Each check prints how many
cases it EXAMINED (a 0 means "0 of N").

  1.1  the probe's moving BOS_0 (`unified_probe._bos0_inner_at_start`, every reset) and its constant ad-hoc BOS_0
       reference zone, on the frame as passed vs on the frame cut at the probe's bound `probe_end_idx`;
  1.3  MS's POI-inner snapshot vs the same read as of the checked candle (`_check_proximity_at_candle`): the inner,
       and the proximity HIT (the decision); the first differing decision per (MS run, sid, cycle);
  2.1  every `has_unfilled_imbalance` answer False in FibTracker / cross_cycle_fib, re-asked 1..5 candles later: an
       exposure = it turns True; then DELAY (the cycle's fib was activated later anyway) vs DROP (never);
  latent (the final event streams + the pool tables):
  1.2  each bounded MS run (end_idx < the frame's last row): the full-frame `attrs["imbalances"]` vs `compute_
       imbalance` on the frame cut at the bound — instances formed by the bound whose bounds / DEGENERACY differ;
  1.6  a kept REVERSAL_CANDIDATE on a sub that ended for another reason, applying after the sub's end;
  1.7  (B5) a BOS_CONFIRMED whose moment == its structure's reversal candle (+ informational: confirmed after the
       watch froze the barrier);
  1.8  two RANGE_STARTED with one range start in one structure (the parked "double"; gone from its fixture since
       PLAN_J J1) + a range RESET before the moment its RANGE_STARTED is confirmed (what the fixture still shows) —
       the detectors run on that fixture first and must flag it.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import io
import itertools
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import engine_v2.run_replay as rr
from engine_v2.patterns import imbalance as imb
from engine_v2.structure import event_fields as ef
from engine_v2.structure import market_structure as msm
from engine_v2.structure import reference_zone as rz
from engine_v2.structure import unified_probe as up
from engine_v2.zones import cross_cycle_fib as ccf
from engine_v2.zones import fib_tracker as ftm

start, end, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
OUT: list[str] = []


def say(s: str = "") -> None:
    OUT.append(s)


def _t(df, idx) -> str:
    try:
        return str(df["time"].iloc[int(idx)])[:16]
    except Exception:
        return "?"


def _positional(df) -> bool:
    return len(df) > 0 and int(df.index[0]) == 0 and int(df.index[-1]) == len(df) - 1


# ------------------------------------------------------------------ 1.1 the probe's BOS_0 vs its bound
P11 = Counter()
P11_rows: list[str] = []
_probe_ctx: list[tuple] = []          # (probe_end_idx, phase, input / starting idx, timeframe)
_orig_p1, _orig_p2, _orig_b0 = up._run_phase1, up._run_phase2, up._bos0_inner_at_start
_sig_p1, _sig_p2 = inspect.signature(_orig_p1), inspect.signature(_orig_p2)


def _check_ref_zone(df, ref, direction, bound, phase, origin) -> None:
    if bound is None or ref is None:
        P11["ref_unbounded"] += 1
        return
    if getattr(ref, "source", None) != "ad_hoc_bos_0":
        P11[f"ref_source_{getattr(ref, 'source', None)}"] += 1
        return
    P11["ref_adhoc_examined"] += 1
    if not _positional(df) or int(ref.anchor_idx) > int(bound):
        P11["ref_adhoc_skipped"] += 1
        return
    cut = rz.build_ad_hoc_bos0_reference_zone(df.iloc[: int(bound) + 1], int(ref.anchor_idx), int(direction))
    full = (float(ref.inner), float(ref.outer))
    cutv = None if cut is None else (float(cut.inner), float(cut.outer))
    if cutv != full:
        P11["ref_adhoc_DIFFERS"] += 1
        P11_rows.append(f"  REF {phase} {df.attrs.get('timeframe')} origin={origin} anchor={ref.anchor_idx} "
                        f"({_t(df, ref.anchor_idx)}) bound={bound} full={full} cut={cutv}")


def _p1(*a, **k):
    ba = _sig_p1.bind(*a, **k)
    g = ba.arguments
    _check_ref_zone(g["df"], g["reference_zone"], g["direction"], g["probe_end_idx"], "phase1", g["input_idx"])
    _probe_ctx.append((g["probe_end_idx"], "phase1", g["input_idx"], g["df"].attrs.get("timeframe")))
    try:
        return _orig_p1(*a, **k)
    finally:
        _probe_ctx.pop()


def _p2(*a, **k):
    ba = _sig_p2.bind(*a, **k)
    g = ba.arguments
    _probe_ctx.append((g["probe_end_idx"], "phase2", g["starting_idx"], g["df"].attrs.get("timeframe")))
    try:
        return _orig_p2(*a, **k)
    finally:
        _probe_ctx.pop()


def _b0(df, current_start, direction):
    res = _orig_b0(df, current_start, direction)
    if not _probe_ctx:
        P11["reset_no_probe_ctx"] += 1
        return res
    bound, phase, origin, tf = _probe_ctx[-1]
    P11["resets_examined"] += 1
    if bound is None:
        P11["resets_unbounded"] += 1
        return res
    if not _positional(df) or int(current_start) > int(bound):
        P11["resets_skipped"] += 1
        return res
    if int(current_start) + 5 > int(bound):
        P11["resets_anchor_plus5_past_bound"] += 1
    cut = _orig_b0(df.iloc[: int(bound) + 1], current_start, direction)
    if tuple(cut) != tuple(res):
        P11["resets_DIFFER"] += 1
        P11_rows.append(f"  RESET {phase} {tf} origin={origin} new_start={current_start} ({_t(df, current_start)}) "
                        f"bound={bound} ({_t(df, bound)}) full={tuple(round(x, 5) for x in res)} "
                        f"cut={tuple(round(x, 5) for x in cut)}")
    return res


up._run_phase1, up._run_phase2, up._bos0_inner_at_start = _p1, _p2, _b0

# ------------------------------------------------------------------ 1.3 MS's POI-inner snapshot vs as of the candle
P13 = Counter()
P13_first: dict = {}
_run_ids = itertools.count()
_orig_chk = msm.MarketStructure._check_proximity_at_candle


def _run_id(ms) -> int:
    if "_x2_run" not in ms.__dict__:
        ms.__dict__["_x2_run"] = next(_run_ids)
    return ms.__dict__["_x2_run"]


def _chk(self, i):
    res = _orig_chk(self, i)
    st = self.state
    P13["candles_examined"] += 1
    if self._fib_tracker is None:
        P13["no_tracker"] += 1
        return res
    try:
        fib = self._fib_tracker.representative_fib(int(st.structure_id), int(st.cts_cycle_id))
        inner = self._poi_inner_for_fib(self._resolver_df(), fib, float(self._fib_tracker.config.fill_threshold),
                                        tracker=self._fib_tracker, evaluated_at=int(i))
    except Exception as exc:                                   # recorded, never raised into the run
        P13["asof_read_raised"] += 1
        P13_first.setdefault(("raise", type(exc).__name__), f"  RAISE {exc!r}"[:300])
        return res
    asof = [] if inner is None else [float(inner)]
    snap = [float(v) for v in (st.poi_inners_for_cycle or [])]
    hit_asof = msm._check_sd_proximity_at_candle(self.df, i, self.struct_direction, st.bos_inner_for_cycle,
                                                 self._proximity_threshold, asof or None)
    if asof:
        P13["candles_with_asof_poi"] += 1
    if snap:
        P13["candles_with_snapshot_poi"] += 1
    if asof != snap:
        P13["inner_differs"] += 1
    if hit_asof != res:
        P13["DECISION_differs"] += 1
        key = (_run_id(self), int(st.structure_id), int(st.cts_cycle_id))
        if key not in P13_first:
            phase = _probe_ctx[-1][1] if _probe_ctx else "run"
            P13_first[key] = (f"  {self.timeframe} {phase} run#{key[0]} start={self.start_idx} ({_t(self.df, self.start_idx)}) "
                              f"sid={key[1]} cycle={key[2]} candle={i} ({_t(self.df, i)}): snapshot={snap} hit={res} | "
                              f"as-of={asof} hit={hit_asof}")
    return res


msm.MarketStructure._check_proximity_at_candle = _chk

# ------------------------------------------------------------------ 2.1 a False imbalance answer, re-asked later
P21 = Counter()
P21_hits: dict = {}                   # dedupe key -> (len(df), record)
_trackers: dict = {}
_orig_has = imb.has_unfilled_imbalance
_sig_has = inspect.signature(_orig_has)


def _context():
    """(tracker, sid, cycle_id, the decision site, embedded-in-MS) from the stack: the site = the first fib_tracker /
    cross_cycle_fib frame that is not the `_has_unfilled` wrapper; sid / cycle_id = the first FibTracker method frame
    that holds them (the handler), walking up from the call."""
    tracker = sid = cyc = None
    site = None
    embedded = False
    f = sys._getframe(2)
    while f is not None:
        loc = f.f_locals
        me = loc.get("self")
        fn = f.f_code.co_filename.replace(chr(92), "/").rsplit("/", 1)[-1]
        if site is None and fn in ("fib_tracker.py", "cross_cycle_fib.py") and f.f_code.co_name != "_has_unfilled":
            site = f"{fn}:{f.f_code.co_name}:{f.f_lineno}"
        if isinstance(me, ftm.FibTracker):
            tracker = tracker or me
            if sid is None and "sid" in loc and ("cycle_id" in loc or "target_cycle" in loc):
                sid, cyc = loc["sid"], loc.get("cycle_id", loc.get("target_cycle"))
        if isinstance(me, msm.MarketStructure):
            embedded = True
        f = f.f_back
    return tracker, sid, cyc, site, embedded


def _make_has(module_name: str):
    def has(*a, **k):
        r = _orig_has(*a, **k)
        ba = _sig_has.bind(*a, **k)
        ba.apply_defaults()
        g = dict(ba.arguments)
        at = g["evaluated_at"]
        if r or at is None:
            return r
        P21[f"false_answers_{module_name}"] += 1
        df = g["df"]
        for kk in range(1, 6):
            t = int(at) + kk
            if t > len(df) - 1:
                break
            g2 = dict(g, evaluated_at=t)
            if int(g["check_to_idx"]) == int(at):          # a moment fill horizon moves with the re-ask
                g2["check_to_idx"] = t
            if _orig_has(**g2):
                tracker, sid, cyc, site, embedded = _context()
                P21[f"EXPOSED_{module_name}_k{kk}"] += 1
                if tracker is not None:
                    _trackers[id(tracker)] = tracker
                key = (df.attrs.get("timeframe"), _t(df, 0), embedded, site, sid, cyc, _t(df, at))
                rec = dict(tf=df.attrs.get("timeframe"), embedded=embedded, site=site, sid=sid, cycle=cyc,
                           moment=int(at), moment_t=_t(df, at), k=kk, window=(int(g["start_idx"]), int(g["end_idx"])),
                           tracker=id(tracker) if tracker is not None else None, n=len(df), first_t=_t(df, 0))
                prev = P21_hits.get(key)
                if prev is None or len(df) > prev[0]:
                    P21_hits[key] = (len(df), rec)
                break
        return r
    return has


ftm.has_unfilled_imbalance = _make_has("fib_tracker")
ccf.has_unfilled_imbalance = _make_has("cross_cycle_fib")

# ------------------------------------------------------------------ 1.2 bounded MS runs: full-frame imbalances
P12 = Counter()
P12_rows: list[str] = []
_orig_run = msm.MarketStructure.run


def _run(self, *a, **k):
    n = len(self.df)
    if self.end_idx is not None and int(self.end_idx) < n - 1 and _positional(self.df):
        B = int(self.end_idx)
        P12["bounded_runs"] += 1
        full = [x for x in (self.df.attrs.get("imbalances") or []) if x.formed_at <= B]
        cut_df = imb.compute_imbalance(self.df.iloc[: B + 1].copy())
        cut = {x.start_idx: x for x in (cut_df.attrs.get("imbalances") or [])}
        for x in full:
            P12["instances_formed_by_bound"] += 1
            y = cut.get(x.start_idx)
            if y is None:
                P12["missing_in_cut"] += 1
                P12_rows.append(f"  MISSING bound={B} inst {x.start_idx}..{x.end_idx} dir={x.direction}")
                continue
            if (x.end_idx, round(x.gap_top, 6), round(x.gap_bottom, 6)) != (y.end_idx, round(y.gap_top, 6), round(y.gap_bottom, 6)):
                P12["bounds_differ"] += 1
            if (x.gap_size <= 0) != (y.gap_size <= 0):
                P12["DEGENERACY_differs"] += 1
                P12_rows.append(f"  DEGENERACY bound={B} ({_t(self.df, B)}) inst {x.start_idx}..{x.end_idx} "
                                f"full gap {x.gap_size:.6f} cut {y.start_idx}..{y.end_idx} gap {y.gap_size:.6f}")
    return _orig_run(self, *a, **k)


msm.MarketStructure.run = _run


# ------------------------------------------------------------------ latent detectors on an event stream
def _streams(events, key_fn):
    out = defaultdict(list)
    for e in events:
        out[key_fn(e)].append(e)
    return out


def range_checks(stream):
    """(a) duplicate range starts; (b) a RANGE_RESET at t while the latest RANGE_STARTED (start S, confirmed C) has
    S <= t < C — the range was reset before the moment it is confirmed."""
    starts = Counter(int(e.meta["range_anchor_idx"]) for e in stream if e.type == "RANGE_STARTED")
    dup = {s: c for s, c in starts.items() if c > 1}
    early = []
    last = None
    for e in stream:
        if e.type == "RANGE_STARTED":
            last = (int(e.meta["range_anchor_idx"]), int(e.meta["confirmed_at"]))
        elif e.type == "RANGE_RESET" and last is not None:
            if last[0] <= int(e.idx) < last[1]:
                early.append((last[0], last[1], int(e.idx)))
            last = None
    return dup, early


def b5_checks(stream):
    """A BOS_CONFIRMED whose moment == its structure's reversal candle; informational: one confirmed after the
    reversal watch froze the barrier (the last REVERSAL_WATCH_START before the reversal) and up to it."""
    exact, after_freeze = [], []
    revs = [int(e.idx) for e in stream if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
    for r in revs:
        watch = [int(e.meta["pattern_anchor_idx"]) for e in stream
                 if e.type == "REVERSAL_WATCH_START" and int(e.meta["pattern_anchor_idx"]) <= r]
        w = max(watch) if watch else None
        for e in stream:
            if e.type != "BOS_CONFIRMED":
                continue
            m = int(e.meta["confirmed_at"])
            if m == r:
                exact.append((r, int(e.idx), m))
            elif w is not None and w < m < r:
                after_freeze.append((r, w, int(e.idx), m))
    return exact, after_freeze


# ------------------------------------------------------------------ the detectors on the fixture first
def _fixture_selfcheck() -> None:
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests._cross_fib_limits import NO_P
    from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
    from engine_v2.tests.test_unified_probe import _prepare_df
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_double_rewind_data()), 0, 1, cross_fib_limit=NO_P)
    dup, early = range_checks(res.events)
    say(f"fixture self-check (_make_double_rewind_data, sd +1): duplicate range starts {dup} (none expected since "
        f"PLAN_J J1); range reset before its confirmation {early} (expected [(9, 13, 12)])")
    assert early == [(9, 13, 12)], early


# ------------------------------------------------------------------ run the window
def main() -> None:
    _fixture_selfcheck()
    rr.CONFIG = dataclasses.replace(
        rr.CONFIG,
        start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        end=datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc),
    )
    log = io.StringIO()
    with redirect_stdout(log):
        df, effective_start, _ = rr.fetch_history_with_auto_extend(
            pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
            end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
        )
        df.attrs["pair"] = rr.CONFIG.pair
        seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
        res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)
    text = log.getvalue()
    fetched = [l for l in text.splitlines() if "[data_bridge] Fetched" in l or l.startswith("[floor_seed]")]
    say(f"window {start} -> {end} (H1 from {effective_start}); H1 frame {len(df)} candles "
        f"{_t(df, 0)} -> {_t(df, len(df) - 1)}")
    for l in fetched:
        say(f"  {l.strip()}")

    say("\n== 1.1 the probe's BOS_0 vs its bound (probe_end_idx) ==")
    say(f"  {dict(sorted(P11.items()))}")
    OUT.extend(P11_rows or ["  (no differing zone)"])

    say("\n== 1.3 MS's POI-inner snapshot vs as of the checked candle ==")
    say(f"  {dict(sorted(P13.items()))}")
    OUT.extend(list(P13_first.values()) or ["  (no differing decision)"])

    say("\n== 2.1 a False imbalance answer that turns True 1..5 candles later ==")
    say(f"  {dict(sorted(P21.items()))}")
    final = {}
    for tid, tr in _trackers.items():
        recs = defaultdict(list)
        for fs in tr.get_fibs_for_charting():
            m = fs.meta
            recs[(int(fs.structure_id), int(fs.cycle_id))].append(
                (m.get("activated_at"),
                 f"{'cross' if m.get('cross_cycle') else 'single'} {fs.bos_idx}..{fs.cts_idx} act={m.get('activated_at')}"
                 f" on={m.get('activated_on')} react={m.get('reactivated_at')} deact={m.get('deactivated_at')}"
                 f" locked={fs.locked} end={fs.end_idx}"))
        final[tid] = recs
    m15_idx = {}
    for _lens, _ldf in (res.meta.get("sub_lens_dfs") or {}).items():
        m15_idx = {str(t)[:16]: i for i, t in enumerate(_ldf["time"])}
        break
    if not P21_hits:
        say("  (no exposure)")
    for _key, (_n, r) in sorted(P21_hits.items(), key=lambda kv: str(kv[0])):
        if r["tf"] == "M15":
            r["moment_t"] = f"{r['moment_t']} = M15 {m15_idx.get(r['moment_t'], '?')}"
        fibs = final.get(r["tracker"], {}).get((r["sid"], r["cycle"]), []) if r["sid"] is not None else []
        site = str(r["site"])
        if ":_on_cts_established:" in site or ":_handle_cross_cycle_cts_updated:" in site:
            acts = sorted(a for a, _ in fibs if a is not None)
            verdict = f"ACTIVATION -> {'DELAY (activated at ' + str(acts) + ')' if acts else 'DROP (never activated)'}"
        elif ":_update_fib_cts:" in site:
            verdict = "UPDATE re-check (the fib's re- / de-activation decided before the gap formed)"
        elif site.startswith("cross_cycle_fib.py:resolve_cross_cycle_eligibility:"):
            line = int(site.rsplit(":", 1)[-1])
            verdict = ("CROSS: the target cycle's own gap read missing -> no cross / an active cross ends here"
                       if line < 290 else "CROSS walk-back: an earlier cycle read dead -> a shorter cross")
        else:
            verdict = "?"
        say(f"  {r['tf']} {'MS-embedded' if r['embedded'] else 'downstream'} frame-from {r['first_t']} n={r['n']} "
            f"{site} sid={r['sid']} cycle={r['cycle']} moment={r['moment']} ({r['moment_t']}) window={r['window']} "
            f"k={r['k']} -> {verdict}")
        for _a, x in fibs:
            say(f"      final fib: {x}")

    say("\n== latent: 1.2 bounded MS runs vs the full-frame imbalance list ==")
    say(f"  {dict(sorted(P12.items()))}")
    OUT.extend(P12_rows or ["  (no differing degeneracy)"])

    h1 = list(res.df.attrs.get("structure_events") or [])
    lens_dfs = dict(res.meta.get("sub_lens_dfs") or {})
    say("\n== latent: 1.6 / 1.7 / 1.8 on the final event streams ==")
    streams = {f"H1 sid {k}": v for k, v in _streams(h1, lambda e: int(e.meta["structure_id"])).items()}
    seen_subs = set()
    for lens, ldf in sorted(lens_dfs.items()):
        evs = list(ldf.attrs.get("events") or [])
        for sub_id, st in _streams(evs, lambda e: e.meta.get("sub_id")).items():
            if sub_id in seen_subs:                      # a sub on both lenses: one stream
                continue
            seen_subs.add(sub_id)
            streams[f"sub {sub_id} ({lens})"] = st
    n_rev = n_rs = 0
    for name, st in streams.items():
        dup, early = range_checks(st)
        exact, after = b5_checks(st)
        n_rev += sum(1 for e in st if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal")
        n_rs += sum(1 for e in st if e.type == "RANGE_STARTED")
        for label, val in (("1.8 duplicate range start", dup), ("1.8 range reset before its confirmation", early),
                           ("1.7 B5 BOS confirmed on the reversal candle", exact),
                           ("1.7 (info) BOS confirmed after the watch froze", after)):
            if val:
                say(f"  {name}: {label}: {val}")
    say(f"  streams {len(streams)}, reversals {n_rev}, RANGE_STARTED {n_rs}")
    n_cand = 0
    for lens, ldf in sorted(lens_dfs.items()):
        recs = {r.sub_id: r for r in (ldf.attrs.get("sids") or [])}
        for e in (ldf.attrs.get("events") or []):
            if e.type != "REVERSAL_CANDIDATE":
                continue
            n_cand += 1
            r = recs.get(e.meta.get("sub_id"))
            if r is None or r.end_event_idx is None:
                continue
            if r.end_reason != "reversal" and int(e.meta["apply_idx"]) > int(r.end_event_idx):
                say(f"  1.6 straddle: {lens} sub {r.sub_id} ended {r.end_reason} at {r.end_event_idx}; candidate "
                    f"anchor {e.meta['pattern_anchor_idx']} apply {e.meta['apply_idx']}")
    say(f"  1.6 kept REVERSAL_CANDIDATEs examined (both lenses): {n_cand}")


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(OUT) + "\n")
        print("\n".join(OUT))
