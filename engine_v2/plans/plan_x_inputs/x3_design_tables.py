"""PLAN_X X3 (design input, measurement only) — for every fib decision the X2 census found EXPOSED (an imbalance
answer False at the decision's moment m that turns True at m + 1, the c3 of a gap whose c2 is the window's last
candle), the facts the re-ask design turns on:

  - the next event the SAME tracker is fed for the decision's cycle (CTS_ESTABLISHED / CTS_UPDATED / CTS_CONFIRMED of
    (sid, cycle); CTS_THRESHOLD_UPDATED whose target cycle = cycle) and its moment: AT m + 1 -> that event asks the
    question itself (a re-ask at m + 1 = today); LATER or NONE -> a re-ask at m + 1 changes the timing;
  - a candle table m - 2 .. m + 2: OHLC, the sd gap that forms at m + 1 (its c1 / c2 / c3, formed_at, bounds) and
    the events fed for the cycle on each candle.

One window per run, in-process, nothing written to `artifacts/` (X2's harness):

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_x_inputs/x3_design_tables.py START END OUT.txt

Every wrapper calls the original and returns ITS value — no behaviour change. Prints how many False answers it
EXAMINED (a 0 means "0 of N").
"""
from __future__ import annotations

import dataclasses
import inspect
import io
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import engine_v2.run_replay as rr
from engine_v2.patterns import imbalance as imb
from engine_v2.structure import event_fields as ef
from engine_v2.structure import market_structure as msm
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


# ------------------------------------------------------------------ the feed of every tracker
FEED: dict = defaultdict(list)        # id(tracker) -> [(type, moment, sid, key_cycle, cts_anchor, method)]
DF: dict = {}                         # id(tracker) -> the frame it was fed
QUIET: dict = {}                      # id(tracker) -> True = MS's embedded tracker
KEEP: list = []                       # every tracker stays alive: an id() is never reused by a later one


def _wrap(name: str):
    orig = getattr(ftm.FibTracker, name)

    def h(self, event, *a, **k):
        if id(self) not in QUIET:
            KEEP.append(self)
        sid = int(event.meta.get("structure_id", 0))
        cyc = int(event.meta.get("cycle_id", 0))
        key_cycle = cyc + 1 if event.type == "CTS_THRESHOLD_UPDATED" else cyc
        anchor = ef.cts_anchor_idx(event) if event.type in ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED") else None
        FEED[id(self)].append((event.type, ef.event_moment(event), sid, key_cycle, anchor,
                               event.meta.get("confirmation_method") or event.meta.get("via")))
        if a and hasattr(a[0], "attrs"):
            DF[id(self)] = a[0]
        QUIET[id(self)] = bool(self._quiet)
        return orig(self, event, *a, **k)
    setattr(ftm.FibTracker, name, h)


for _n in ("on_cts_established", "on_cts_updated", "on_cts_confirmed", "on_cts_threshold_updated"):
    _wrap(_n)

# ------------------------------------------------------------------ the exposed False answers
P = Counter()
HITS: dict = {}                       # dedupe key -> record
_orig_has = imb.has_unfilled_imbalance
_sig_has = inspect.signature(_orig_has)


def _context():
    """(tracker, sid, cycle_id, the decision site) from the stack (X2's `_context`)."""
    tracker = sid = cyc = None
    site = None
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
        f = f.f_back
    return tracker, sid, cyc, site


def _make_has(module_name: str):
    def has(*a, **k):
        r = _orig_has(*a, **k)
        ba = _sig_has.bind(*a, **k)
        ba.apply_defaults()
        g = dict(ba.arguments)
        at = g["evaluated_at"]
        if r or at is None:
            return r
        P[f"false_answers_{module_name}"] += 1
        df = g["df"]
        t = int(at) + 1
        if t > len(df) - 1:
            return r
        g2 = dict(g, evaluated_at=t)
        if int(g["check_to_idx"]) == int(at):          # a moment fill horizon moves with the re-ask
            g2["check_to_idx"] = t
        if not _orig_has(**g2):
            return r
        tracker, sid, cyc, site = _context()
        P[f"EXPOSED_{module_name}"] += 1
        inst = next(iter(imb.iter_unfilled_imbalances(
            g2["df"], g2["start_idx"], g2["end_idx"], g2["check_to_idx"], g2["fill_threshold"],
            direction=g2["direction"], evaluated_at=t)))
        key = (id(tracker), site, sid, cyc, int(at))
        HITS.setdefault(key, dict(tracker=id(tracker), site=site, sid=sid, cycle=cyc, moment=int(at),
                                  window=(int(g["start_idx"]), int(g["end_idx"])), sd=g["direction"],
                                  inst=(int(inst.start_idx), int(inst.end_idx), int(inst.formed_at),
                                        float(inst.gap_bottom), float(inst.gap_top)),
                                  tf=df.attrs.get("timeframe"), first_t=_t(df, 0), n=len(df)))
        return r
    return has


ftm.has_unfilled_imbalance = _make_has("fib_tracker")
ccf.has_unfilled_imbalance = _make_has("cross_cycle_fib")


def _table(rec) -> None:
    df = DF[rec["tracker"]]
    m = rec["moment"]
    s, e, formed, lo, hi = rec["inst"]
    feed = [x for x in FEED[rec["tracker"]] if x[2] == rec["sid"] and x[3] == rec["cycle"]]
    say(f"      {'idx':>5} {'time':16} {'open':>8} {'high':>8} {'low':>8} {'close':>8}  gap ({'bull' if rec['sd'] == 1 else 'bear'}) / events fed for the cycle")
    for i in range(max(0, m - 2), min(len(df), m + 3)):
        r = df.iloc[i]
        role = []
        if i == s - 1:
            role.append("c1")
        if s <= i <= e:
            role.append("c2")
        if i == formed:
            role.append(f"c3 = formed_at (gap {lo:.5f}-{hi:.5f})")
        evs = [f"{x[0]}{'' if x[5] is None else '/' + str(x[5])}" for x in feed if x[1] == i]
        say(f"      {i:>5} {str(r['time'])[:16]:16} {r['o']:8.5f} {r['h']:8.5f} {r['l']:8.5f} {r['c']:8.5f}  "
            f"{' '.join(role):38} {'; '.join(evs)}")


def main() -> None:
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
        rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)
    fetched = [l for l in log.getvalue().splitlines() if "[data_bridge] Fetched" in l or l.startswith("[floor_seed]")]
    say(f"window {start} -> {end} (H1 from {effective_start}); H1 frame {len(df)} candles")
    for l in fetched:
        say(f"  {l.strip()}")
    say(f"examined: {dict(sorted(P.items()))}")

    # One record per (frame, site, sid, cycle, moment) on the DOWNSTREAM tracker; its MS-embedded twin noted.
    down = [r for r in HITS.values() if not QUIET.get(r["tracker"], False)]
    emb = {(r["first_t"], r["n"], r["site"], r["sid"], r["cycle"], r["moment"])
           for r in HITS.values() if QUIET.get(r["tracker"], False)}
    seen = {}
    for r in down:
        k = (r["first_t"], r["n"], r["site"], r["sid"], r["cycle"], r["moment"])
        seen.setdefault(k, r)
    cls = Counter()
    for k in sorted(seen, key=lambda k: (k[0], k[5])):
        r = seen[k]
        m = r["moment"]
        feed = [x for x in FEED[r["tracker"]] if x[2] == r["sid"] and x[3] == r["cycle"]]
        nxt = next((x for x in feed if x[1] > m), None)
        same = [x for x in feed if x[1] == m]
        if nxt is not None and nxt[1] == m + 1:
            verdict = f"SUPERSEDED at m+1 by {nxt[0]} (moment {nxt[1]}) -> a re-ask at m+1 = today"
            cls["superseded at m+1"] += 1
        else:
            verdict = (f"NO fed event for the cycle at m+1 (next: "
                       f"{'none' if nxt is None else f'{nxt[0]} at {nxt[1]}'}) -> a re-ask at m+1 CHANGES the timing")
            cls["re-ask changes the timing"] += 1
        twin = "MS-embedded twin: yes" if k in emb else "MS-embedded twin: NO"
        say(f"\n  {r['tf']} frame-from {r['first_t']} n={r['n']} {r['site']} sid={r['sid']} cycle={r['cycle']} "
            f"moment m={m} window={r['window']} sd={r['sd']}; {twin}")
        say(f"    fed for the cycle at m: {[x[0] for x in same]}; {verdict}")
        _table(r)
    say(f"\nclassified (downstream decisions): {dict(cls)}")
    open(out_path, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
