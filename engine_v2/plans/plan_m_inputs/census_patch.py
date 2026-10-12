"""PLAN_M — the LIVE-`P` CENSUS patch (PLAN_M §6; read-only for the engine's behaviour: it only PRINTS).

After the sweep, prints one `[census] h1 {json}` line (the H1 BOS_CONFIRMED / BOS_THRESHOLD_UPDATED facts every
counter sub's live `P` would read) and one `[census] sub {json}` line per numbered unique sub (its started_by record's
parent cycle, counter or not, its life, the parent cycle's threshold moves with their LOH moments, and its own
cycles' establishment moments + confirmed CTS prices — what `census_report.py` needs to replay the decided rule
offline). The engine's outputs are unchanged.

    python census_patch.py TREE START END      (TREE = a `git archive` of the base; START / END = YYYY-MM-DD; the
                                                active CONFIG block's window is rewritten in TREE only)
"""
import sys
from pathlib import Path

TREE, START, END = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def patch(rel, edits):
    p = TREE / rel
    src = p.read_text(encoding="utf-8")
    for old, new in edits:
        n = src.count(old)
        assert n == 1, (rel, n, old[:100])
        src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")


def ymd(s):
    y, m, d = (int(x) for x in s.split("-"))
    return f"datetime({y}, {m}, {d}, 0, 0, 0, tzinfo=timezone.utc)"


patch("engine_v2/config.py", [
    ("    start=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),\n"
     "    end=datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc),\n"
     "    lower_timeframes=(\"M15\",),\n",
     f"    start={ymd(START)},\n    end={ymd(END)},\n    lower_timeframes=(\"M15\",),\n"),
])

CENSUS = '''
    # ---- PLAN_M CENSUS (prints only) ----
    import json as _cj
    _c_bos, _c_thr = {}, {}
    for _e in sorted_events:
        if _e.type in ("BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"):
            _k = (int(_e.meta["structure_id"]), int(_e.meta["cycle_id"]))
            (_c_bos if _e.type == "BOS_CONFIRMED" else _c_thr).setdefault(_k, []).append(
                (int(ef.event_moment(_e)), float(_e.price)))
    _c_not_further, _c_after_end = [], []
    for _k, _v in _c_thr.items():
        _psd = tables.parent_sd.get(_k[0])
        _prev = _c_bos[_k][-1][1] if _k in _c_bos else None
        for _m, _p in _v:
            if _prev is not None and not ((_p < _prev) if _psd == 1 else (_p > _prev)):
                _c_not_further.append([_k[0], _k[1], _m, _prev, _p])
            _prev = _p
            _eh = tables.end_h1.get(_k)
            if _eh is not None and _m >= _eh:
                _c_after_end.append([_k[0], _k[1], _m, _eh])
    print("[census] h1 " + _cj.dumps({
        "n_bos": sum(len(v) for v in _c_bos.values()),
        "bos_not_one": [[k[0], k[1], len(v)] for k, v in _c_bos.items() if len(v) != 1],
        "n_thr": sum(len(v) for v in _c_thr.values()),
        "thr_cycles": sorted([k[0], k[1], len(v)] for k, v in _c_thr.items()),
        "thr_not_further": _c_not_further,
        "thr_at_or_after_cycle_end": _c_after_end,
        "cycles": len(tables.cts_moment),
    }))
    for _s in sorted((s for s in sub_pool.all() if s.sub_id is not None), key=lambda s: s.sub_id):
        _live = _s.live_records()
        if not _live:
            continue
        _f = min(_live, key=lambda r: (r.start_idx, r.seq))
        _S, _C = int(_f.parent_sid), int(_f.parent_cycle_id)
        _bounded, _sb = _s.geometry
        _cyc = {}
        for _e in _bounded.events:
            if _e.type == "CTS_ESTABLISHED":
                _cyc.setdefault(int(_e.meta["cycle_id"]), {})["est"] = int(ef.event_moment(_e)) + int(_sb)
            elif _e.type == "CTS_CONFIRMED":
                _cyc.setdefault(int(_e.meta["cycle_id"]), {})["conf_price"] = float(_e.price)
                _cyc[int(_e.meta["cycle_id"])]["conf_at"] = int(ef.event_moment(_e)) + int(_sb)
        _lim = _s.cross_fib_limit
        print("[census] sub " + _cj.dumps({
            "sub_id": int(_s.sub_id), "dir": int(_s.direction), "parent": [_S, _C],
            "parent_sd": int(tables.parent_sd[_S]), "counter": int(_s.direction) != int(tables.parent_sd[_S]),
            "first_type": _f.trigger_type, "n_records": len(_s.records),
            "start": int(_s.start_idx), "end": None if _s.end_idx is None else int(_s.end_idx),
            "end_reason": _s.end_reason,
            "parent_floor_m15": int(tables.floor(_S, _C)), "parent_end_m15": tables.end(_S, _C),
            "parent_end_h1": tables.end_h1.get((_S, _C)),
            "bos": _c_bos.get((_S, _C), []),
            "bos_m15": [_loh(m) for m, _p in _c_bos.get((_S, _C), [])],
            "moves": [[m, _loh(m), p] for m, p in _c_thr.get((_S, _C), [])],
            "P_now": _lim.price, "P_src": _lim.source,
            "cycles": {str(k): v for k, v in sorted(_cyc.items())},
        }))
    # ---- end PLAN_M CENSUS ----
'''

patch("engine_v2/pipeline/orchestrator.py", [(
    "    check_cross_fib_limits(sub_pool, sorted_events)\n",
    "    check_cross_fib_limits(sub_pool, sorted_events)\n" + CENSUS,
)])
print(f"[census_patch] {TREE}: window {START} -> {END}")
