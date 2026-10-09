"""PLAN_J: a STRUCTURAL signature per random-tail trial, through the tree first on PYTHONPATH.

    PYTHONPATH=TREE python core_sig.py waves|real SEED N OUT.jsonl

The search's `sig` hashes every event in emission order + every MS output column, so a re-ordered emission or an
as-of debug column counts as a difference. Here, per trial: `core` = the structure's decisions, order-free —
CTS_ESTABLISHED (cycle, moment, CTS anchor, price), BOS_CONFIRMED (cycle, moment, BOS anchor, price), CTS_CONFIRMED
(cycle, moment, CTS anchor, price, method), the realised reversal (STATE_CHANGED to=reversal: idx), sorted;
`ev` = a hash of the sorted multiset of every event (type, idx, price, sorted meta) — order-free; `trip` = MS's order
tripwire. Same trials as `random_tail_search.py waves|real SEED` (its generators).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys

import engine_v2
import engine_v2.structure.event_fields as ef

tree = os.path.dirname(os.path.dirname(engine_v2.__file__))
spec = importlib.util.spec_from_file_location(
    "rts", os.path.join(tree, "engine_v2", "plans", "plan_e_inputs", "review_scripts", "random_tail_search.py"))
rts = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rts)

stream, seed, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]


def core(events):
    c = []
    for e in events:
        m = e.meta
        if e.type == "CTS_ESTABLISHED":
            c.append(("EST", m["cycle_id"], int(e.idx), ef.cts_anchor_idx(e), e.price))
        elif e.type == "BOS_CONFIRMED":
            c.append(("BOS", m["cycle_id"], int(e.idx), ef.bos_anchor_idx(e), e.price))
        elif e.type == "CTS_CONFIRMED":
            c.append(("CONF", m["cycle_id"], int(e.idx), ef.cts_anchor_idx(e), e.price, m.get("confirmation_method")))
        elif e.type == "STATE_CHANGED" and m.get("to") == "reversal":
            c.append(("REV", int(e.idx)))
    return sorted(c, key=repr)


def evhash(events):
    rows = sorted(repr((e.type, int(e.idx), e.price, sorted((k, repr(v)) for k, v in e.meta.items()))) for e in events)
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:16]


gen = rts.wave_trials(seed) if stream == "waves" else rts.real_trials(seed)
with open(out, "w") as f:
    for tup in gen:
        t = tup[0]
        if t >= n:
            break
        if stream == "waves":
            _, sd, tf, base, rows, nw = tup
            start = 0
        else:
            _, sd, tf, off, start, length, jit, rows = tup
        rec = {"t": t}
        try:
            df = rts._prepare_df(rows)
            ms = rts._make_market_structure(df, cross_fib_limit=rts._NO_P, struct_direction=sd, start_idx=start,
                                            structure_id=0, timeframe=tf, pip_size=rts._pip_size_from_pair(df))
            with contextlib.redirect_stdout(io.StringIO()):
                rts._ms_run(ms)
            rec.update(core=core(ms.events), ev=evhash(ms.events), trip=bool(getattr(ms, "_rts_tripwire", None)))
        except Exception as x:  # noqa: BLE001
            rec["err"] = f"{type(x).__name__}: {str(x)[:120]}"
        f.write(json.dumps(rec) + "\n")
