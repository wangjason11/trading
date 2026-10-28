"""PLAN_X X1 — `REVERSAL_CANDIDATE.meta["pattern"]` names the scheduled reversal pattern.

Until 2026-10-28 `_schedule_reversal_from_anchor` read `getattr(ev_r, "pat") or getattr(ev_r, "pattern") or "?"` on a
`PatternEvent`, whose name field is `name` — so every candidate (and the `[RV_SCHEDULE]` / `[RV_APPLY]` debug lines)
said `"?"` (GOTCHAS "`REVERSAL_CANDIDATE.meta["pattern"]` Is Always `"?"`", found by Plan A's L4 verifier). The
candidate that realises is applied by the STATE_CHANGED -> reversal, which names the same `PatternEvent` (`ev.name`),
so the two must agree. Two paths: a pending reversal applied by `_maybe_apply_pending_reversal` (the `[RV_APPLY]`
line) and a reversal winner applied at its own anchor's step.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_ms_bounded_equals_truncated as _bet
from engine_v2.tests._cross_fib_limits import NO_P
from engine_v2.tests.test_unified_probe import _make_cycle0_reversal_data, _prepare_df

_CASES = {
    # (rows, sd, the realised reversal candle, how it applies)
    "pending_reversal_applied": (_bet._make_reversing_data, -1, 11),
    "reversal_winner": (_make_cycle0_reversal_data, 1, 10),
}


def _run(rows_fn, sd):
    out = io.StringIO()
    with redirect_stdout(out):
        res = compute_bounded_structure(_prepare_df(rows_fn()), 0, sd, cross_fib_limit=NO_P,
                                        cycle0_proximity_blocked=True)
    return res, out.getvalue()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_candidate_names_the_pattern_its_reversal_applies(case):
    rows_fn, sd, rev_idx = _CASES[case]
    res, _ = _run(rows_fn, sd)
    reversals = [e for e in res.events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"]
    assert [int(e.idx) for e in reversals] == [rev_idx]
    realised = [e for e in res.events
                if e.type == "REVERSAL_CANDIDATE" and int(e.meta["apply_idx"]) == rev_idx]
    assert realised, "the fixture's reversal was scheduled as a candidate"
    for cand in realised:
        assert cand.meta["pattern"] == reversals[0].meta["pat"]
    for cand in (e for e in res.events if e.type == "REVERSAL_CANDIDATE"):
        assert cand.meta["pattern"] not in ("?", "", None)


def test_the_rv_schedule_and_rv_apply_lines_name_the_pattern():
    res, log = _run(_bet._make_reversing_data, -1)
    pat = next(e.meta["pat"] for e in res.events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal")
    applied = [line for line in log.splitlines() if line.startswith("[RV_APPLY]")]
    scheduled = [line for line in log.splitlines() if line.startswith("[RV_SCHEDULE]")]
    assert applied and scheduled
    assert all(line.endswith(f"pat={pat}") for line in applied)
    assert all(f"pat={pat} " in line for line in scheduled if "apply=11 " in line)
    assert not [line for line in applied + scheduled if "pat=?" in line]
