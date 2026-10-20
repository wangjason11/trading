"""Read-only edge-case probes of ProximityOverlay against PLAN_R §1 (scratch; reuses the test helpers)."""
from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE as CONF, LENS_COUNTER as CTR, SubStructurePool
from engine_v2.tests.test_proximity_overlay import _Run, _sub, _record, _bos, _poi, _ref
from engine_v2.tests.test_proximity_reference_ladders import _cycle, _df, _scan, _set


def show(run, hs, side):
    out = []
    for h in hs:
        r = _ref(run, h, side)
        out.append((h, None if r is None else (r.key, r.inner)))
    return out


# ---- C: the opp_sd lock basis captured AFTER the refresh advanced to a newer source in the same tick ----
def lock_case(b_record_start):
    """A (counter, c0 BOS 1.4250/1.4260) = opp_sd ref; candle 3 (sd expected) closes 1.4270 -> A passed (from 4).
    B (newer counter sub) record starts at `b_record_start`; its c0 BOS 1.4320/1.4330 (usable 15) lies BEYOND the CTS
    1.4300; its c0 POI 1.4245/1.4255 (inward) is usable at 33 (candle 9). 5 fires sd; 10's high 1.4242 reaches the POI."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    _record(pool, a, CTR, 7, seq=0)
    b = _sub(pool, -1, 12, start=b_record_start, zones=[])
    _record(pool, b, CTR, b_record_start, seq=1, ttype="subsequent_counter")
    views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)],
             b.pool_idx: [_bos(0, 1.4320, 1.4330, 15), _poi(0, 1.4245, 1.4255, 33)]}
    df = _df(16)
    _set(df, 3, h=1.4275, c=1.4270)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4242)
    run = _Run(df, _scan(df, *_cycle()), pool, views)
    trig = run.drive()
    return trig, run.scan.locked, show(run, [4, 5, 9, 10], "opp_sd"), run.overlay._opp_lock_basis


for s in (15, 19):
    trig, locked, refs, basis = lock_case(s)
    print(f"C lock case, B record start {s}: triggers={trig}\n   locked={locked} refs={refs} lock_basis={basis}")


# ---- A: sd eligibility on ANY usable qualifying zone vs rungs from the STRICT most recent cycle ----
def sd_case(s2_record_start):
    """S1 (same-direction, c0 BOS 1.4050 qualifying) is the sd source. S2 (newer) has c0 BOS 1.4060 (qualifying, usable 9)
    and c1 BOS 1.4150 (inside the parent POI 1.41 -> non-qualifying, usable 15). Parent POI passed at close(4)."""
    pool = SubStructurePool()
    s1 = _sub(pool, +1, 4, start=7, zones=[])
    _record(pool, s1, CONF, 7, seq=0, ttype="first_confluence")
    s2 = _sub(pool, +1, 8, start=s2_record_start, zones=[])
    _record(pool, s2, CONF, s2_record_start, seq=1, ttype="subsequent_confluence")
    views = {s1.pool_idx: [_bos(0, 1.4050, 1.4040, 7)],
             s2.pool_idx: [_bos(0, 1.4060, 1.4050, 9), _bos(1, 1.4150, 1.4140, 15)]}
    df = _df(16)
    _set(df, 3, l=1.4105)                 # sd fires on the parent POI
    _set(df, 4, l=1.4085, c=1.4085)       # closes beyond the POI outer 1.4090 -> passed from 5
    run = _Run(df, _scan(df, *_cycle()), pool, views)
    trig = run.drive()
    return trig, run.scan.locked, show(run, [4, 5, 6], "sd"), [r.key for r in run.scan.overlay["sd"]]


for s in (11, 19):
    print(f"A sd case, S2 record start {s}:", sd_case(s))

# ---- F: equal start_idx tie -> higher pool idx ----
pool = SubStructurePool()
a = _sub(pool, -1, 4, start=7, zones=[])
b = _sub(pool, -1, 5, start=7, zones=[])
_record(pool, a, CTR, 7, seq=0)
_record(pool, b, CONF, 7, seq=1, ttype="reversal")
run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)],
                                                      b.pool_idx: [_bos(0, 1.4270, 1.4280, 7)]})
run.drive()
print("F tie:", show(run, [3], "opp_sd"), "pool idx a/b", a.pool_idx, b.pool_idx)

# ---- G: an ended candidate first considered after its end (zone usable before the end) ----
pool = SubStructurePool()
a = _sub(pool, -1, 4, start=7, zones=[])
_record(pool, a, CTR, 7, seq=0)
b = _sub(pool, -1, 9, start=11, zones=[])
_record(pool, b, CTR, 11, seq=1, end=19, ttype="subsequent_counter")
b.end_idx = 19
run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)],
                                                      b.pool_idx: [_bos(0, 1.4240, 1.4250, 13), _bos(1, 1.4230, 1.4235, 21)]})
run.drive()
print("G ended newer candidate:", show(run, [3, 4, 6, 8], "opp_sd"))
