"""(4): a newer counter sub whose only usable zone lies BEYOND the parent CTS displaces the held reachable zone."""
from engine_v2.multitf.sub_structure_pool import LENS_COUNTER as CTR, SubStructurePool
from engine_v2.tests.test_proximity_overlay import _Run, _sub, _record, _bos, _ref
from engine_v2.tests.test_proximity_reference_ladders import _cycle, _df, _scan

pool = SubStructurePool()
a = _sub(pool, -1, 4, start=7, zones=[])
_record(pool, a, CTR, 7, seq=0)
b = _sub(pool, -1, 12, start=19, zones=[])
_record(pool, b, CTR, 19, seq=1, ttype="subsequent_counter")
views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)], b.pool_idx: [_bos(0, 1.4320, 1.4330, 19)]}
run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views)
run.drive()
for h in (4, 5, 6):
    r = _ref(run, h, "opp_sd")
    print(h, r.key, r.inner, "overlay:", [x.key for x in run.scan.overlay["opp_sd"]])
print("source", run.overlay._source)
