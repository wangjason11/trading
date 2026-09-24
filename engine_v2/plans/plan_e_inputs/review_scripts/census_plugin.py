import sys, json, os, atexit, collections
from engine_v2.structure import market_structure as ms
_orig = ms.StructureEvent.__init__
REC = collections.Counter()
SITES = collections.defaultdict(set)
def _init(self, *a, **k):
    _orig(self, *a, **k)
    if self.type in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        f = sys._getframe(1)
        while f and "dataclass" in f.f_code.co_filename:
            f = f.f_back
        fn = f.f_code.co_filename.replace("\\", "/")
        if "/tests/" in fn:
            base = os.path.basename(fn)
            m = self.meta or {}
            REC[(base, self.type, "confirmed_at" in m)] += 1
            SITES[base].add(f.f_lineno)
ms.StructureEvent.__init__ = _init
@atexit.register
def _dump():
    out = {"counts": [[*k, v] for k, v in REC.items()], "sites": {k: sorted(v) for k, v in SITES.items()}}
    with open(os.path.join(os.path.dirname(__file__), "census.json"), "w") as fh:
        json.dump(out, fh, indent=1)
