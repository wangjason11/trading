"""Run the given test files in-process with market_structure._FALLBACK_PROXIMITY_PIPS patched (argv[1]) and record
which tests construct MarketStructure with proximity_pips=None (i.e. actually read the fallback)."""
import os, sys, collections
import pytest
import engine_v2.structure.market_structure as ms
VAL = int(sys.argv[1]); files = sys.argv[2:]
ms._FALLBACK_PROXIMITY_PIPS = VAL
HITS = collections.Counter(); ALL = collections.Counter()
_orig = ms.MarketStructure.__init__
def _init(self, *a, **kw):
    t = os.environ.get("PYTEST_CURRENT_TEST", "?").split("::")[0]
    ALL[t] += 1
    if kw.get("proximity_pips") is None:
        HITS[t] += 1
    return _orig(self, *a, **kw)
ms.MarketStructure.__init__ = _init
class P:
    def pytest_terminal_summary(self, terminalreporter):
        st = terminalreporter.stats
        print(f"\n[fallback={VAL}] passed={len(st.get('passed', []))} failed={len(st.get('failed', []))} "
              f"errors={len(st.get('error', []))} skipped={len(st.get('skipped', []))}")
        print("[fallback] files constructing MS with proximity_pips=None:", dict(HITS))
        print("[fallback] files constructing MS at all:", len(ALL))
rc = pytest.main(["-q", "-p", "no:cacheprovider", *files], plugins=[P()])
print("rc", rc)
