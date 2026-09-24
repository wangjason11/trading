import os
from engine_v2.structure import market_structure as ms
WHICH = os.environ.get("FLIP", "both")
_oc, _ob = ms.MarketStructure._emit_cts_established, ms.MarketStructure._emit_bos_confirmed
def _c(self, idx, price, **kw):  # E2a: the emitters take keyword-only anchors
    _oc(self, idx, price, **kw)
    if WHICH in ("both", "est"):
        ev = self.events[-1]; ev.idx = int(ev.meta["confirmed_at"])
def _b(self, idx, price, **kw):
    _ob(self, idx, price, **kw)
    if WHICH in ("both", "bos"):
        ev = self.events[-1]; ev.idx = int(ev.meta["confirmed_at"])
ms.MarketStructure._emit_cts_established = _c
ms.MarketStructure._emit_bos_confirmed = _b
