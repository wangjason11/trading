"""PLAN_I S5 step A (session 5, 2026-10-08) — §9.5 d's evidence that MS's new read is not dead code, measured on
the BUILT code (the V5 numbers were measured on the pre-revamp code + the V5 patch).

    cd WORKDIR && python REPO/engine_v2/plans/plan_i_inputs/s5_agreement.py REPO OUT.json

WORKDIR holds a copy of `oanda.cfg` (the replay writes WORKDIR/artifacts/, the main checkout is never touched).
Needs the S5 STEP-A tree: MS's embedded tracker wired AND the retired resolver still importable + MS's cycle-0
snapshot still maintained (step B deletes both). Instruments, in process, without touching the tree:

- every FibTracker handler: (type, cycle, stamped idx, moment, the cycle's representative fib after it);
- MS's POI refresh: the new inner (the embedded tracker's representative fib → `poi_inner_for_fib`) next to the
  retired resolver's (`poi_zones.compute_poi_inners_for_cycle` with MS's cycle-0 snapshot) — main + geometry MS
  runs (not probes: V5's scope);
- every MS proximity check: the hit with the new inners vs with the retired ones.

Pairs each main sid's MS tracker with the H1 downstream tracker (sequence EQUAL expected) and each sub's geometry
MS tracker with its projection's downstream tracker (the downstream sequence a PREFIX of MS's — the geometry runs
past the sub's end)."""
import atexit
import json
import os
import runpy
import sys

repo = os.path.abspath(sys.argv[1])
out_path = os.path.abspath(sys.argv[2])
sys.path.insert(0, repo)

import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orc  # noqa: E402
import engine_v2.structure.event_fields as ef  # noqa: E402
import engine_v2.structure.market_structure as msmod  # noqa: E402
import engine_v2.structure.unified_probe as up  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402
from engine_v2.zones.fib_tracker import FibTracker  # noqa: E402

assert os.path.normcase(msmod.__file__).startswith(os.path.normcase(repo)), msmod.__file__

CTX = []            # the context stack: "main" / "probe" / "geom" / ("proj", sub)
TRACKERS = []       # every FibTracker: (tracker, context label)
MAIN_MS = {}        # sid -> the main MS run's embedded tracker
PAIRS = []          # (sub, the downstream tracker) per projection
H1_DS = []          # the H1 downstream tracker(s)
INNERS = []         # per refresh (main + geom): label, sid, cycle, moment, new, old
PROX = []           # per proximity check (main + geom): label, cycle, i, new hit, old hit


def _ctx():
    return CTX[-1] if CTX else None


def _wrap_ctx(mod, name, label_fn):
    real = getattr(mod, name)

    def w(*a, **k):
        CTX.append(label_fn(*a, **k))
        try:
            return real(*a, **k)
        finally:
            CTX.pop()
    setattr(mod, name, w)


_wrap_ctx(orc, "compute_structure", lambda *a, **k: "main")
_wrap_ctx(up, "unified_probe", lambda *a, **k: "probe")
_wrap_ctx(edm, "_build_geometry", lambda *a, **k: "geom")
_wrap_ctx(edm, "render_sub_projection", lambda sub, *a, **k: ("proj", sub))

_real_bog = edm.build_or_get_geometry


def _bog(pool, *a, **k):
    n = len(TRACKERS)
    out = _real_bog(pool, *a, **k)
    if out is not None and out[1]:
        made = [t for t, lab in TRACKERS[n:] if lab == "geom"]
        assert len(made) == 1, f"one geometry MS per new sub, got {len(made)}"
        out[0]._s5_ms_tracker = made[0]
    return out


edm.build_or_get_geometry = _bog

_real_ft_init = FibTracker.__init__


def _ft_init(self, *a, **k):
    _real_ft_init(self, *a, **k)
    c = _ctx()
    self._s5_label = c
    self._s5_seq = []
    TRACKERS.append((self, c if not isinstance(c, tuple) else "proj"))
    if isinstance(c, tuple) and not self._quiet:
        PAIRS.append((c[1], self))
    elif c is None and not self._quiet:
        H1_DS.append(self)


FibTracker.__init__ = _ft_init


def _rep(t, sid, cyc):
    f = t.representative_fib(sid, cyc)
    if f is None:
        return None
    return [f.meta.get("version"), "cross" if f.meta.get("cross_cycle") is not None else "single",
            int(f.bos_idx), int(f.cts_idx)]


def _wrap_handler(name):
    real = getattr(FibTracker, name)

    def w(self, event, *a, **k):
        r = real(self, event, *a, **k)
        sid, cyc = int(event.meta.get("structure_id", 0)), int(event.meta.get("cycle_id", 0))
        self._s5_seq.append([event.type, sid, cyc, int(ef.stamped_idx(event)), _rep(self, sid, cyc),
                             _rep(self, sid, cyc + 1) if event.type == "CTS_THRESHOLD_UPDATED" else None])
        return r
    setattr(FibTracker, name, w)


for _h in ("on_cts_established", "on_cts_updated", "on_cts_confirmed", "on_cts_threshold_updated"):
    _wrap_handler(_h)

_real_ms_init = msmod.MarketStructure.__init__


def _ms_init(self, *a, **k):
    _real_ms_init(self, *a, **k)
    c = _ctx()
    self._s5_label = c
    if c == "main" and self._fib_tracker is not None:
        sid = int(self.state.structure_id)
        assert sid not in MAIN_MS, sid
        MAIN_MS[sid] = self._fib_tracker


msmod.MarketStructure.__init__ = _ms_init

_real_refresh = msmod.MarketStructure._refresh_poi_inners_for_cycle


def _refresh(self, moment_idx):
    _real_refresh(self, moment_idx)
    if self._s5_label not in ("main", "geom"):
        return
    st = self.state
    if st.cts is None or st.bos is None:
        old = []
    else:
        old = pz.compute_poi_inners_for_cycle(
            self._resolver_df(), int(st.bos.idx), float(st.bos.price), int(st.cts.idx), float(st.cts.price),
            int(self.struct_direction), int(st.structure_id), int(st.cts_cycle_id), float(self._fill_threshold),
            st.cycle0_data, fill_horizon_idx=int(moment_idx), snapshot_horizon_idx=int(st.cts_established_moment_idx))
    self._s5_old = [float(x) for x in old]
    INNERS.append([self._s5_label, int(st.structure_id), int(st.cts_cycle_id), int(moment_idx),
                   list(st.poi_inners_for_cycle), self._s5_old])


msmod.MarketStructure._refresh_poi_inners_for_cycle = _refresh

_real_prox = msmod.MarketStructure._check_proximity_at_candle


def _prox(self, i):
    hit = _real_prox(self, i)
    if self._s5_label in ("main", "geom"):
        st = self.state
        old_in = getattr(self, "_s5_old", None)
        hit_old = msmod._check_sd_proximity_at_candle(
            self.df, candle_idx=i, struct_direction=self.struct_direction, bos_inner=st.bos_inner_for_cycle,
            threshold=self._proximity_threshold, poi_inners=old_in or None)
        PROX.append([self._s5_label, int(st.cts_cycle_id), int(i), hit, hit_old])
    return hit


msmod.MarketStructure._check_proximity_at_candle = _prox


@atexit.register
def _report():
    res = {"main": {}, "subs": {}, "inners": {}, "prox": {}}
    assert len(H1_DS) == 1, len(H1_DS)
    ds = H1_DS[0]._s5_seq
    for sid, t in sorted(MAIN_MS.items()):
        d = [x for x in ds if x[1] == sid]
        res["main"][sid] = {"ms": len(t._s5_seq), "ds": len(d), "equal": t._s5_seq == d}
    for sub, t in PAIRS:
        ms_t = sub._s5_ms_tracker
        res["subs"][int(sub.sub_id)] = {"ms": len(ms_t._s5_seq), "ds": len(t._s5_seq),
                                        "prefix": ms_t._s5_seq[:len(t._s5_seq)] == t._s5_seq}
    diff = [x for x in INNERS if x[4] != x[5]]
    res["inners"] = {"refreshes": len(INNERS), "differ": len(diff), "rows": diff}
    flips = [x for x in PROX if (x[3] is None) != (x[4] is None) or (x[3] is not None and x[3] != x[4])]
    res["prox"] = {"checks": len(PROX), "flips": len(flips), "rows": flips}
    with open(out_path, "w") as fh:
        json.dump(res, fh, default=str, indent=1)
    print(f"[s5_agreement] main {res['main']} | subs {res['subs']} | inners {len(diff)} / {len(INNERS)} differ | "
          f"prox {len(flips)} / {len(PROX)} flips -> {out_path}")


os.makedirs("artifacts/debug", exist_ok=True)
os.makedirs("artifacts/charts", exist_ok=True)
sys.argv = ["run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__")
