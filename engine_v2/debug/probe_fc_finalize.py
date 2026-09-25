"""first_confluence probe ground truth — which finalize condition fired, and where the value came from.

Reproduces the EXACT production resolution path for every first_confluence trigger on the
current replay window:
  saved H1 events -> detect_first_confluence_triggers -> to_multi_tf_trigger
  -> _resolve_first_confluence_via_unified_probe
and captures the full ProbeResult (starting_idx, finalize_idx, finalize_condition, iterations)
plus every candidate H1->M15 mapping (price-mapped anchor, LOH of anchor / CTS_EST /
CTS_CONFIRMED / trigger), so a finalize value can be attributed to native-M15 vs mapped
without inference. See GOTCHAS "ProbeResult.finalize_idx Is Native-M15 OR a Mapped H1 Value".

Also retains EVERY Phase-2 MarketStructure the probe constructs (one per Phase-2 iteration)
and reports, per cycle, `n_ms` (Phase-2 MS runs), `max_ev` = max(ev.idx) over ALL of them
and `leak = max_ev - m15_end` (> 0 = an event stamped past the probe's bound — the MS bounds
leak Plan A fixed; the last iteration alone would hide earlier leaks). MS debug lines other
than the per-anchor `[POST_STEP]` flood (`[RV_EXPIRE]`, `[REWIND]`, `[RV_SCHEDULE]`, ...) are
printed so a delta in a probe row can be named by mechanism; `[POST_STEP]` is reported as a
count. A post-Plan-A `AssertionError` from MS (event past `effective_end`) is caught per
cycle and shown in the row instead of aborting the other cycles.

Reads artifacts/debug/*_final.csv + the H1 *_structure_events.csv from the last replay;
fetches M15 from OANDA (~10 s). Run from the repo root:

    PYTHONPATH=. python engine_v2/debug/probe_fc_finalize.py

Added 2026-09-19 during the pool/lifecycle redesign (Phase-2 MS retention + leak column
added for Plan A the same day; per-iteration `early_stop_idx` + `cts1_anchor`/`cts1_moment`
added for Plan B 2026-09-20; `cts1_ext` until Plan E E2d); safe to keep as a diagnostic.
"""
import ast
import csv
import glob

import pandas as pd

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.multitf.first_confluence_trigger import detect_first_confluence_triggers
from engine_v2.multitf.first_confluence_pipeline import to_multi_tf_trigger
from engine_v2.multitf.data_bridge import (
    fetch_lower_tf_data, prepare_lower_tf_data, map_candle_to_lower_tf,
)
import engine_v2.structure.unified_probe as up
import engine_v2.multitf.entity_df_mutation as edm
from engine_v2.structure import event_fields as ef

# ---- H1 parent frame + events (from the saved replay) -----------------------
final_path = glob.glob("artifacts/debug/*_final.csv")[0]
h1 = pd.read_csv(final_path)
h1 = h1.reset_index(drop=True)
h1["time"] = pd.to_datetime(h1["time"], utc=True)
h1.attrs["pair"] = "NZD_USD"
print(f"H1 frame: {len(h1)} rows, {h1['time'].iloc[0]} .. {h1['time'].iloc[-1]}")

ev_path = [p for p in glob.glob("artifacts/debug/*_structure_events.csv")
           if "M15" not in p][0]
print(f"H1 events file: {ev_path}")
events = []
with open(ev_path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        meta = ast.literal_eval(r["meta"]) if r.get("meta") else {}
        price = r.get("price")
        events.append(StructureEvent(
            idx=int(r["idx"]), category=r["category"], type=r["type"],
            price=float(price) if price not in (None, "", "nan") else None,
            meta=meta,
        ))
print(f"H1 events: {len(events)}")

# ---- M15 entity frame -------------------------------------------------------
m15_raw = fetch_lower_tf_data("NZD_USD", "M15", h1["time"].iloc[0], h1["time"].iloc[-1])
m15 = prepare_lower_tf_data(m15_raw.copy())
m15.attrs["pair"] = "NZD_USD"
print(f"M15 frame: {len(m15)} rows\n")

# ---- lookups for the alternative mappings -----------------------------------
cts_est = {}
cts_conf = {}
for e in events:
    k = (int(e.meta.get("structure_id", -1)), int(e.meta.get("cycle_id", -1)))
    if e.type == "CTS_ESTABLISHED":
        cts_est[k] = ef.cts_anchor_idx(e)
    elif e.type == "CTS_CONFIRMED":
        cts_conf.setdefault(k, e)

def loh(h1_idx):
    """last M15 candle of that H1 hour"""
    return edm._map_parent_idx_to_m15_hour_end(int(h1_idx), h1, m15)

# ---- capture the ProbeResult ------------------------------------------------
captured = {}
_orig = up.unified_probe

def _wrapped(df, input_idx, direction, reference_zone, probe_end_idx, timeframe, **kw):
    res = _orig(df, input_idx, direction, reference_zone, probe_end_idx, timeframe, **kw)
    captured["res"] = res
    captured["probe_input_m15"] = input_idx
    captured["probe_end_m15"] = probe_end_idx
    captured["enable_phase2"] = kw.get("enable_phase2")
    return res

up.unified_probe = _wrapped
edm.unified_probe = _wrapped  # in case of a module-level rebind

# Retain every Phase-2 MarketStructure. `_run_phase2` looks `_make_market_structure`
# up in unified_probe's module globals at call time, so patch THAT name (rebinding
# structure_engine._make_market_structure would not reach Phase 2).
retained_ms = []
_orig_mms = up._make_market_structure

def _mms_wrapped(df, struct_direction, **kw):
    ms = _orig_mms(df, struct_direction, **kw)
    retained_ms.append(ms)
    return ms

up._make_market_structure = _mms_wrapped

# ---- run --------------------------------------------------------------------
trigs = detect_first_confluence_triggers(events)
print(f"first_confluence triggers detected: {len(trigs)}\n")
print("=" * 132)

rows = []
for t in trigs:
    key = (t.parent_sid, t.parent_cycle_id)
    mt = to_multi_tf_trigger(t, h1)
    captured.clear()
    retained_ms.clear()
    import contextlib, io
    _sink = io.StringIO()
    ms_error = None
    with contextlib.redirect_stdout(_sink):
        try:
            # pool=None → no probe cache (each cycle probes fresh, as the diagnostic wants)
            out = edm._resolve_first_confluence_via_unified_probe(mt, h1, m15, pool=None)
        except AssertionError as exc:  # Plan A post-run assert: show it, keep going
            ms_error = str(exc)
            out = None
    print(f"--- cycle {key}")
    n_post_step = 0
    for _line in _sink.getvalue().splitlines():
        if _line.startswith("[POST_STEP]"):
            n_post_step += 1
            continue
        print("   " + _line)
    if n_post_step:
        print(f"   ([POST_STEP] lines: {n_post_step})")
    if ms_error:
        print(f"   ASSERTION: {ms_error}")
    # Per-iteration Phase-2 MS report: bound, event count, max event idx, events past the
    # bound, and (Plan B) the early stop: `early_stop_idx` (None = ran to the bound) plus the
    # 2nd CTS_ESTABLISHED's anchor (`cts_anchor_idx`, the extreme) vs moment (`confirmed_at`) — the finalize value
    # is the moment; a difference here would move the row's finalize_idx.
    for k, ms in enumerate(retained_ms, 1):
        mx = max((int(ev.idx) for ev in ms.events), default=None)
        past = [(ev.type, int(ev.idx)) for ev in ms.events if int(ev.idx) > int(ms.end_idx)]
        _est = sorted((ev for ev in ms.events if ev.type == "CTS_ESTABLISHED"), key=ef.cts_anchor_idx)
        _c1 = _est[1] if len(_est) >= 2 else None
        print(f"   [phase2 ms {k}] start={ms.start_idx} end_idx={ms.end_idx} "
              f"n_ev={len(ms.events)} max_ev_idx={mx} past_bound={past} "
              f"early_stop_idx={getattr(ms, 'early_stop_idx', None)} "
              f"cts1_anchor={None if _c1 is None else ef.cts_anchor_idx(_c1)} "
              f"cts1_moment={None if _c1 is None else ef.event_moment(_c1)}")
    max_ev_all = max((int(ev.idx) for ms in retained_ms for ev in ms.events), default=None)
    res = captured.get("res")
    if isinstance(out, edm.ResolvedStart):
        m15_start, validated, bos0_inner, finalize_idx = (
            out.starting_idx, out.validated_parent_idx, out.bos0_inner, out.finalize_idx,
        )
    else:  # ProbeFailure / None
        m15_start, validated, bos0_inner, finalize_idx = None, None, None, None

    cc = cts_conf.get(key)
    rows.append(dict(
        key=key,
        status=t.status,
        bos_idx_h1=t.input_idx,
        tei_h1=t.trigger_event_idx,
        cts_est_h1=cts_est.get(key),
        cts_anchor_h1=t.probe_end_idx,
        cts_conf_h1=int(cc.idx) if cc is not None else None,
        m15_input=captured.get("probe_input_m15"),
        m15_end_TODAY=captured.get("probe_end_m15"),
        loh_cts_anchor=loh(t.probe_end_idx) if t.probe_end_idx is not None else None,
        loh_cts_est=loh(cts_est[key]) if key in cts_est else None,
        loh_cts_conf=loh(int(cc.idx)) if cc is not None else None,
        loh_tei=loh(t.trigger_event_idx),
        probe_start=getattr(res, "starting_idx", None),
        finalize_idx=finalize_idx,
        finalize_cond=getattr(res, "finalize_condition", None),
        iters=getattr(res, "iterations", None),
        probe_status=getattr(res, "status", None),
        n_ms=len(retained_ms),
        max_ev=max_ev_all,
        leak=(None if max_ev_all is None or captured.get("probe_end_m15") is None
              else int(max_ev_all) - int(captured["probe_end_m15"])),
        error=ms_error,
    ))

up.unified_probe = _orig
up._make_market_structure = _orig_mms

hdr = ("cycle   st     BOS  tei  CTSest  CTSanch  CTSconf | m15_in  m15_end*  "
       "LOH(anch) LOH(est) LOH(conf) LOH(tei) | start  finalize  condition            it | n_ms  max_ev  leak")
print(hdr)
print("-" * 156)
for r in rows:
    print(
        f"{str(r['key']):<7} {str(r['status'])[:4]:<5} "
        f"{str(r['bos_idx_h1']):>5} {str(r['tei_h1']):>4} {str(r['cts_est_h1']):>6} "
        f"{str(r['cts_anchor_h1']):>8} {str(r['cts_conf_h1']):>8} | "
        f"{str(r['m15_input']):>6} {str(r['m15_end_TODAY']):>8}  "
        f"{str(r['loh_cts_anchor']):>9} {str(r['loh_cts_est']):>8} "
        f"{str(r['loh_cts_conf']):>9} {str(r['loh_tei']):>8} | "
        f"{str(r['probe_start']):>5} {str(r['finalize_idx']):>9}  "
        f"{str(r['finalize_cond']):<20} {str(r['iters']):>2} | "
        f"{str(r['n_ms']):>4} {str(r['max_ev']):>7} {str(r['leak']):>5}"
        + (f"   ASSERTION: {r['error']}" if r.get('error') else "")
    )
print("=" * 156)
print("* m15_end_TODAY = production value = map_candle_to_lower_tf(cts_anchor_h1, +lower_sd)  [PRICE-mapped]")
print("  n_ms = Phase-2 MS runs for the cycle; max_ev = max(ev.idx) over ALL of them; leak = max_ev - m15_end (>0 = event past the bound)")
