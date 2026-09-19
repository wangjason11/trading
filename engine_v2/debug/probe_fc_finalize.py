"""first_confluence probe ground truth — which finalize condition fired, and where the value came from.

Reproduces the EXACT production resolution path for every first_confluence trigger on the
current replay window:
  saved H1 events -> detect_first_confluence_triggers -> to_multi_tf_trigger
  -> _resolve_first_confluence_via_unified_probe
and captures the full ProbeResult (start_idx, finalize_idx, finalize_condition, iterations)
plus every candidate H1->M15 mapping (price-mapped anchor, LOH of anchor / CTS_EST /
CTS_CONFIRMED / trigger), so a finalize value can be attributed to native-M15 vs mapped
without inference. See GOTCHAS "ProbeResult.finalize_idx Is Native-M15 OR a Mapped H1 Value".

Reads artifacts/debug/*_final.csv + the H1 *_structure_events.csv from the last replay;
fetches M15 from OANDA (~2 min). Run from the repo root:

    PYTHONPATH=. python engine_v2/debug/probe_fc_finalize.py

Added 2026-09-19 during the pool/lifecycle redesign; safe to keep as a diagnostic.
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
        cts_est[k] = int(e.idx)
    elif e.type == "CTS_CONFIRMED":
        cts_conf.setdefault(k, e)

def loh(h1_idx):
    """last M15 candle of that H1 hour"""
    return edm._map_parent_idx_to_m15_hour_end(int(h1_idx), h1, m15)

# ---- capture the ProbeResult ------------------------------------------------
captured = {}
_orig = up.unified_probe

def _wrapped(df, input_idx, direction, reference_zone, end_idx, timeframe, **kw):
    res = _orig(df, input_idx, direction, reference_zone, end_idx, timeframe, **kw)
    captured["res"] = res
    captured["probe_input_m15"] = input_idx
    captured["probe_end_m15"] = end_idx
    captured["enable_phase2"] = kw.get("enable_phase2")
    return res

up.unified_probe = _wrapped
edm.unified_probe = _wrapped  # in case of a module-level rebind

# ---- run --------------------------------------------------------------------
trigs = detect_first_confluence_triggers(events)
print(f"first_confluence triggers detected: {len(trigs)}\n")
print("=" * 132)

rows = []
for t in trigs:
    key = (t.parent_sid, t.parent_cycle_id)
    mt = to_multi_tf_trigger(t, h1)
    captured.clear()
    import contextlib, io
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink):
        out = edm._resolve_first_confluence_via_unified_probe(mt, h1, m15)
    for _line in _sink.getvalue().splitlines():
        if "unified_probe" in _line or "WARNING" in _line:
            print("   " + _line)
    res = captured.get("res")
    m15_start, validated, bos0_inner, finalize_idx = out

    cc = cts_conf.get(key)
    rows.append(dict(
        key=key,
        status=t.status,
        bos_idx_h1=t.input_idx,
        tei_h1=t.trigger_event_idx,
        cts_est_h1=cts_est.get(key),
        cts_anchor_h1=t.end_idx,
        cts_conf_h1=int(cc.idx) if cc is not None else None,
        m15_input=captured.get("probe_input_m15"),
        m15_end_TODAY=captured.get("probe_end_m15"),
        loh_cts_anchor=loh(t.end_idx) if t.end_idx is not None else None,
        loh_cts_est=loh(cts_est[key]) if key in cts_est else None,
        loh_cts_conf=loh(int(cc.idx)) if cc is not None else None,
        loh_tei=loh(t.trigger_event_idx),
        probe_start=getattr(res, "start_idx", None),
        finalize_idx=finalize_idx,
        finalize_cond=getattr(res, "finalize_condition", None),
        iters=getattr(res, "iterations", None),
        probe_status=getattr(res, "status", None),
    ))

up.unified_probe = _orig

hdr = ("cycle   st     BOS  tei  CTSest  CTSanch  CTSconf | m15_in  m15_end*  "
       "LOH(anch) LOH(est) LOH(conf) LOH(tei) | start  finalize  condition            it")
print(hdr)
print("-" * 132)
for r in rows:
    print(
        f"{str(r['key']):<7} {r['status'][:4]:<5} "
        f"{str(r['bos_idx_h1']):>5} {str(r['tei_h1']):>4} {str(r['cts_est_h1']):>6} "
        f"{str(r['cts_anchor_h1']):>8} {str(r['cts_conf_h1']):>8} | "
        f"{str(r['m15_input']):>6} {str(r['m15_end_TODAY']):>8}  "
        f"{str(r['loh_cts_anchor']):>9} {str(r['loh_cts_est']):>8} "
        f"{str(r['loh_cts_conf']):>9} {str(r['loh_tei']):>8} | "
        f"{str(r['probe_start']):>5} {str(r['finalize_idx']):>9}  "
        f"{str(r['finalize_cond']):<20} {str(r['iters']):>2}"
    )
print("=" * 132)
print("* m15_end_TODAY = production value = map_candle_to_lower_tf(cts_anchor_h1, +lower_sd)  [PRICE-mapped]")
