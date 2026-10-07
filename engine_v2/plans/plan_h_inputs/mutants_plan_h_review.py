"""PLAN_H §8.5c — the landing review of B (e54f509) + F1 (73e82ed) + F1·1 (811e742), 2026-10-06c: the mutation
lens's 26 NEW mutants + 1 control (C00, a comment only — MUST SURVIVE), and two of mine for the review's own pins
(X01 / X02: a RuntimeError scope on an IP path — the C1 guard + the C2 Phase-2 pin). The two older files
(`mutants_plan_h_b.py` 32, `mutants_plan_h_f1.py` 19) cover the rest. First run (the lens, before the fix-up): 11
KILLED / 15 SURVIVED; after the fix-up: see PLAN_H §8.5c.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_h_inputs/mutants_plan_h_review.py --out <short dir>

`old` matched in TEXT mode (LF).
"""
PZ = "engine_v2/zones/poi_zones.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
ZR = "engine_v2/charting/_zone_render.py"
H1C = "engine_v2/charting/export_plotly.py"
M15C = "engine_v2/charting/export_m15_chart.py"
UP = "engine_v2/structure/unified_probe.py"

_POI_TPL_HEAD = (
    '                        "side=%{customdata[0]}<br>"\n'
    '                        "structure_id=%{customdata[1]}<br>"\n'
    '                        "struct_direction=%{customdata[2]}<br>"\n'
    '                        "ic_idx=%{customdata[3]}<br>"\n'
    '                        "confirmed_idx=%{customdata[4]}<br>"\n'
    '                        "cycle_id=%{customdata[5]}<br>"\n'
)

MUT = [
    # --- control: a comment only — MUST SURVIVE ------------------------------------------------------------------
    ("C00 control (comment only)", PZ,
     "    # The pick — OUTSIDE the try (fail loudly; MS's own fill threshold).\n",
     "    # The pick — OUTSIDE the try (control mutant: comment text only).\n"),

    # --- the resolver's fail-loud scope (PR1) --------------------------------------------------------------------
    ("R01 resolver: chaining dropped", PZ,
     "f\"fib {fib_state.bos_idx}..{fib_state.cts_idx}): {exc!r}\") from exc\n",
     "f\"fib {fib_state.bos_idx}..{fib_state.cts_idx}): {exc!r}\") from None\n"),
    ("R02 resolver: narrow except (KeyError / AssertionError only)", PZ,
     "    except Exception as exc:\n        raise IPSelectionError(",
     "    except (KeyError, AssertionError) as exc:\n        raise IPSelectionError("),
    ("R03 IPSelectionError is an IndexError", PZ,
     "class IPSelectionError(RuntimeError):",
     "class IPSelectionError(IndexError):"),
    ("R04 resolver: the anchor try no longer graceful", PZ,
     "    except Exception:\n        return []\n\n    # The pick",
     "    except Exception:\n        raise\n\n    # The pick"),

    # --- select_ip_records: the gate order / the gate's terms ----------------------------------------------------
    ("G01 record gate: ended never-locked records pass", PZ,
     "    return bool((fib_state.active and fib_state.end_idx is None) or fib_state.locked)",
     "    return bool(fib_state.active or fib_state.locked)"),
    ("G02 records: selection BEFORE the CTS_ESTABLISHED skip / assert", PZ,
     "        key = (int(fib_state.structure_id), int(fib_state.cycle_id))\n"
     "        cts_event = cts_established_by_key.get(key)\n",
     "        key = (int(fib_state.structure_id), int(fib_state.cycle_id))\n"
     "        _early = select_ip_ic(df, fib_state, config, classify_misses=True)\n"
     "        cts_event = cts_established_by_key.get(key)\n"),

    # --- the shared list: each consumer ignores the handed records ---------------------------------------------
    ("D01 derive_ip_candidates re-selects (ignores ip_records)", PZ,
     "    if ip_records is None:\n        ip_records = select_ip_records(df, structure_events, fib_tracker, config)\n"
     "    out: List[IPCandidateRecord] = []\n",
     "    if True:\n        ip_records = select_ip_records(df, structure_events, fib_tracker, config)\n"
     "    out: List[IPCandidateRecord] = []\n"),
    ("D02 derive_poi_zones re-selects (ignores ip_records)", PZ,
     "    if ip_records is None:\n        ip_records = select_ip_records(df, structure_events, fib_tracker, config)\n\n",
     "    if True:\n        ip_records = select_ip_records(df, structure_events, fib_tracker, config)\n\n"),
    ("O01 orchestrator: the CSV gets no shared list", ORCH,
     "structure_path_id=structure_path_id, ip_records=ip_records,\n",
     "structure_path_id=structure_path_id,\n"),

    # --- the activation sweep / fib_zone_bounds ------------------------------------------------------------------
    ("A01 sweep: sell as-of anchors swapped", PZ,
     "        else:\n            anchor_high = bos_price\n            anchor_low = cts_price_at_t\n",
     "        else:\n            anchor_high = cts_price_at_t\n            anchor_low = bos_price\n"),
    ("A02 fib_zone_bounds: the 61.8 edge at 50", PZ,
     "    a = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_min, sd)\n",
     "    a = calculate_fib_price(anchor_high, anchor_low, 50.0, sd)\n"),

    # --- the mirror: each remaining IP meta key dropped ----------------------------------------------------------
    ("E01 mirror: pattern_anchor_idx not shifted", EDM,
     '    "pattern_anchor_idx",   # POI (the initial leg',
     '    # "pattern_anchor_idx",   # POI (the initial leg'),
    ("E02 mirror: opp_leg_first_idx not shifted", EDM,
     '    "opp_leg_first_idx",    # POI\n',
     '    # "opp_leg_first_idx",    # POI\n'),
    ("E03 mirror: pattern_end_idx not shifted", EDM,
     '    "pattern_end_idx",      # POI (the continuation',
     '    # "pattern_end_idx",      # POI (the continuation'),
    ("E04 mirror: ip_tb_imbalance_first_c2_idx not shifted", EDM,
     '    "ip_tb_imbalance_first_c2_idx",  # POI (TB2 only)',
     '    # "ip_tb_imbalance_first_c2_idx",  # POI (TB2 only)'),

    # --- the hover builder ----------------------------------------------------------------------------------------
    ("H01 hover: legs count displayed reversed", ZR,
     "    text = (f\"IP {meta['ip_rank']} · legs {n_init}-{n_opp}-{n_cont} \"",
     "    text = (f\"IP {meta['ip_rank']} · legs {n_cont}-{n_opp}-{n_init} \""),

    # --- the 3 POI hover sites: the template's IP cell -----------------------------------------------------------
    ("P01 H1 POI hover (top line): template reads cycle_id for the IP cell", H1C,
     '                    y=[y1] * len(seg_times),\n'
     '                    mode="lines",\n'
     '                    name=zone_name,\n'
     '                    showlegend=False,\n'
     '                    line=poi_hover_line,\n'
     '                    line_shape="hv",\n'
     '                    hovertemplate=(\n'
     '                        "TF=1H<br>"\n'
     '                        "<b>POI Zone</b><br>"\n' + _POI_TPL_HEAD +
     '                        "%{customdata[6]}<br>"\n',
     '                    y=[y1] * len(seg_times),\n'
     '                    mode="lines",\n'
     '                    name=zone_name,\n'
     '                    showlegend=False,\n'
     '                    line=poi_hover_line,\n'
     '                    line_shape="hv",\n'
     '                    hovertemplate=(\n'
     '                        "TF=1H<br>"\n'
     '                        "<b>POI Zone</b><br>"\n' + _POI_TPL_HEAD +
     '                        "%{customdata[5]}<br>"\n'),
    ("M01 M15 sub-POI hover: template reads cycle_id for the IP cell", M15C,
     '                            "confirmed_idx=%{customdata[4]}<br>"\n'
     '                            "%{customdata[6]}<br>"\n',
     '                            "confirmed_idx=%{customdata[4]}<br>"\n'
     '                            "%{customdata[5]}<br>"\n'),
    ("M02 M15 sub POIs never drawn", M15C,
     '        if zone_cfg.get("POI", False) and sid_pois:\n',
     '        if False and sid_pois:\n'),
    ("M09 M15 H1-overlay POI hover: template reads cycle_id for the IP cell", M15C,
     '                    mode="lines", name="H1 POI zone (overlay)", showlegend=False,\n'
     '                    line=h1_hover_line, line_shape="hv",\n'
     '                    hovertemplate=(\n'
     '                        "TF=1H<br>"\n'
     '                        "<b>POI Zone</b><br>"\n' + _POI_TPL_HEAD +
     '                        "%{customdata[6]}<br>"\n',
     '                    mode="lines", name="H1 POI zone (overlay)", showlegend=False,\n'
     '                    line=h1_hover_line, line_shape="hv",\n'
     '                    hovertemplate=(\n'
     '                        "TF=1H<br>"\n'
     '                        "<b>POI Zone</b><br>"\n' + _POI_TPL_HEAD +
     '                        "%{customdata[5]}<br>"\n'),

    # --- the fib drawer / hovers (F1, F1.1) ----------------------------------------------------------------------
    ("P02 H1 fib hover: 61.8%% back (F1.1 revert)", H1C,
     '                        "61.8%=%{customdata[6]}<br>"\n',
     '                        "61.8%%=%{customdata[6]}<br>"\n'),
    ("M03 M15 fib hover: 61.8%% back (F1.1 revert)", M15C,
     '                        "61.8%=%{customdata[10]}<br>"\n',
     '                        "61.8%%=%{customdata[10]}<br>"\n'),
    ("P03 drawer: the 61.8 level drawn at 50", H1C,
     "    price_618 = fib.price_at_pct(61.8)\n",
     "    price_618 = fib.price_at_pct(50.0)\n"),
    ("P04 drawer: the zone rect always bright", H1C,
     '    zone_style = _style(f"fib.zone_rect.{tier}")\n',
     '    zone_style = _style("fib.zone_rect.active")\n'),
    ("M04 M15 fib hover: 61.8 / 80 cells swapped", M15C,
     '                    f"{price_618:.5f}", f"{price_80:.5f}",\n'
     '                    fib_hover_status_text(fib_st, ip_row),\n',
     '                    f"{price_80:.5f}", f"{price_618:.5f}",\n'
     '                    fib_hover_status_text(fib_st, ip_row),\n'),
    ("M05 M15 fib hover: parent sid / cycle cells swapped", M15C,
     "                    eid, p_sid, p_cycle, sub_window_str, sub_records_str, fib_st.cycle_id,\n",
     "                    eid, p_cycle, p_sid, sub_window_str, sub_records_str, fib_st.cycle_id,\n"),
    # --- the review's own pins (mine, 2026-10-06c) -------------------------------------------------------------------
    ("X01 sub geometry scope widened to RuntimeError", EDM,
     "except (ValueError, IndexError)",
     "except (ValueError, IndexError, RuntimeError)"),
    ("X02 probe Phase 2 swallows a RuntimeError", UP,
     "        df_probe, probe_events, _ = ms.run()\n",
     "        try:\n            df_probe, probe_events, _ = ms.run()\n        except RuntimeError:\n            return None\n"),
]
