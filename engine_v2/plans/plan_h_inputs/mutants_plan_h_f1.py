"""PLAN_H §8.9 F1 (the fib chart aid) — mutants of the shared fib draw gate / hover / drawer and their call sites
(run: `review_scripts/mutant_runner.py`, full suite, from the repo root).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_h_inputs/mutants_plan_h_f1.py --out <short dir>

Each mutant bends ONE rule or call site of the aid. `old` is matched against the file read in TEXT mode (LF line ends).
"""
ZR = "engine_v2/charting/_zone_render.py"
H1C = "engine_v2/charting/export_plotly.py"
M15C = "engine_v2/charting/export_m15_chart.py"

MUT = [
    # --- the gate ---------------------------------------------------------------------------------------------------
    ("F01 collapsed cycles drawn", ZR,
     "    if (int(fib_st.structure_id), int(fib_st.cycle_id)) in collapsed:\n        return False\n",
     "    if (int(fib_st.structure_id), int(fib_st.cycle_id)) in collapsed:\n        pass\n"),
    ("F02 ended never-locked vanish (the §9.1 term back)", ZR,
     "    return bool(fib_st.locked or fib_st.active)\n",
     "    return bool(fib_st.locked or (fib_st.active and fib_st.status != \"ended\"))\n"),
    ("F03 condition-inactive drawn", ZR,
     "    return bool(fib_st.locked or fib_st.active)\n",
     "    return True\n"),
    ("F04 disappeared drawn", ZR,
     "    if getattr(fib_st, \"status\", \"active\") == \"disappeared\":\n        return False\n",
     "    if getattr(fib_st, \"status\", \"active\") == \"disappeared\":\n        pass\n"),
    ("F05 bright ignores locked", ZR,
     "    return bool(fib_st.active and not fib_st.locked\n",
     "    return bool(fib_st.active\n"),
    # --- the IP lookup + hover cell -----------------------------------------------------------------------------------
    ("F06 lookup keeps losing candidates", ZR,
     "        if not (r.picked or r.ic_idx is None):\n            continue\n",
     "        if not (r.picked or r.ic_idx is None):\n            pass\n"),
    ("F07 lookup ignores sub_id", ZR,
     "        if by_sub:\n            key = (int(r.meta[\"sub_id\"]),) + key\n",
     "        if False:\n            key = (int(r.meta[\"sub_id\"]),) + key\n"),
    ("F08 hover: no-pattern reads as not evaluated", ZR,
     "    elif ip_row.ic_idx is None:\n",
     "    elif False:\n"),
    ("F09 hover: life start/end swapped", ZR,
     "    life = f\"life=[{start}, {end}]\"",
     "    life = f\"life=[{end}, {start}]\""),
    # --- the drawer ----------------------------------------------------------------------------------------------------
    ("F10 drawer: tiers inverted", H1C,
     "    tier = \"active\" if fib_is_bright(fib_st) else \"historical\"\n",
     "    tier = \"historical\" if fib_is_bright(fib_st) else \"active\"\n"),
    ("F11 drawer: shapes outside the legend group", H1C,
     "    group = {\"legendgroup\": legendgroup} if legendgroup is not None else {}\n",
     "    group = {}\n"),
    # --- the H1 call site ----------------------------------------------------------------------------------------------
    ("F12 H1: no collapsed set", H1C,
     "        _collapsed = collapsed_cycles(dfx.attrs.get(\"kl_zones\", []))\n"
     "        _ip_rows = ip_rows_by_fib_record(",
     "        _collapsed = set()\n"
     "        _ip_rows = ip_rows_by_fib_record("),
    ("F13 H1: lookup key anchors swapped", H1C,
     "                    int(fib_st.bos_idx), int(fib_st.cts_idx)))),\n",
     "                    int(fib_st.cts_idx), int(fib_st.bos_idx)))),\n"),
    # --- the M15 call site ---------------------------------------------------------------------------------------------
    ("F14 M15: no collapsed set", M15C,
     "                if fib_st.fib is None or not fib_record_drawn(fib_st, _collapsed_sub_cycles):\n",
     "                if fib_st.fib is None or not fib_record_drawn(fib_st, set()):\n"),
    ("F15 M15: lookup without the sub", M15C,
     "                    (eid, int(fib_st.structure_id), int(fib_st.cycle_id), bos_idx, cts_idx))\n",
     "                    (5, int(fib_st.structure_id), int(fib_st.cycle_id), bos_idx, cts_idx))\n"),
    ("F16 M15: every box a legend entry", M15C,
     "                    legend_name=None if fib_drawn else \"M15 fib\")\n",
     "                    legend_name=\"M15 fib\")\n"),
    ("F17 M15: hover outside the legend group", M15C,
     "                    mode=\"lines\", name=\"M15 fib\", showlegend=False, legendgroup=\"m15_fib\",\n",
     "                    mode=\"lines\", name=\"M15 fib\", showlegend=False,\n"),
    ("F18 M15: the toggle ignored", M15C,
     "        if fib_cfg.get(\"lines\", True) and sid_fibs:\n",
     "        if sid_fibs:\n"),
    ("F19 M15: grouped by the wrong key", M15C,
     "    for fs in all_fib_states:\n        ident = _sub_identity(fs.meta)\n",
     "    for fs in all_fib_states:\n        ident = 5\n"),
]
