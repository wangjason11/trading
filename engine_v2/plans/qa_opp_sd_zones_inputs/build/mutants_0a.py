EDM = "engine_v2/multitf/entity_df_mutation.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
MUT = [
    # PLAN_R §2's list
    ("0a-1 drop the max", EDM,
     "            lo = max(int(mapped), lo)",
     "            lo = int(mapped)"),
    ("0a-2 floor on first_counter only", EDM,
     "    lo = int(parent_floor_idx)\n    if prior_parent_idx is not None:\n        mapped = _map_parent_idx_to_m15_hour_end(\n            int(prior_parent_idx), parent_df, m15_df,\n        )\n        if mapped is not None:\n            lo = max(int(mapped), lo)",
     "    lo = int(parent_floor_idx) if uc == \"first_counter\" else 0\n    if prior_parent_idx is not None:\n        mapped = _map_parent_idx_to_m15_hour_end(\n            int(prior_parent_idx), parent_df, m15_df,\n        )\n        if mapped is not None:\n            lo = int(mapped)"),
    ("0a-3a closure passes trigger_idx", ORCH,
     "parent_floor_idx=tables.floor(*t.cycle)",
     "parent_floor_idx=t.trigger_idx"),
    ("0a-3b resolver passes hi as the floor", EDM,
     "        trigger, parent_df, m15_df, hi, parent_floor_idx=int(parent_floor_idx),",
     "        trigger, parent_df, m15_df, hi, parent_floor_idx=int(hi),"),
    ("0a-4 closure passes 0", ORCH,
     "parent_floor_idx=tables.floor(*t.cycle)",
     "parent_floor_idx=0"),
    # extra
    ("0a-5 dispatcher forwards 0", EDM,
     "            parent_floor_idx=int(parent_floor_idx), parent_path=parent_path,",
     "            parent_floor_idx=0, parent_path=parent_path,"),
    ("0a-6 missing meta degrades to 0", EDM,
     "    lo = int(parent_floor_idx)\n",
     "    lo = 0\n"),
]
