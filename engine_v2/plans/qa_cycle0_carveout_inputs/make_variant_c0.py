"""Patch a `git archive` copy of the tree (argv[1]) for the cycle-0 carve-out revisit (SESSION ORDER item 5b).

mode (argv[2]):
  shadow - behaviour UNCHANGED; logs every cycle-0 candle on which the proximity check WOULD hit
           (every other gate of `_maybe_confirm_cts_via_proximity` evaluated and printed, incl. the gap)
  drop   - the carve-out removed (`st.cts_cycle_id > 0` dropped from the gate) + the same log

    git archive HEAD | tar -x -C <short %TEMP% dir>; python <this> <dir> shadow|drop
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
MODE = sys.argv[2]
assert MODE in ("shadow", "drop"), MODE

rel = "engine_v2/structure/market_structure.py"
p = ROOT / rel
b = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in b else "\n"

old_gate = (
    "        if not (\n"
    "            st.cts_phase == \"EST_OR_UPD\"\n"
    "            and st.cts_cycle_id > 0\n"
    "            and cycle_gap_ok\n"
    "            and st.bos_inner_for_cycle is not None\n"
    "            and i > st.cts.idx\n"
    "        ):\n"
    "            return\n"
)
log = (
    "        if (st.cts_phase == \"EST_OR_UPD\" and st.cts_cycle_id == 0\n"
    "                and st.bos_inner_for_cycle is not None and i > st.cts.idx):\n"
    "            _hit = self._check_proximity_at_candle(i)\n"
    "            if _hit is not None:\n"
    "                _t = self.df[\"time\"].iloc[i] if \"time\" in self.df.columns else None\n"
    "                print(f\"[c0probe] tf={self.timeframe} sid={st.structure_id} sd={self.struct_direction} \"\n"
    "                      f\"start={self.start_idx} probe={self.stop_after_cts_established} n={len(self.df)} i={i} t={_t} \"\n"
    "                      f\"cts_idx={st.cts.idx} cts={float(st.cts.price):.5f} bos_th={float(st.bos_threshold):.5f} \"\n"
    "                      f\"gap_pips={abs(float(st.cts.price) - float(st.bos_threshold)) / self.pip_size:.1f} \"\n"
    "                      f\"gap_ok={cycle_gap_ok} bos_inner={float(st.bos_inner_for_cycle):.5f} \"\n"
    "                      f\"poi_inners={list(st.poi_inners_for_cycle or [])} inner={float(_hit[0]):.5f} kind={_hit[1]}\")\n"
)
new_gate = old_gate if MODE == "shadow" else old_gate.replace("            and st.cts_cycle_id > 0\n", "")
old_n, new_n = old_gate.replace("\n", nl), (log + new_gate).replace("\n", nl)
assert b.count(old_n) == 1, b.count(old_n)
b = b.replace(old_n, new_n)
p.write_bytes(b.encode("utf-8"))
print(f"patched {rel}: mode={MODE}")
