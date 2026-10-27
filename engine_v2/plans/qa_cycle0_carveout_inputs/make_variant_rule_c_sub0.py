"""On top of make_variant_c0.py drop + make_variant_probe_tf.py: the round-1 rule, HACKED for a measurement — the
carve-out stays on the main sid 0 (tf H1, structure_id 0) and on sub 0's geometry + its FC probe (tf M15, sd +1, the
candle at start_idx = 2025-11-21 15:30, sub 0's starting_idx 454 on the window); dropped everywhere else."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "engine_v2/structure/market_structure.py"
b = p.read_bytes().decode("utf-8"); nl = "\r\n" if "\r\n" in b else "\n"
old = ("            st.cts_phase == \"EST_OR_UPD\"\n"
       "            and cycle_gap_ok\n")
new = ("            st.cts_phase == \"EST_OR_UPD\"\n"
       "            and (st.cts_cycle_id > 0 or not self._c0_keep())\n"
       "            and cycle_gap_ok\n")
helper_anchor = "    def _maybe_confirm_cts_via_proximity(self, i: int) -> None:\n"
helper = ("    def _c0_keep(self) -> bool:\n"
          "        if self.timeframe == \"H1\":\n"
          "            return int(self.state.structure_id) == 0\n"
          "        t = str(self.df[\"time\"].iloc[self.start_idx]) if \"time\" in self.df.columns else \"\"\n"
          "        return self.struct_direction == 1 and t.startswith(\"2025-11-21 15:30\")\n\n")
for o, n in ((old, new), (helper_anchor, helper + helper_anchor)):
    o, n = o.replace("\n", nl), n.replace("\n", nl)
    assert b.count(o) == 1, (o, b.count(o))
    b = b.replace(o, n)
p.write_bytes(b.encode("utf-8")); print("patched: carve-out kept on main sid 0 + sub 0 (geometry + probe)")
