"""Plan E E4c pre-review mutation loop (the 14 pattern-path CTS_UPDATED location readers; PLAN_E §8.1). Kept since
2026-09-25 as the template for any stage that moves an index role.

    python engine_v2/plans/plan_e_inputs/review_scripts/mut_loc_e4c.py <REPO_ROOT> [SITE ...]

Copies REPO_ROOT/engine_v2 (+ oanda.cfg + the Plan-B H1 events CSV a test needs) to
%TEMP%/pe/e4c_mut/repo, applies each mutant ALONE to a fresh copy, runs the full
suite with sys.executable (never the bare "python": another interpreter without pytest
is on PATH) and prints KILLED / SURVIVED / ERROR (a non-zero rc without FAILED lines).
The SITES needles are exact source snippets at commit c182eb6 — re-check them before reuse
(`mutate` asserts each needle occurs). Original docstring:

E4c pre-review mutation loop: at every CTS LOCATION read that can see a
pattern-path CTS_UPDATED, read the raw `ev.idx` (the MOMENT since E4c) for
CTS_UPDATED only. Each mutant on a fresh copy of the working tree; full suite.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
ROOT = Path(os.environ["TEMP"]) / "pe" / "e4c_mut"
BASE = ROOT / "repo"
RES = ROOT / "results"


def raw_if_upd(x):
    return f'(int({x}.idx) if {x}.type == "CTS_UPDATED" else ef.cts_anchor_idx({x}))'


KEY = 'key=lambda _e: (int(_e.idx) if _e.type == "CTS_UPDATED" else ef.cts_anchor_idx(_e))'
# (name, file, needle, occurrence, replacement)
SITES = [
    ("U1_m15_unconf", "charting/export_m15_chart.py",
     "and ef.cts_anchor_idx(e) > last_slice_idx", 0, f"and {raw_if_upd('e')} > last_slice_idx"),
    ("U2_m15_unconf_pt", "charting/export_m15_chart.py",
     "latest_idx = ef.cts_anchor_idx(latest)", 0, f"latest_idx = {raw_if_upd('latest')}"),
    ("U3_ovl_unconf_pt", "charting/export_m15_chart.py",
     "latest_idx = ef.cts_anchor_idx(latest)", 1, f"latest_idx = {raw_if_upd('latest')}"),
    ("U4_h1_unconf_pt", "charting/export_plotly.py",
     "cts_idx = ef.cts_anchor_idx(latest_cts)", 0, f"cts_idx = {raw_if_upd('latest_cts')}"),
    ("U5_prevbos_end", "pipeline/orchestrator.py",
     "end_idx = ef.cts_anchor_idx(min(qualifying, key=ef.event_moment))", 0,
     "end_idx = (lambda _w: " + raw_if_upd("_w") + ")(min(qualifying, key=ef.event_moment))"),
    ("U6_levels", "structure/market_structure.py",
     "time=t.iloc[ef.cts_anchor_idx(ev)]", 0, f"time=t.iloc[{raw_if_upd('ev')}]"),
    ("U7_refzone", "structure/reference_zone.py",
     "    cts_anchor_idx = ef.cts_anchor_idx(ev)", 0, f"    cts_anchor_idx = {raw_if_upd('ev')}"),
    ("U8_fib_upd", "zones/fib_tracker.py",
     "        cts_idx = ef.cts_anchor_idx(event)\n        cts_price = float(event.price) if event.price else 0.0", 0,
     f"        cts_idx = {raw_if_upd('event')}\n        cts_price = float(event.price) if event.price else 0.0"),
    ("U9_poi_pre", "zones/poi_zones.py",
     "cts_anchor_idx_at_t = ef.cts_anchor_idx(ev)", 0, f"cts_anchor_idx_at_t = {raw_if_upd('ev')}"),
    ("U10_poi_in", "zones/poi_zones.py",
     "cts_anchor_idx_at_t = ef.cts_anchor_idx(payload)", 0, f"cts_anchor_idx_at_t = {raw_if_upd('payload')}"),
    ("U11_wave_sort", "zones/wave_candles.py",
     "ordered_events.sort(key=ef.cts_anchor_idx)", 0, f"ordered_events.sort({KEY})"),
    ("U12_wave_ev", "zones/wave_candles.py",
     "        ev_idx = ef.cts_anchor_idx(ev)", 0, f"        ev_idx = {raw_if_upd('ev')}"),
    ("U13_wave_next", "zones/wave_candles.py",
     "next_ev_idx = ef.cts_anchor_idx(ordered_events[ei + 1])", 0,
     "next_ev_idx = (lambda _w: " + raw_if_upd("_w") + ")(ordered_events[ei + 1])"),
    ("U14_stamped", "structure/event_fields.py",
     '    if ev.type == "CTS_UPDATED":\n        return cts_anchor_idx(ev)\n    return int(ev.idx)', 0,
     '    if ev.type == "CTS_UPDATED":\n        return int(ev.idx)\n    return int(ev.idx)'),
]


def mutate(dst: Path, rel: str, needle: str, occ: int, new: str) -> None:
    p = dst / "engine_v2" / rel
    src = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in src
    s = src.replace("\r\n", "\n")
    parts = s.split(needle)
    assert len(parts) >= occ + 2, (rel, needle[:50], len(parts) - 1)
    s = needle.join(parts[:occ + 1]) + new + needle.join(parts[occ + 1:])
    if crlf:
        s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8"))


def run(site) -> dict:
    name, rel, needle, occ, new = site
    dst = ROOT / f"w_{name}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    mutate(dst, rel, needle, occ, new)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", "--tb=no"]
    proc = subprocess.run(cmd, cwd=dst, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout + "\n" + proc.stderr
    (RES / f"{name}.txt").write_text(out, encoding="utf-8")
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)
    tail = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    verdict = "KILLED" if failed else ("SURVIVED" if proc.returncode == 0 else "ERROR")
    shutil.rmtree(dst, ignore_errors=True)
    return {"name": name, "verdict": verdict, "rc": proc.returncode, "killed_by": failed[:4],
            "n": len(failed), "tail": tail[-1:] if tail else []}


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    if BASE.exists():
        shutil.rmtree(BASE)
    shutil.copytree(REPO / "engine_v2", BASE / "engine_v2", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "oanda.cfg", BASE / "oanda.cfg")
    csv_rel = Path("artifacts/commits/week8-volmom-multitf/20260920_104606_189c127")
    (BASE / csv_rel).mkdir(parents=True, exist_ok=True)
    for f in (REPO / csv_rel).glob("*_structure_events.csv"):
        shutil.copy2(f, BASE / csv_rel / f.name)
    names = sys.argv[2:]
    sites = [s for s in SITES if not names or s[0] in names]
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "7"))) as ex:
        results = list(ex.map(run, sites))
    for r in results:
        print(r["name"], r["verdict"], r["n"], r["killed_by"][:3], r["tail"])
    (RES / "summary.json").write_text(json.dumps(results, indent=1))
