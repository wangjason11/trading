"""Plan E E4b pre-review mutation loop (the 12 BOS location readers; PLAN_E §8.1). Kept since
2026-09-25 as the template for any stage that moves an index role.

    python engine_v2/plans/plan_e_inputs/review_scripts/mut_loc_e4b.py <REPO_ROOT> [SITE ...]

Copies REPO_ROOT/engine_v2 (+ oanda.cfg + the Plan-B H1 events CSV a test needs) to
%TEMP%/pe/e4b_mut/repo, applies each mutant ALONE to a fresh copy, runs the full
suite with sys.executable (never the bare "python": another interpreter without pytest
is on PATH) and prints KILLED / SURVIVED / ERROR (a non-zero rc without FAILED lines).
The SITES needles are exact source snippets at commit 7216c9d — re-check them before reuse
(`mutate` asserts each needle occurs). Original docstring:

E4b pre-review mutation loop: every BOS LOCATION reader switched from the anchor
accessor to the raw `ev.idx` (the MOMENT since E4b). Each mutant on a fresh copy of
the working tree; full suite; killed-by recorded. Scratch only.
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
ROOT = Path(os.environ["TEMP"]) / "pe" / "e4b_mut"
BASE = ROOT / "repo"
RES = ROOT / "results"

# (name, file, unique line substring, regex old, new)
ACC = r"ef\.bos_anchor_idx\((\w+)\)"
SITES = [
    ("B1_m15_dot", "charting/export_m15_chart.py", "b_idx = ef.bos_anchor_idx(ev)   # the BOS dot sits at its anchor", 0),
    ("B2_m15_line", "charting/export_m15_chart.py", "fb_idx = ef.bos_anchor_idx(fb)   # the line's BOS end (location)", 0),
    ("B3_ovl_dot", "charting/export_m15_chart.py", "b_idx = ef.bos_anchor_idx(ev)   # the BOS dot sits at its anchor", 1),
    ("B4_ovl_line", "charting/export_m15_chart.py", "fb_idx = ef.bos_anchor_idx(fb)   # the line's BOS end (location)", 1),
    ("B5_h1_dot", "charting/export_plotly.py", "p_idx = ef.bos_anchor_idx(ev)", 0),
    ("B6_h1_line", "charting/export_plotly.py", "bos_idx = ef.bos_anchor_idx(first_bos)", 0),
    ("B7_fc_input", "multitf/first_confluence_trigger.py", "input_idx = ef.bos_anchor_idx(ev)", 0),
    ("B8_prevbos_start", "pipeline/orchestrator.py", "last_bos_anchor_by_sid[sid] = (ef.bos_anchor_idx(ev)", 0),
    ("B9_fib_bos", "pipeline/orchestrator.py", "bos_anchor_by_cycle[key] = (ef.bos_anchor_idx(ev)", 0),
    ("B10_levels", "structure/market_structure.py", "time=t.iloc[ef.bos_anchor_idx(ev)]", 0),
    ("B11_kl_anchor", "zones/kl_zones_v1.py", "anchor_idx = ef.bos_anchor_idx(ev)   # the BOS anchor", 0),
    ("B12_stamped", "structure/event_fields.py", '    if ev.type == "BOS_CONFIRMED":\n        return bos_anchor_idx(ev)\n    if ev.type == "CTS_UPDATED":', 0),
]


def mutate(dst: Path, rel: str, needle: str, occ: int) -> None:
    p = dst / "engine_v2" / rel
    src = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in src
    s = src.replace("\r\n", "\n")
    parts = s.split(needle)
    assert len(parts) >= occ + 2, (rel, needle, len(parts) - 1)
    if "\n" in needle:  # stamped_idx block
        new = needle.replace("return bos_anchor_idx(ev)", "return int(ev.idx)")
    else:
        new = re.sub(ACC, r"int(\1.idx)", needle)
        assert new != needle
    s = needle.join(parts[:occ + 1]) + new + needle.join(parts[occ + 1:])
    if crlf:
        s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8"))


def run(site) -> dict:
    name, rel, needle, occ = site
    dst = ROOT / f"w_{name}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    mutate(dst, rel, needle, occ)
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
    shutil.copytree(REPO / "engine_v2", BASE / "engine_v2",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "oanda.cfg", BASE / "oanda.cfg")
    csv_rel = Path("artifacts/commits/week8-volmom-multitf/20260920_104606_189c127")
    (BASE / csv_rel).mkdir(parents=True, exist_ok=True)
    for f in (REPO / csv_rel).glob("*_structure_events.csv"):
        shutil.copy2(f, BASE / csv_rel / f.name)
    base = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no"],
                          cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("BASELINE rc", base.returncode, [l for l in base.stdout.splitlines() if " passed" in l][-1:])
    names = sys.argv[2:]
    sites = [s for s in SITES if not names or s[0] in names]
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        results = list(ex.map(run, sites))
    for r in results:
        print(r["name"], r["verdict"], r["n"], r["killed_by"][:3], r["tail"])
    (RES / "summary.json").write_text(json.dumps(results, indent=1))
