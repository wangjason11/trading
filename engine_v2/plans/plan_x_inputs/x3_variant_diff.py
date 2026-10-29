"""PLAN_X X3 — diff a base and a re-ask VARIANT scratch-tree replay of one window (`x3_variant_replay.py`):

  1. `plan_w_inputs/w9b_window_compare.py` (every CSV by md5 + the rows added / removed; the charts' traces);
  2. the MS decisions: every structure-events CSV (H1 + both lenses) — rows added / removed (any = an MS event moved);
     the triggers / subs / unresolved CSVs likewise;
  3. the fib records keyed (structure_path_id, structure_id, cycle_id, version): records only in one side, and per
     shared record the fields + meta keys that differ;
  4. the variant's `[fib][reask]` log lines (the downstream trackers' CHANGED re-asks, the summaries, any walk-back
     WARNING).

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_x_inputs/x3_variant_diff.py BASE_TREE VAR_TREE OUT.txt
"""
import ast
import collections
import glob
import os
import subprocess
import sys

import pandas as pd

base, var, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = []
here = os.path.dirname(os.path.abspath(__file__))
cmp = subprocess.run([sys.executable, os.path.join(here, "..", "plan_w_inputs", "w9b_window_compare.py"), base, var],
                     capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
OUT.append("== 1. w9b_window_compare ==")
OUT.append(cmp.stdout.rstrip() + (("\n" + cmp.stderr[-2000:]) if cmp.returncode else ""))


def _csvs(tree, suffix):
    return {os.path.basename(p): p for p in glob.glob(os.path.join(tree, "artifacts", "debug", f"*{suffix}"))}


OUT.append("\n== 2. MS decisions: structure events + triggers / subs (rows only in one side) ==")
for suffix in ("structure_events.csv", "triggers.csv", "subs.csv", "unresolved_triggers.csv"):
    b, v = _csvs(base, suffix), _csvs(var, suffix)
    for n in sorted(set(b) | set(v)):
        if n not in b or n not in v:
            OUT.append(f"  {n}: only in {'base' if n in b else 'variant'}")
            continue
        a = collections.Counter(open(b[n], encoding="utf-8").read().splitlines())
        c = collections.Counter(open(v[n], encoding="utf-8").read().splitlines())
        rem, add = list((a - c).elements()), list((c - a).elements())
        OUT.append(f"  {n}: rows {sum(a.values())} -> {sum(c.values())}; removed {len(rem)}, added {len(add)}")
        for line in rem[:12]:
            OUT.append(f"     - {line[:300]}")
        for line in add[:12]:
            OUT.append(f"     + {line[:300]}")

OUT.append("\n== 3. fib records keyed (structure_path_id, structure_id, cycle_id, version) ==")
KEY = ["structure_path_id", "structure_id", "cycle_id", "version"]
b, v = _csvs(base, "fib_lifecycle.csv"), _csvs(var, "fib_lifecycle.csv")
for n in sorted(set(b) & set(v)):
    fb, fv = pd.read_csv(b[n]), pd.read_csv(v[n])

    def keyed(df):
        rows = {}
        for _, r in df.iterrows():
            k = tuple("NA" if pd.isna(r[c]) else r[c] for c in KEY) + (int(r["bos_idx"]),)
            rows.setdefault(k, []).append(r)
        return rows
    kb, kv = keyed(fb), keyed(fv)
    lines = []
    for k in sorted(set(kb) | set(kv), key=str):
        if k not in kb or k not in kv:
            r = (kv if k in kv else kb)[k][0]
            lines.append(f"  only in {'variant' if k in kv else 'base'}: {k} start={r['start_idx']} end={r['end_idx']} "
                         f"{r['end_reason']} status={r['status']} active={r['active']} locked={r['locked']} "
                         f"cts={r['cts_idx']} meta={str(r['meta'])[:220]}")
            continue
        for rb, rv in zip(kb[k], kv[k]):
            fields = [c for c in fb.columns if c != "meta" and str(rb[c]) != str(rv[c])]
            mb, mv = ast.literal_eval(rb["meta"]), ast.literal_eval(rv["meta"])
            mkeys = sorted(x for x in set(mb) | set(mv) if mb.get(x, "<absent>") != mv.get(x, "<absent>"))
            if fields or mkeys:
                lines.append(f"  {k}: " + "; ".join(
                    [f"{c} {rb[c]} -> {rv[c]}" for c in fields]
                    + [f"meta.{x} {mb.get(x, '<absent>')} -> {mv.get(x, '<absent>')}" for x in mkeys]))
    OUT.append(f"  {n}: records {len(fb)} -> {len(fv)}; differing {len(lines)}")
    OUT.extend(lines)

OUT.append("\n== 4. the variant's [fib][reask] log lines ==")
log = open(os.path.join(var, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
rl = [l for l in log if "[fib][reask]" in l]
OUT.append(f"  {len(rl)} lines; CHANGED {sum('CHANGED' in l for l in rl)}; WARNING {sum('WARNING' in l for l in rl)}")
OUT.extend(f"  {l}" for l in rl if "CHANGED" in l or "WARNING" in l)
summ = [l for l in rl if "summary" in l]
OUT.append(f"  summaries ({len(summ)}; one per downstream feed — the last carries the process walk-back total):")
OUT.extend(f"  {l}" for l in summ)
open(out_path, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print("\n".join(OUT))
