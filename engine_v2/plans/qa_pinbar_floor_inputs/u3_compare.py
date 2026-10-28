"""PLAN_U U3 /compare: the final window's downstream vs U2, per structure, keyed (never positional) — `u2_compare.py`
with section 1 checked against the prediction's U3 column.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_compare.py BASE_DIR CUR_DIR [OUT.txt]

BASE_DIR = the S_U2 save, CUR_DIR = the U3 variant's (or the landing's) artifacts/debug. Prints:
  1. the H1 candle types by name (`*_final.csv`): U2 -> U3 changes vs the prediction's per-candle U3 list; the floor
     column; every other `*_final.csv` column whose values changed (row counts);
  2. the structure events per (path, sub_id, structure_id, cycle_id, type): the (idx, anchor) lists before -> after;
  3. KL zones, POI zones, fibs, the picked ICs, WVMI records — each keyed per structure, before -> after;
  4. the files byte-equal.
"""
from __future__ import annotations

import ast
import filecmp
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from u3_engine_vs_oracle import predicted  # noqa: E402

B, C = Path(sys.argv[1]), Path(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else str(HERE / "u3_compare.txt")
P = "NZD_USD_H1_2025-11-15_2026-01-20"
Q = P + "_sd-1_eps0p0001_rk2-5"
LENSES = (("H1", ""), ("counter", "_M15_counter"), ("confluence", "_M15_confluence"))
lines: list[str] = []


def out(s=""):
    lines.append(s)


def meta(x):
    return {} if pd.isna(x) else ast.literal_eval(x)


def read(d, name):
    return pd.read_csv(d / name, dtype=str, keep_default_na=False)


# --- 1. the H1 candle types -----------------------------------------------------------------------------------------
fb, fc = read(B, f"{P}_final.csv"), read(C, f"{P}_final.csv")
assert list(fb.columns) == list(fc.columns) and len(fb) == len(fc)
tb, tc = fb["candle_type"].to_numpy(), fc["candle_type"].to_numpy()
before = fc["candle_type_before_floor"].to_numpy()
ch = tb != tc
pred = predicted(HERE / "plan_u_prediction.txt", "U3")["H1"]
exp = tb.copy()
for i, t in pred.items():
    exp[i] = t
bad = [i for i in range(len(tc)) if tc[i] != exp[i]]
isfp = fc["is_floor_pinbar"].to_numpy() == "True"
vb, vc = pd.Series(tb).value_counts(), pd.Series(tc).value_counts()
out("## 1. H1 candle types (*_final.csv, by name)")
out(f"   S_U2 -> U3: {int(ch.sum())} candles change type ({int((ch & (tc == 'pinbar')).sum())} become pinbars, "
    f"{int((ch & (tb == 'pinbar')).sum())} stop); from maru {int((ch & (before == 'maru')).sum())}, "
    f"normal {int((ch & (before == 'normal')).sum())}; with a pin direction "
    f"{int((ch & ((fb['pinbar_dir'] != '0') | (fc['pinbar_dir'] != '0'))).sum())} (on the side where it is a pinbar)")
out(f"   types maru/normal/pinbar {vb['maru']}/{vb['normal']}/{vb['pinbar']} -> {vc['maru']}/{vc['normal']}/{vc['pinbar']}")
out(f"   is_floor_pinbar {int(isfp.sum())} (maru {int((isfp & (before == 'maru')).sum())}, "
    f"normal {int((isfp & (before == 'normal')).sum())}); pinbar_body_floor empty on "
    f"{int((fc['pinbar_body_floor'] == '').sum())} rows")
out(f"   vs the prediction's per-candle list: {len(bad)} mismatches {bad[:10]}")
diffcols = [(c, int((fb[c] != fc[c]).sum())) for c in fb.columns if (fb[c] != fc[c]).any()]
out("   *_final.csv columns that changed (rows):")
for j in range(0, len(diffcols), 8):
    out("     " + ", ".join(f"{c} {n}" for c, n in diffcols[j:j + 8]))
out()


# --- 2. structure events ----------------------------------------------------------------------------------------------
def anchor(m):
    for k in ("cts_anchor_idx", "bos_anchor_idx", "pattern_anchor_idx", "range_anchor_idx"):
        if m.get(k) is not None:
            return f"{k[:3]}{m[k]}"
    return ""


def events(d, suffix):
    ev = read(d, f"{Q}{suffix}_structure_events.csv")
    g = defaultdict(list)
    for r in ev.itertuples():
        m = meta(r.meta)
        tag = m.get("via") or m.get("confirmation_method") or m.get("to") or ""
        g[(m.get("sub_id"), m.get("structure_id"), m.get("cycle_id"), r.type)].append(
            (int(r.idx), anchor(m), str(tag)))
    return {k: sorted(v) for k, v in g.items()}, len(ev)


out("## 2. structure events, per (sub_id, structure_id, cycle_id, type): (idx, anchor, via/to) before -> after")
for lens, sfx in LENSES:
    eb, nb = events(B, sfx)
    ec, nc = events(C, sfx)
    keys = sorted(set(eb) | set(ec), key=lambda k: tuple(str(x) for x in k))
    diff = [k for k in keys if eb.get(k) != ec.get(k)]
    out(f"### {lens}: {nb} -> {nc} events; {len(diff)} keys differ")
    for k in diff:
        b, c = eb.get(k, []), ec.get(k, [])
        gone = [x for x in b if x not in c]
        new = [x for x in c if x not in b]
        out(f"   sub={k[0]} sid={k[1]} cyc={k[2]} {k[3]}: -{gone} +{new}")
out()


# --- 3. zones, fibs, ICs, WVMI ----------------------------------------------------------------------------------------
def keyed(d, name, keyf, valf):
    df = read(d, name)
    g = defaultdict(list)
    for r in df.to_dict("records"):
        g[keyf(r)].append(valf(r))
    out_ = {}
    for k, v in g.items():   # collapse repeated rows to (value, count)
        c = defaultdict(int)
        for x in v:
            c[x] += 1
        out_[k] = sorted(((x if n == 1 else (x, f"x{n}")) for x, n in c.items()), key=str)
    return out_


def report(title, name, keyf, valf):
    for lens, sfx in LENSES:
        n = name.format(sfx=sfx)
        if filecmp.cmp(B / n, C / n, shallow=False):
            out(f"### {title} {lens}: byte-equal")
            continue
        kb, kc = keyed(B, n, keyf, valf), keyed(C, n, keyf, valf)
        diff = [k for k in sorted(set(kb) | set(kc), key=str) if kb.get(k) != kc.get(k)]
        out(f"### {title} {lens}: {sum(map(len, kb.values()))} -> {sum(map(len, kc.values()))} rows; {len(diff)} keys differ")
        for k in diff:
            out(f"   {k}: {kb.get(k)} -> {kc.get(k)}")
        db, dc = read(B, n), read(C, n)
        if "meta" in db.columns and len(db) == len(dc):
            keys = defaultdict(int)
            for mb, mc in zip(db["meta"], dc["meta"]):
                if mb != mc:
                    a, z = meta(mb), meta(mc)
                    for kk in sorted(set(a) | set(z)):
                        if a.get(kk) != z.get(kk):
                            keys[kk] += 1
            if keys:
                out("   meta keys that differ (rows, positional): " + ", ".join(f"{kk} {v}" for kk, v in keys.items()))


def kl_key(r):
    m = meta(r["meta"])
    return (m.get("sub_id"), m.get("structure_id"), m.get("cycle_id"), r["source_kind"])


def kl_val(r):
    m = meta(r["meta"])
    return (m.get("confirmed_idx"), m.get("base_idx"), m.get("base_pattern"), r["top"], r["bottom"], r["end_time"][:16])


out("## 3. zones, fibs, ICs, WVMI (keyed per structure)")
report("KL", Q + "{sfx}_kl_zones.csv", kl_key, kl_val)
report("POI", Q + "{sfx}_poi_zones.csv",
       lambda r: (meta(r["meta"]).get("sub_id"), r["structure_id"], r["cycle_id"]),
       lambda r: (r["ic_idx"], r["confirmed_idx"], r["ip_rank"], r["top"], r["bottom"]))
report("fib", Q + "{sfx}_fib_lifecycle.csv",
       lambda r: (r["structure_path_id"], meta(r["meta"]).get("sub_id"), r["structure_id"], r["cycle_id"], r["version"]),
       lambda r: (r["start_idx"], r["end_idx"], r["status"], r["bos_idx"], r["cts_idx"], r["active"], r["locked"]))
report("picked IC", Q + "{sfx}_ic_candidates.csv",
       lambda r: (r["sub_id"], r["structure_id"], r["cycle_id"]),
       lambda r: (r["ic_idx"], r["candle_type"], r["ip_rank"], r["ip_outcome"], r["cts_anchor_idx"])
       if r["picked"] == "True" else ("check row",))
report("WVMI", Q + "{sfx}_wvmi.csv",
       lambda r: (r["sub_id"], r["bos_structure_id"], r["bos_cycle_id"]),
       lambda r: (r["fb_idx"], r["lb_idx"], r["fp_idx"], r["lp_idx"], r["breakout_momentum"], r["pullback_momentum"],
                  r["status"]))
for lens, sfx in LENSES:
    n = f"{Q}{sfx}_wvmi_lps.csv"
    out(f"### WVMI LP rows {lens}: {len(read(B, n))} -> {len(read(C, n))}")
out()
eq = sorted(p.name for p in B.glob("*.csv") if filecmp.cmp(p, C / p.name, shallow=False))
out(f"## 4. byte-equal ({len(eq)} of {len(list(B.glob('*.csv')))}): " + ", ".join(x.replace(Q, "").replace(P, "") for x in eq))

text = "\n".join(lines)
print(text)
open(OUT, "w", encoding="utf-8").write(text + "\n")
