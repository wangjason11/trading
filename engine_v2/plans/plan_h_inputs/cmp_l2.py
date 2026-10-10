"""PLAN_L landing 2 (Step 3, the ranking rules) — the keyed compare vs S_L1 (PLAN_L §4.3). A positional diff breaks on
the POI `meta` cell (5 new keys) and on the IC CSVs' new columns.

    PYTHONPATH=. python engine_v2/plans/plan_h_inputs/cmp_l2.py BASE_SAVE_DIR [CUR_DEBUG] [CUR_CHARTS] [--trial DIR]

Checks (prints RESULT OK / FAIL, exit 1 on FAIL):
  1. every CSV of the save except the 3 POI + 3 IC CSVs is md5-equal (no structure / trigger / sub / WVMI change);
  2. the POI CSVs (key = sub, sid, cycle, IC): every baseline row present, every column equal, its meta equal but the
     5 new keys (`ip_outcome` "pattern" right before `ip_rank`; the 4 `ranking_*` None right after
     `ip_tb_imbalance_first_c2_idx`); the NEW rows == the predicted 7 (H1 180 / 694; confluence 670 / 929 / 1804 /
     2575 / 3809; counter none): `ip_outcome` "no_pattern", the pattern meta all None, the ranking rule, the
     activation history, the end; the ranking values typed per rule;
  3. the IC CSVs (key = the record + `ic_idx`): the columns = the baseline's + `ip_outcome` after `is_ic_candidate`
     + the 4 `ranking_*` at the end; every baseline row present and equal on the baseline columns, but `picked` on
     the 7 ranked picks; the 7 records' baseline `no_pattern` rows gone, every other no-IC row kept
     (`no_ic_candidate`); `ip_outcome` per record on every row (pattern / no_pattern / empty); the `ranking_*` on the
     7 picked rows only;
  4. the figures' trace / shape counts == the prediction; no `IP: none` / `IP: not evaluated` text left;
  5. with `--trial DIR` (the trial replay's debug dir, old CSV names): its ranked POIs / RANKED rows == these.
"""
import ast
import glob
import hashlib
import json
import os
import sys

import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given args
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
TRIAL = None
if "--trial" in sys.argv:
    TRIAL = sys.argv[sys.argv.index("--trial") + 1]
    args.remove(TRIAL)
BASE = args[0]
CUR = args[1] if len(args) > 1 else "artifacts/debug"
CHARTS = args[2] if len(args) > 2 else "artifacts/charts"

LENS = {"H1": "*rk2-5_{}.csv", "confluence": "*M15_confluence_{}.csv", "counter": "*M15_counter_{}.csv"}
# (sub_id | None, sid, cycle, ic): (ranking_rule, activation history [(idx, active)], end_idx, end_reason)
NEW_POIS = {
    "H1": {(None, 0, 0, 180): ("exclude_pinbar", [(430, True)], 652, "next_cycle"),
           (None, 1, 0, 694): ("most_recent_before_imbalance", [], 902, "next_cycle")},
    "confluence": {(0, 0, 0, 670): ("most_recent_before_imbalance", [], 1020, "next_cycle"),
                   (0, 0, 1, 929): ("exclude_pinbar", [(1020, True)], 1224, "next_cycle"),
                   (1, 0, 0, 1804): ("most_recent_before_imbalance", [], 1940, "next_cycle"),
                   (2, 0, 1, 2575): ("immediate_unfilled_imbalance", [(2608, True)], 2829, "reversal"),
                   (6, 0, 0, 3809): ("immediate_unfilled_imbalance", [(3819, True)], 4200, "reversal")},
    "counter": {},
}
RANKING_KEYS = ["ranking_rule", "ranking_runner_up_ic_idx", "ranking_in_play_count", "ranking_imbalance_first_c2_idx"]
NEW_META = ["ip_outcome"] + RANKING_KEYS
PATTERN_META = ["ip_rank", "ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx", "pattern_end_idx",
                "pattern_confirmed_idx", "ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count",
                "ip_tb_imbalance_first_c2_idx"]
# which ranking values a rule carries: (runner-up, gap)
RULE_PARTS = {"exclude_pinbar": (False, False), "immediate_unfilled_imbalance": (True, True),
              "ic_candle_type": (True, False), "most_recent_before_imbalance": (True, True)}
FIGS = {"rk2-5.html": (81, 83, 236, 239), "rk2-5_M15_confluence.html": (274, 282, 264, 276),
        "rk2-5_M15_counter.html": (135, 137, 120, 123)}
REC = ["structure_path_id", "sub_id", "structure_id", "cycle_id", "bos_anchor_idx", "cts_anchor_idx"]
ok = True


def fail(msg):
    global ok
    ok = False
    print("  FAIL", msg)


def one(d, pat):
    got = glob.glob(os.path.join(d, pat))
    assert len(got) == 1, (d, pat, got)
    return got[0]


def cur_path(d, pat):
    """The current file = the SAVE's file name in `d` (the debug / charts dirs hold other windows' outputs too)."""
    return os.path.join(d, os.path.basename(one(BASE, pat)))


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def poi_key(r, m):
    return (m.get("sub_id"), int(r["structure_id"]), int(r["cycle_id"]), int(r["ic_idx"]))


# 1. every other CSV byte-equal
changed = {os.path.basename(one(BASE, LENS[l].format(k))) for l in LENS for k in ("poi_zones", "ic_candidates")}
base_csv = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASE, "*.csv")))
print(f"== CSVs: {len(base_csv)} in the save, {len(changed)} expected to differ")
for f in base_csv:
    if f in changed:
        continue
    cur = os.path.join(CUR, f)
    if not os.path.exists(cur):
        fail(f"missing {f}")
    elif md5(cur) != md5(os.path.join(BASE, f)):
        fail(f"differs: {f}")

# 2. the POI CSVs
for lens, pat in LENS.items():
    a, b = read(one(BASE, pat.format("poi_zones"))), read(cur_path(CUR, pat.format("poi_zones")))
    print(f"== POI {lens}: rows {len(a)} -> {len(b)}")
    if list(a.columns) != list(b.columns):
        fail(f"POI {lens} columns {list(b.columns)}")
    am = {poi_key(r, ast.literal_eval(r["meta"])): (r, ast.literal_eval(r["meta"])) for _, r in a.iterrows()}
    bm = {poi_key(r, ast.literal_eval(r["meta"])): (r, ast.literal_eval(r["meta"])) for _, r in b.iterrows()}
    if len(am) != len(a) or len(bm) != len(b):
        fail(f"POI {lens}: duplicate keys")
    for k, (ra, ma) in am.items():
        if k not in bm:
            fail(f"POI {lens} {k}: baseline row missing")
            continue
        rb, mb = bm[k]
        cols = [c for c in a.columns if c != "meta" and ra[c] != rb[c]]
        if cols:
            fail(f"POI {lens} {k}: columns differ {cols}")
        stripped = {kk: v for kk, v in mb.items() if kk not in NEW_META}
        if stripped != ma or list(stripped) != list(ma):
            fail(f"POI {lens} {k}: meta differs beyond the new keys")
        keys = list(mb)
        if keys[keys.index("ip_rank") - 1] != "ip_outcome" or \
                keys[keys.index("ip_tb_imbalance_first_c2_idx") + 1:keys.index("ip_tb_imbalance_first_c2_idx") + 5] \
                != RANKING_KEYS:
            fail(f"POI {lens} {k}: new key order {keys}")
        if mb["ip_outcome"] != "pattern" or any(mb[kk] is not None for kk in RANKING_KEYS):
            fail(f"POI {lens} {k}: old row new keys {[mb[kk] for kk in NEW_META]}")
    new = {k: bm[k] for k in bm if k not in am}
    if set(new) != set(NEW_POIS[lens]):
        fail(f"POI {lens}: new rows {sorted(new, key=str)} != predicted {sorted(NEW_POIS[lens], key=str)}")
    for k, (rb, mb) in new.items():
        if k not in NEW_POIS[lens]:
            continue
        rule, hist, end_idx, end_reason = NEW_POIS[lens][k]
        got_hist = [(e["idx"], e["active"]) for e in mb["activation_history"]]
        print(f"   NEW {k}: rule {mb['ranking_rule']} runner-up {mb['ranking_runner_up_ic_idx']} "
              f"in-play {mb['ranking_in_play_count']} gap {mb['ranking_imbalance_first_c2_idx']} hist {got_hist} "
              f"end {mb['end_idx']} {mb['end_reason']} status {mb['status']}")
        if mb["ip_outcome"] != "no_pattern" or rb["ip_rank"] != "":
            fail(f"POI {lens} {k}: ip_outcome {mb['ip_outcome']} ip_rank {rb['ip_rank']!r}")
        if any(mb[kk] is not None for kk in PATTERN_META):
            fail(f"POI {lens} {k}: pattern meta {[mb[kk] for kk in PATTERN_META]}")
        if (mb["ranking_rule"], got_hist, mb["end_idx"], mb["end_reason"]) != (rule, hist, end_idx, end_reason):
            fail(f"POI {lens} {k}: {(mb['ranking_rule'], got_hist, mb['end_idx'], mb['end_reason'])} != predicted")
        has_r, has_g = RULE_PARTS[rule]
        r_up, cnt, gap = mb["ranking_runner_up_ic_idx"], mb["ranking_in_play_count"], mb["ranking_imbalance_first_c2_idx"]
        if (type(r_up) is int) != has_r or (type(gap) is int) != has_g or type(cnt) is not int \
                or (not has_r and r_up is not None) or (not has_g and gap is not None):
            fail(f"POI {lens} {k}: ranking values {r_up!r} {cnt!r} {gap!r} for {rule}")

# 3. the IC CSVs
for lens, pat in LENS.items():
    a, b = read(one(BASE, pat.format("ic_candidates"))), read(cur_path(CUR, pat.format("ic_candidates")))
    print(f"== IC {lens}: rows {len(a)} -> {len(b)}")
    ca = list(a.columns)
    k = ca.index("is_ic_candidate")
    want = ca[:k + 1] + ["ip_outcome"] + ca[k + 1:] + RANKING_KEYS
    if list(b.columns) != want:
        fail(f"IC {lens} columns {list(b.columns)}")
    key = lambda df: df[REC + ["ic_idx"]].astype(str).agg("|".join, axis=1)
    a["_k"], b["_k"] = key(a), key(b)
    a["_r"], b["_r"] = a[REC].astype(str).agg("|".join, axis=1), b[REC].astype(str).agg("|".join, axis=1)
    if a["_k"].duplicated().any() or b["_k"].duplicated().any():
        fail(f"IC {lens}: duplicate keys")
    preds = {(None if s is None else str(s), str(sid), str(cyc), str(ic)): rule
             for (s, sid, cyc, ic), (rule, *_r) in NEW_POIS[lens].items()}
    ranked_recs = {}
    for _, r in a[a["ic_idx"] == ""].iterrows():
        if r["reason"] == "no_pattern":
            ranked_recs[r["_r"]] = r
        elif r["reason"] != "no_ic_candidate":
            fail(f"IC {lens}: baseline no-IC reason {r['reason']}")
    bi = b.set_index("_k")
    picks_now = {}
    for _, r in a.iterrows():
        if r["ic_idx"] == "":
            if r["_r"] in ranked_recs:
                if r["_k"] in bi.index:
                    fail(f"IC {lens} {r['_k']}: the no_pattern row survived")
                continue
            if r["_k"] not in bi.index or bi.loc[r["_k"], "reason"] != "no_ic_candidate":
                fail(f"IC {lens} {r['_k']}: no-IC row lost / changed")
            continue
        if r["_k"] not in bi.index:
            fail(f"IC {lens} {r['_k']}: baseline row missing")
            continue
        c = bi.loc[r["_k"]]
        diff = [col for col in ca if col not in ("_k", "_r") and r[col] != c[col]]
        if diff == ["picked"] and r["_r"] in ranked_recs and c["picked"] == "True":
            picks_now[r["_r"]] = c
        elif diff:
            fail(f"IC {lens} {r['_k']}: differs on {diff}")
    if len(b) != len(a) - len(ranked_recs):
        fail(f"IC {lens}: {len(b)} rows, expected {len(a) - len(ranked_recs)}")
    if set(picks_now) != set(ranked_recs):
        fail(f"IC {lens}: ranked picks for {sorted(picks_now)} vs no_pattern records {sorted(ranked_recs)}")
    got = {}
    for rk, c in picks_now.items():
        got[(c["sub_id"] or None, c["structure_id"], c["cycle_id"], c["ic_idx"])] = c["ranking_rule"]
    if got != preds:
        fail(f"IC {lens}: ranked picks {got} != predicted {preds}")
    # ip_outcome per record on every row; ranking only on the 7 picked rows
    for rec, g in b.groupby("_r"):
        outs = set(g["ip_outcome"])
        if rec in ranked_recs:
            want_out = "no_pattern"
        elif (g["ic_idx"] == "").any():
            want_out = ""
        else:
            want_out = "pattern"
        if outs != {want_out}:
            fail(f"IC {lens} {rec}: ip_outcome {outs} != {want_out!r}")
        for _, r in g.iterrows():
            is_ranked_pick = rec in ranked_recs and r["picked"] == "True"
            if not is_ranked_pick and any(r[kk] != "" for kk in RANKING_KEYS):
                fail(f"IC {lens} {r['_k']}: ranking values on a non-ranked row")
    print(f"   ranked picks {sorted(got.items(), key=str)}; no-IC rows kept "
          f"{int((b['ic_idx'] == '').sum())} ({sorted(set(b.loc[b['ic_idx'] == '', 'reason']))})")

# 4. the figures
for suffix, (t0, t1, s0, s1) in FIGS.items():
    (da, la), (db, lb) = load_fig(one(BASE, "*" + suffix)), load_fig(cur_path(CHARTS, "*" + suffix))
    na = (len(da), len(la.get("shapes", [])))
    nb = (len(db), len(lb.get("shapes", [])))
    text = json.dumps([db, lb], default=str, ensure_ascii=False)
    print(f"== figure {suffix}: traces {na[0]} -> {nb[0]}, shapes {na[1]} -> {nb[1]}; "
          f"'No pattern ·' x{text.count('No pattern ·')}, 'IC: none' x{text.count('IC: none')}")
    if (na, nb) != ((t0, s0), (t1, s1)):
        fail(f"figure {suffix}: {na} -> {nb}, predicted {(t0, s0)} -> {(t1, s1)}")
    # keyed: the traces / shapes as multisets of their full JSON (a positional diff shifts at the first new POI)
    from collections import Counter
    ka = Counter(json.dumps(t, sort_keys=True, default=str) for t in da)
    kb = Counter(json.dumps(t, sort_keys=True, default=str) for t in db)
    gone = Counter(json.loads(k).get("name") for k in (ka - kb).elements())
    new = Counter(json.loads(k).get("name") for k in (kb - ka).elements())
    sa = Counter(json.dumps(x, sort_keys=True, default=str) for x in la.get("shapes", []))
    sb = Counter(json.dumps(x, sort_keys=True, default=str) for x in lb.get("shapes", []))
    print(f"   traces only in the save {dict(gone)}; only now {dict(new)}; "
          f"shapes only in the save {sum((sa - sb).values())}, only now {sum((sb - sa).values())}")
    rest = {kk: v for kk, v in la.items() if kk != "shapes"} != {kk: v for kk, v in lb.items() if kk != "shapes"}
    if rest:
        fail(f"figure {suffix}: layout differs beyond the shapes")
    allowed = ("POI zone", "M15 POI zone", "H1 POI zone (overlay)", "fib:", "M15 fib")
    odd = [nm for nm in list(gone) + list(new) if not str(nm).startswith(allowed)]
    if odd:
        fail(f"figure {suffix}: traces beyond the POI / fib hovers changed: {odd}")
    if "IP: none" in text or "IP: not evaluated" in text:
        fail(f"figure {suffix}: an IP: none / IP: not evaluated label left")

# 5. the trial cross-check (old CSV names; the trial kept the failing counts, its reason "RANKED:<rule>")
if TRIAL:
    print("== trial cross-check")
    for lens, pat in LENS.items():
        t = read(one(TRIAL, pat.format("poi_zones")))
        b = read(cur_path(CUR, pat.format("poi_zones")))
        tm = {poi_key(r, ast.literal_eval(r["meta"])): (r, ast.literal_eval(r["meta"])) for _, r in t.iterrows()}
        bm = {poi_key(r, ast.literal_eval(r["meta"])): (r, ast.literal_eval(r["meta"])) for _, r in b.iterrows()}
        if set(tm) != set(bm):
            fail(f"trial POI {lens}: keys differ")
        for k in tm:
            if k not in bm:
                continue
            (rt, mt), (rb, mb) = tm[k], bm[k]
            cols = [c for c in t.columns if c not in ("meta", "ip_rank") and rt[c] != rb[c]]
            skip = set(NEW_META) | ({"ip_leg_candle_counts"} if mt["ip_rank"] is None else set())
            mdiff = sorted(kk for kk in set(mt) | set(mb) if kk not in skip and mt.get(kk) != mb.get(kk))
            if cols or mdiff:
                fail(f"trial POI {lens} {k}: columns {cols} meta {mdiff}")
        ti = read(one(TRIAL, pat.format("ip_candidates")))
        trial_ranked = {(r["sub_id"] or None, r["structure_id"], r["cycle_id"], r["ic_idx"]): r["reason"].split(":")[1]
                        for _, r in ti.iterrows() if r["reason"].startswith("RANKED")}
        bi = read(cur_path(CUR, pat.format("ic_candidates")))
        cur_ranked = {(r["sub_id"] or None, r["structure_id"], r["cycle_id"], r["ic_idx"]): r["ranking_rule"]
                      for _, r in bi.iterrows() if r["picked"] == "True" and r["ip_outcome"] == "no_pattern"}
        print(f"   {lens}: trial {sorted(trial_ranked.items(), key=str)}")
        if trial_ranked != cur_ranked:
            fail(f"trial IC {lens}: {trial_ranked} != {cur_ranked}")

print("RESULT", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
