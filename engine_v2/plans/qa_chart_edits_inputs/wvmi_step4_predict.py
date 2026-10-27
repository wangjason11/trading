"""Step 4 prediction ("WVMI back on the chart", 2026-10-26; was a scratchpad script): the per-LP rows of every
NON-collapsed WVMI record of S_lp_furthest, from `wvmi_asof.py`'s `new_to_bound` (the engine's pre-lock search, to
end - 1) + the lock (BOS_{n+1}'s last wave candle) + the cycle's lifecycle end / reason; `relative_dir` at the LP
candle from the save's `_subs.csv` step function. Prints the rows per CSV, per status / reason, per relative_dir and
per record; writes OUT_JSON for `wvmi_step4_check.py`. Measured: H1 57 / counter 74 / confluence 185 (= the replay).
`S` hard-codes S_lp_furthest (the step-4 baseline; its records = S_lp_rows').

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wvmi_asof.py ASOF_JSON
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_step4_predict.py ASOF_JSON OUT_JSON
"""
import ast, json, sys
from collections import Counter
import pandas as pd

J = json.load(open(sys.argv[1]))
S = ("artifacts/commits/week9-entries-decisions/20261026_192655_f63df34/"
     "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5")


def pick(lens, row):
    out = []
    for c in J:
        if c["lens"] != lens:
            continue
        if lens != "H1" and (c["sub_id"] is None or int(c["sub_id"]) != int(row["sub_id"])):
            continue
        if lens == "H1" and c["sid"] != int(row["bos_structure_id"]):
            continue
        if c["cyc"] != int(row["bos_cycle_id"]):
            continue
        off = c["off"]
        lp = None if pd.isna(row["lp_idx"]) else int(row["lp_idx"])
        clp = None if c["lp_final"] is None else int(c["lp_final"]) + off
        if clp != lp or int(c["fp"]) + off != int(row["fp_idx"]) or bool(c["locked"]) != bool(row["lp_locked"]) \
                or bool(c["collapsed"]) != bool(row["cycle_collapsed"]):
            continue
        out.append(c)
    return out


def rel_dir_fn(lens):
    subs = pd.read_csv(f"{S}_M15_{lens}_subs.csv")
    tab = {int(r.sub_id): (int(r.start_idx), ast.literal_eval(r.relative_dir_segments)) for r in subs.itertuples()}

    def at(sub_id, idx):
        start, segs = tab[int(sub_id)]
        if idx < start:
            return "NA (before start_idx)"
        cur = None
        for f, rd in segs:
            if f <= idx:
                cur = rd
        assert cur is not None
        return cur
    return at


REASON = {"reversal": "cycle ended by a reversal", "same_dir_replacement": "sub replaced", "parent_end": "sub ended"}
out, counts, statuses, problems = [], Counter(), Counter(), []
for lens, csv in (("H1", S + "_wvmi.csv"), ("counter", S + "_M15_counter_wvmi.csv"),
                  ("confluence", S + "_M15_confluence_wvmi.csv")):
    rows = pd.read_csv(csv)
    rd_at = rel_dir_fn(lens) if lens != "H1" else None
    for _, row in rows.iterrows():
        if row["cycle_collapsed"]:
            continue
        cs = pick(lens, row)
        tag = (f"{lens} {'sid' if lens == 'H1' else 'sub'}="
               f"{int(row['bos_structure_id'] if lens == 'H1' else row['sub_id'])} c{int(row['bos_cycle_id'])}")
        assert cs, tag
        keys = {(tuple(map(tuple, [x[:2] for x in c["new_to_bound"]])), c["lock_lp"], c["b"]) for c in cs}
        assert len(keys) == 1, (tag, keys)
        c = cs[0]; o = c["off"]
        seq = [(t + o, i + o, pm) for t, i, pm in c["new_to_bound"]]
        start, end, reason = c["life"]
        recs = []
        for k, (t, i, pm) in enumerate(seq[:-1]):
            recs.append(dict(lp_idx=i, as_of_idx=t, lp_status="temporary", lp_status_reason="replaced by a further close",
                             lp_status_idx=seq[k + 1][0], pm=pm))
        if c["locked"]:
            b = c["b"] + o
            lock_in = c["lock_lp_in_frame"]
            lock = c["lock_lp"] + o if lock_in else None
            reason_l = f"cycle {int(row['locked_by_cycle_id'])}'s BOS confirmed"
            if seq and (lock is None or lock == seq[-1][1]):
                t, i, pm = seq[-1]
                recs.append(dict(lp_idx=i, as_of_idx=t, lp_status="locked", lp_status_reason=reason_l, lp_status_idx=b, pm=pm))
            else:
                if seq:
                    t, i, pm = seq[-1]
                    recs.append(dict(lp_idx=i, as_of_idx=t, lp_status="temporary",
                                     lp_status_reason="replaced by the locked LP", lp_status_idx=b, pm=pm))
                if lock is not None:
                    recs.append(dict(lp_idx=lock, as_of_idx=b, lp_status="locked", lp_status_reason=reason_l,
                                     lp_status_idx=b, pm=float(row["pullback_momentum"])))
        elif seq:
            t, i, pm = seq[-1]
            if end is None:
                recs.append(dict(lp_idx=i, as_of_idx=t, lp_status="open", lp_status_reason="still moving at the data end",
                                 lp_status_idx=None, pm=pm))
            else:
                if reason not in REASON:
                    problems.append((tag, "unlocked end reason", reason, end))
                recs.append(dict(lp_idx=i, as_of_idx=t, lp_status="unlocked", lp_status_reason=REASON.get(reason, reason),
                                 lp_status_idx=end + o, pm=pm))
        # the last row's LP / momentum must be the record's
        if recs:
            if recs[-1]["lp_idx"] != int(row["lp_idx"]):
                problems.append((tag, "last row lp != record lp", recs[-1]["lp_idx"], row["lp_idx"]))
            if abs(recs[-1]["pm"] - float(row["pullback_momentum"])) > 1e-9:
                problems.append((tag, "last row pm != record pm", recs[-1]["pm"], row["pullback_momentum"]))
        elif not pd.isna(row["lp_idx"]):
            problems.append((tag, "no rows but the record has an LP", row["lp_idx"]))
        for r in recs:
            r.update(lens=lens, tag=tag, sub_id=None if lens == "H1" else int(row["sub_id"]),
                     relative_dir=None if lens == "H1" else rd_at(row["sub_id"], r["lp_idx"]),
                     bm=float(row["breakout_momentum"]), side=row["zone_side"])
            counts[lens] += 1
            statuses[(lens, r["lp_status"], r["lp_status_reason"])] += 1
        out.append(dict(tag=tag, rows=recs, life=c["life"], locked=bool(row["lp_locked"])))

print("rows per CSV:", dict(counts))
for k in sorted(statuses):
    print("  ", k, statuses[k])
print("problems:", problems)
rd = Counter((r["lens"], r["relative_dir"]) for rec in out for r in rec["rows"] if r["lens"] != "H1")
print("relative_dir at the LP candle:", dict(rd))
# per record summary
for rec in out:
    rs = rec["rows"]
    print(f"{rec['tag']:<26} rows={len(rs):>3}  last={rs[-1]['lp_status'] if rs else '-'}"
          f" ({rs[-1]['lp_status_reason'] if rs else ''} at {rs[-1]['lp_status_idx'] if rs else ''})"
          f"  life={rec['life']}")
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
