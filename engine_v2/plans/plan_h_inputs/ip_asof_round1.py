import ast, pandas as pd
exec(open("ip_proto.py").read().split("for _, fb in fibs.iterrows():")[0])  # reuse helpers, no report
ev = pd.read_csv(SAVE + r"\NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_structure_events.csv")
def picks(bos, cts, sd):
    bos_p = float(df.at[bos, "l"] if sd == 1 else df.at[bos, "h"]); cts_p = float(df.at[cts, "h"] if sd == 1 else df.at[cts, "l"])
    rng = abs(cts_p - bos_p)
    z = sorted([cts_p - sd*rng*.618, cts_p - sd*rng*.80]); res = []
    for i in range(bos, cts + 1):
        if df.at[i, "direction"] != -sd: continue
        if not (df.at[i, "h"] >= z[0] and df.at[i, "l"] <= z[1]): continue
        if not has_unfilled_imbalance(df, i + 1, cts, check_to_idx=cts, direction=sd, evaluated_at=None): continue
        p = classify(i, sd)
        if p: res.append((p[0][0], i, p[0][1], p[0][2]))
    res.sort(key=lambda r: (r[0], r[1]))  # rank, then earlier idx (TB not applied)
    return res
for _, fb in fibs.iterrows():
    sid, cyc, bos = int(fb.structure_id), int(fb.cycle_id), int(fb.bos_idx)
    sd = int(df.at[int(fb.cts_idx), "struct_direction"])
    rows = []
    for _, e in ev.iterrows():
        if e["type"] not in ("CTS_ESTABLISHED", "CTS_UPDATED"): continue
        m = ast.literal_eval(e["meta"])
        if m.get("structure_id") != sid or m.get("cycle_id") != cyc: continue
        anc = int(e["idx"]) if (e["type"] == "CTS_UPDATED" and m.get("via") == "replay_raw") else int(m["cts_anchor_idx"]); rows.append((int(m.get("confirmed_at", e["idx"])), anc, e["type"] + ("(raw)" if m.get("via") == "replay_raw" else "")))
    print(f"\nH1 sid {sid} cyc {cyc} BOS {bos} (final CTS {int(fb.cts_idx)})")
    for moment, anc, typ in rows:
        ps = picks(bos, anc, sd)
        print(f"  {typ:15s} moment {moment} CTS anchor {anc}: " + (" | ".join(f"{r[1]} {r[2]} {r[3].split(' ')[0]}" for r in ps) or "no pattern"))
