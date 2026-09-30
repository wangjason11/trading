"""Analyze the diagnostic CSVs from zone_proximity_diag.py.

Validates the three hypothesized root causes for var3/var4 over-firing:
  (a) POI-inner drift toward CTS as cycle progresses
  (b) absurdly short window spans (sd->opp_sd->sd in <5 candles)
  (c) apex never reaches CTS inner (stays well outside trigger threshold)

Prints summary tables to stdout. Delete this file after the investigation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

PER_TRIGGER = Path("/tmp/zpt_per_trigger.csv")
VAR3 = Path("/tmp/zpt_var3_windows.csv")
VAR4 = Path("/tmp/zpt_var4_windows.csv")
PER_CYCLE = Path("/tmp/zpt_per_cycle.csv")


def main() -> None:
    pt = pd.read_csv(PER_TRIGGER)
    v3 = pd.read_csv(VAR3)
    v4 = pd.read_csv(VAR4)
    pc = pd.read_csv(PER_CYCLE)

    print("\n========= PER-CYCLE OVERVIEW =========")
    print(pc.to_string(index=False))

    print(f"\nTotal triggers: {len(pt)}  var3: {len(v3)}  var4: {len(v4)}")
    print(f"Mean var3/cycle: {pc['n_var3'].mean():.1f}  "
          f"max: {pc['n_var3'].max()}")
    print(f"Mean var4/cycle: {pc['n_var4'].mean():.1f}  "
          f"max: {pc['n_var4'].max()}")

    # ----- Hypothesis (a): POI-inner drift -----
    print("\n========= HYPOTHESIS (a) -- POI-inner drift =========")
    print("Per-cycle: how often does the sd-trigger zone_kind shift "
          "BOS -> POI as the cycle progresses?")
    for (sid, cycle), g in pt[pt["direction"] == "sd"].groupby(
        ["sid", "cycle_id"], sort=True,
    ):
        kinds = list(g["zone_kind"])
        first_poi_seq = next(
            (i for i, k in enumerate(kinds) if k == "POI"), None,
        )
        cnt = Counter(kinds)
        print(
            f"  sid={sid} cycle={cycle}: n_sd_trigs={len(g):2d} "
            f"BOS={cnt['BOS']:2d} POI={cnt['POI']:2d} "
            f"first_POI_at_seq={first_poi_seq}"
        )
    # Fraction of var4 triggers where this_sd_zone_kind == POI
    if not v4.empty:
        poi_var4 = (v4["this_sd_zone_kind"] == "POI").sum()
        print(
            f"\n  var4 sd triggers using POI: {poi_var4} / {len(v4)} "
            f"({poi_var4 / len(v4):.0%})"
        )

    # ----- Hypothesis (b): short window spans -----
    print("\n========= HYPOTHESIS (b) -- short window spans =========")
    if not v4.empty:
        print("  var4 window_span_total distribution (sd->opp->sd):")
        for thr in [3, 5, 8, 10, 15, 20]:
            cnt = (v4["window_span_total"] <= thr).sum()
            print(
                f"    spans <= {thr:2d} candles: {cnt:3d} / {len(v4)} "
                f"({cnt / len(v4):.0%})"
            )
        print(f"    median: {v4['window_span_total'].median():.0f}  "
              f"min: {v4['window_span_total'].min()}  "
              f"max: {v4['window_span_total'].max()}")
    if not v3.empty:
        print("\n  var3 window_span distribution (sd->cts):")
        for thr in [3, 5, 8, 10, 15, 20]:
            cnt = (v3["window_span"] <= thr).sum()
            print(
                f"    spans <= {thr:2d} candles: {cnt:3d} / {len(v3)} "
                f"({cnt / len(v3):.0%})"
            )
        print(f"    median: {v3['window_span'].median():.0f}  "
              f"min: {v3['window_span'].min()}  "
              f"max: {v3['window_span'].max()}")

    # ----- Hypothesis (c): apex never reaches CTS / BOS inner -----
    print("\n========= HYPOTHESIS (c) -- apex excursion shortfall =========")
    print("  var4: how far did the Lambda apex actually penetrate CTS inner? "
          "(positive = penetration, negative = short of CTS)")
    if not v4.empty:
        # Bins: deep penetration / mild / barely / short of inner
        for label, mask in [
            ("apex penetrates CTS by >=10p", v4["apex_to_cts_inner_pips"] >= 10),
            ("apex penetrates CTS 0-10p ", (v4["apex_to_cts_inner_pips"] >= 0) & (v4["apex_to_cts_inner_pips"] < 10)),
            ("apex within -10p..0 of CTS", (v4["apex_to_cts_inner_pips"] >= -10) & (v4["apex_to_cts_inner_pips"] < 0)),
            ("apex -20p..-10p (short)   ", (v4["apex_to_cts_inner_pips"] >= -20) & (v4["apex_to_cts_inner_pips"] < -10)),
            ("apex < -20p (very short)  ", v4["apex_to_cts_inner_pips"] < -20),
        ]:
            cnt = mask.sum()
            print(f"    {label}: {cnt:3d} / {len(v4)} ({cnt / len(v4):.0%})")
        print(
            f"    apex_to_cts_inner median: "
            f"{v4['apex_to_cts_inner_pips'].median():+.1f}p  "
            f"min: {v4['apex_to_cts_inner_pips'].min():+.1f}  "
            f"max: {v4['apex_to_cts_inner_pips'].max():+.1f}"
        )

    print("\n  var3: how far did the apex (toward BOS) actually penetrate "
          "BOS inner?")
    if not v3.empty:
        for label, mask in [
            ("apex penetrates BOS by >=10p", v3["apex_to_bos_inner_pips"] <= -10),
            ("apex penetrates BOS 0-10p ", (v3["apex_to_bos_inner_pips"] <= 0) & (v3["apex_to_bos_inner_pips"] > -10)),
            ("apex within 0..+10p of BOS", (v3["apex_to_bos_inner_pips"] > 0) & (v3["apex_to_bos_inner_pips"] <= 10)),
            ("apex +10..+20p (short)    ", (v3["apex_to_bos_inner_pips"] > 10) & (v3["apex_to_bos_inner_pips"] <= 20)),
            ("apex > +20p (very short)  ", v3["apex_to_bos_inner_pips"] > 20),
        ]:
            cnt = mask.sum()
            print(f"    {label}: {cnt:3d} / {len(v3)} ({cnt / len(v3):.0%})")
        print(
            f"    apex_to_bos_inner median: "
            f"{v3['apex_to_bos_inner_pips'].median():+.1f}p  "
            f"min: {v3['apex_to_bos_inner_pips'].min():+.1f}  "
            f"max: {v3['apex_to_bos_inner_pips'].max():+.1f}"
        )

    # ----- Lever simulation: trim with each lever and report counts -----
    print("\n========= LEVER SIMULATIONS =========")
    print("  How many var4 / var3 triggers survive each candidate lever?\n")

    if not v4.empty:
        print("  VAR 4 surviving counts under candidate gates:")
        for span_thr in [5, 8, 10, 15, 20]:
            cnt = (v4["window_span_total"] >= span_thr).sum()
            print(f"    Lever A: window_span_total >= {span_thr:2d} -> {cnt} "
                  f"(was {len(v4)})")
        for apex_pen in [-10, -5, 0, 5, 10]:
            cnt = (v4["apex_to_cts_inner_pips"] >= apex_pen).sum()
            print(f"    Lever B: apex_to_cts_inner_pips >= {apex_pen:+d}p -> "
                  f"{cnt}")
        # Lever C: drop POI-driven sd triggers (only BOS counts)
        cnt = (v4["this_sd_zone_kind"] == "BOS").sum()
        print(f"    Lever C-strict: this_sd_zone_kind == BOS -> {cnt}")

    if not v3.empty:
        print("\n  VAR 3 surviving counts under candidate gates:")
        for span_thr in [5, 8, 10, 15, 20]:
            cnt = (v3["window_span"] >= span_thr).sum()
            print(f"    Lever A: window_span >= {span_thr:2d} -> {cnt} "
                  f"(was {len(v3)})")
        for apex_pen in [10, 5, 0, -5, -10]:
            cnt = (v3["apex_to_bos_inner_pips"] <= apex_pen).sum()
            print(f"    Lever B: apex_to_bos_inner_pips <= {apex_pen:+d}p -> "
                  f"{cnt}")
        cnt = (v3["prior_sd_zone_kind"] == "BOS").sum()
        print(f"    Lever C-strict: prior_sd_zone_kind == BOS -> {cnt}")


if __name__ == "__main__":
    main()
