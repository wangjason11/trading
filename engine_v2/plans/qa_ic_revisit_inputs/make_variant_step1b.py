"""Patch a `git archive` copy of the tree (argv[1]) with Step 1B's BEHAVIOUR only (PLAN_S step 3 / change C; the
final spec `qa_findings.md` round 5) — a VARIANT replay that measures the full cascade before the build. No new field,
CSV column, meta key or hover: the landed build must equal this variant on every existing value.

`select_ic`, after Step 1 / Step 1A found the IC candidates: each candidate's closest prior imbalance = the sd
instance whose c2 run starts at / after the fib's BOS anchor and ends before the candidate, the one ending latest; the
candidate PASSES when that gap is not filled through the candle before it (`is_filled(df, IC − 1)`); no prior gap ->
fails. Passers -> Steps 2–3; no IC from them -> Steps 2–3 on all the candidates; no passer -> all the candidates.
One edit, exact-count (CRLF-aware).

    git archive HEAD | tar -x -C <short %TEMP% dir>; python <this> <dir>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def patch(rel, pairs):
    p = ROOT / rel
    b = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in b else "\n"  # the archive may carry CRLF (Windows checkout)
    for old, new, n in pairs:
        old, new = old.replace("\n", nl), new.replace("\n", nl)
        assert b.count(old) == n, (rel, old, b.count(old))
        b = b.replace(old, new)
    p.write_bytes(b.encode("utf-8"))


OLD = '''    pick, tiebreak = select_institutional_pattern(df, fib_state, config, candidate_checks)     # Step 2
    if pick is not None:
        return ICSelection(checks, pick, tiebreak, ip_outcome=IP_OUTCOME_PATTERN, ranking=None, no_ic_reason=None,
                           ic_zone_band=band)
    pick, ranking = apply_ranking_rules(df, fib_state, config, candidate_checks)               # Step 3
    return ICSelection(checks, pick, None, ip_outcome=IP_OUTCOME_NO_PATTERN, ranking=ranking,
                       no_ic_reason=None if pick is not None else ALL_IC_CANDIDATES_PINBAR, ic_zone_band=band)
'''

NEW = '''    passers = tuple(c for c in candidate_checks if _variant_passes_1b(df, int(c.ic_idx), fib_state, config))
    if passers:
        sel = _variant_steps_2_3(df, fib_state, config, passers, checks, band)
        if sel.pick is not None:
            return sel
    return _variant_steps_2_3(df, fib_state, config, candidate_checks, checks, band)


def _variant_steps_2_3(df, fib_state, config, cands, checks, band):
    pick, tiebreak = select_institutional_pattern(df, fib_state, config, cands)     # Step 2
    if pick is not None:
        return ICSelection(checks, pick, tiebreak, ip_outcome=IP_OUTCOME_PATTERN, ranking=None, no_ic_reason=None,
                           ic_zone_band=band)
    pick, ranking = apply_ranking_rules(df, fib_state, config, cands)               # Step 3
    return ICSelection(checks, pick, None, ip_outcome=IP_OUTCOME_NO_PATTERN, ranking=ranking,
                       no_ic_reason=None if pick is not None else ALL_IC_CANDIDATES_PINBAR, ic_zone_band=band)


def _variant_passes_1b(df, ic_idx, fib_state, config):
    sd = int(fib_state.struct_direction)
    best = None
    for inst in df.attrs.get("imbalances", []):
        if inst.direction != sd or inst.end_idx >= ic_idx or inst.start_idx < fib_state.bos_idx:
            continue
        if best is None or inst.end_idx > best.end_idx:
            best = inst
    return best is not None and not best.is_filled(df, ic_idx - 1, config.fill_threshold)
'''

patch("engine_v2/zones/poi_zones.py", [(OLD, NEW, 1)])
print("patched: Step 1B in select_ic (behaviour only)")
