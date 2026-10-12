"""PLAN_M — the TRIAL replay's behaviour patch: a COUNTER sub (its direction opposite its first record's parent)
takes `P` = the parent cycle's BOS threshold, source `parent_bos_threshold`, the anchor behind it = the parent's
`BOS_CONFIRMED(S, C)` price (the IC reference's `cross_limit_anchor` + the cross-limit breakout watch on `P`).

TRIAL ONLY: `P` is the parent's `BOS_CONFIRMED(S, C)` price and the patch ASSERTS that the parent cycle has no
`BOS_THRESHOLD_UPDATED` — true on the reference window (0 on H1), so the live reading (PLAN_M §1 items 2–3) gives
the same constant `P` there. No freeze-at-establishment, no steps, no export changes beyond the source label.

    python counter_limit_trial_patch.py TREE      (TREE = a `git archive` of the base; each edit must match once)

Then, in a SHORT %TEMP% dir holding a copy of oanda.cfg (MAX_PATH): `python REPO/engine_v2/plans/plan_i_inputs/
replay_variant_i.py TREE log.json > run.log 2>&1`, and compare against S_L2 (PLAN_M §5).
"""
import sys
from pathlib import Path

TREE = Path(sys.argv[1])


def patch(rel, edits):
    p = TREE / rel
    src = p.read_text(encoding="utf-8")
    for old, new in edits:
        n = src.count(old)
        assert n == 1, (rel, n, old[:100])
        src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")


patch("engine_v2/zones/cross_cycle_fib.py", [(
    '    "parent_cts_threshold",     # first_confluence on parent cycle c >= 1: the parent\'s (S, c) `cts_frozen`\n',
    '    "parent_cts_threshold",     # first_confluence on parent cycle c >= 1: the parent\'s (S, c) `cts_frozen`\n'
    '    "parent_bos_threshold",     # TRIAL (PLAN_M): a counter sub — the first record\'s parent cycle BOS threshold\n',
)])

patch("engine_v2/multitf/cross_fib_limit.py", [(
    '    if trigger_type == "reversal":\n'
    '        assert reversing_sub is not None, "[cross_fib_limit] a reversal needs its reversing sub"\n',
    '    h1_events = list(h1_events)\n'
    '    S, C = parent_cycle\n'
    '    parent_sd = {int(e.meta["struct_direction"]) for e in h1_events\n'
    '                 if e.type == "CTS_ESTABLISHED" and int(e.meta["structure_id"]) == int(S)}\n'
    '    assert len(parent_sd) == 1, ("[trial] parent direction", S, parent_sd)\n'
    '    if int(direction) != parent_sd.pop():\n'
    '        bos = [e for e in h1_events if e.type == "BOS_CONFIRMED"\n'
    '               and int(e.meta["structure_id"]) == int(S) and int(e.meta["cycle_id"]) == int(C)]\n'
    '        assert len(bos) == 1, ("[trial] parent BOS", S, C, len(bos))\n'
    '        moves = [e for e in h1_events if e.type == "BOS_THRESHOLD_UPDATED"\n'
    '                 and int(e.meta["structure_id"]) == int(S) and int(e.meta["cycle_id"]) == int(C)]\n'
    '        assert not moves, ("[trial] the parent BOS threshold moved — the trial assumes it never does", S, C)\n'
    '        price = float(bos[0].price)\n'
    '        print(f"[trial] counter sub dir {int(direction)} {trigger_type} parent ({S},{C}) at {s}: P={price}")\n'
    '        return CrossFibLimit(price, "parent_bos_threshold", ref_anchor_price=price)\n'
    '    if trigger_type == "reversal":\n'
    '        assert reversing_sub is not None, "[cross_fib_limit] a reversal needs its reversing sub"\n',
)])
print("patched", TREE)
