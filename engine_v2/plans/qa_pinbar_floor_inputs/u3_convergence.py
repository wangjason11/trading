"""PLAN_U U3, before building the per-run convergence check (§8 "For U3": "recompute with the seed's first window of
pinbars dropped, raise if a frame floor differs"): would it pass on this window, and how far back must the final-mode
history start for the frame's floors to stop depending on the start?

For each TF, the replay's own seed (`fetch_floor_seeds` — the guard's 2 steps; the `u3_causes.py` cache) + the frame,
final mode (`rolling_pinbar_body_floor`, the engine's function on the seed + frame body-% types): the frame's floors
and types with the FULL seed vs with the seed's first k windows of body-% pinbars dropped (the seed cut just after its
(k × 90)-th body-% pinbar), k = 1, 2, …; each line: the pinbars left in the seed, the frame candles whose floor /
type differ, the first and last such frame idx. Also: where the two final-mode runs (full vs k = 1) converge — the
first candle (seed or frame) from which their windows hold the same rows.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_convergence.py [OUT]
"""
from __future__ import annotations

import sys

import _frames  # noqa: F401  (sys.path)
import numpy as np

from _frames import HERE
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.pinbar_body_floor import _body_pct_types, rolling_pinbar_body_floor
from u3_causes import frames

OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "u3_convergence.txt")
P = CandleParams(pinbar_floor_window_source="final")
W = P.pinbar_floor_window


def windows_rows(body, btype, demoted, n):
    """The row set of each candle's window (final membership), for the first n rows — as a list of frozensets."""
    out, win = [], []
    for i in range(n):
        out.append(frozenset(win))
        if btype[i] == "pinbar" or demoted[i]:
            win.append(i)
            if len(win) > W:
                win.pop(0)
    return out


def converge(sb, st, fb, ft):
    """Full seed vs the seed minus its first window, final mode: (frame floors differing, frame types differing, where
    the two runs' windows start holding the same rows — 'seed row r' = converged before the frame, for ANY frame)."""
    pin_rows = np.flatnonzero(st == "pinbar")
    s = len(sb)
    full_f, full_d = rolling_pinbar_body_floor(np.r_[sb, fb], np.r_[st, ft], P)
    cut = pin_rows[W - 1] + 1
    f2, d2 = rolling_pinbar_body_floor(np.r_[sb[cut:], fb], np.r_[st[cut:], ft], P)
    ff, g = full_f[s:], f2[s - cut:]
    nf = int((~((ff == g) | (np.isnan(ff) & np.isnan(g)))).sum())
    nt = int((full_d[s:] != d2[s - cut:]).sum())
    n = s + len(fb)
    a = windows_rows(np.r_[sb, fb], np.r_[st, ft], full_d, n)
    b_rows = windows_rows(np.r_[sb[cut:], fb], np.r_[st[cut:], ft], d2, n - cut)
    same = [a[i + cut] == frozenset(r + cut for r in b_rows[i]) for i in range(n - cut)]
    first_same = next((i for i in range(len(same)) if all(same[i:])), None)
    where = "never" if first_same is None else (
        f"seed row {first_same + cut} of {s}" if first_same + cut < s else f"frame idx {first_same + cut - s}")
    return nf, nt, where, full_f[s:]


def longer_seeds(raws, seeds, lines):
    """The engine's guard (`run_replay.fetch_floor_seed`) at 3 / 4 / 5 windows: its steps, and the drop-1 check on it;
    the frame floors vs the replay's 2-window seed (equal = a longer seed changes nothing on this window)."""
    from engine_v2.config import CONFIG
    from engine_v2.run_replay import fetch_floor_seed
    first = raws["H1"]["time"].iloc[0]
    lines.append("## Longer seeds (the engine's guard, 400-candle steps): the drop-1 check on each")
    for tf in ("H1", "M15"):
        fr = _body_pct_types(raws[tf], P)
        fb, ft = fr["body_len"].to_numpy(float), fr["candle_type"].to_numpy()
        sd2 = _body_pct_types(seeds[tf], P)
        base = rolling_pinbar_body_floor(np.r_[sd2["body_len"].to_numpy(float), fb],
                                         np.r_[sd2["candle_type"].to_numpy(), ft], P)[0][len(sd2):]
        for k in (2, 3, 4, 5):
            seed = fetch_floor_seed(CONFIG.pair, tf, first, min_pinbars=k * W)
            sd = _body_pct_types(seed, P)
            sb, st = sd["body_len"].to_numpy(float), sd["candle_type"].to_numpy()
            nf, nt, where, floors = converge(sb, st, fb, ft)
            same_as_2 = bool(np.array_equal(floors, base, equal_nan=True))
            lines.append(f"   {tf} {k} windows ({k * W} pinbars): seed {len(sb)} candles, {int((st == 'pinbar').sum())} "
                         f"body-% pinbars, from {str(seed['time'].iloc[0])[:16]} -> drop-1: frame floors differ on {nf}, "
                         f"types on {nt}; windows the same from {where}; frame floors == the 2-window seed's: {same_as_2}")
    lines.append("")


def main():
    raws, seeds = frames()
    lines = [f"final mode, window {W}, p{P.pinbar_floor_percentile:g}; the seed = the replay's (2 guard steps)", ""]
    for tf in ("H1", "M15"):
        sd = _body_pct_types(seeds[tf], P)
        fr = _body_pct_types(raws[tf], P)
        sb, st = sd["body_len"].to_numpy(float), sd["candle_type"].to_numpy()
        fb, ft = fr["body_len"].to_numpy(float), fr["candle_type"].to_numpy()
        pin_rows = np.flatnonzero(st == "pinbar")
        full_f, full_d = rolling_pinbar_body_floor(np.r_[sb, fb], np.r_[st, ft], P)
        s = len(sb)
        ff, fd = full_f[s:], full_d[s:]
        lines.append(f"## {tf}: seed {len(sb)} candles, {len(pin_rows)} body-% pinbars; frame {len(fb)} candles")
        k = 1
        while k * W <= len(pin_rows):
            cut = pin_rows[k * W - 1] + 1          # drop the seed's rows up to its (k*W)-th body-% pinbar
            f2, d2 = rolling_pinbar_body_floor(np.r_[sb[cut:], fb], np.r_[st[cut:], ft], P)
            g, gd = f2[s - cut:], d2[s - cut:]
            df = np.flatnonzero(~((ff == g) | (np.isnan(ff) & np.isnan(g))))
            dt = np.flatnonzero(fd != gd)
            lines.append(f"   drop {k} window(s): {len(pin_rows) - k * W} body-% pinbars left in the seed -> frame floors "
                         f"differ on {len(df)} candles" + (f" (idx {df[0]} .. {df[-1]})" if len(df) else "")
                         + f", types on {len(dt)}" + (f" (idx {', '.join(map(str, dt[:12]))})" if len(dt) else ""))
            if k == 1:
                n = s + len(fb)
                a = windows_rows(np.r_[sb, fb], np.r_[st, ft], full_d, n)
                b_rows = windows_rows(np.r_[sb[cut:], fb], np.r_[st[cut:], ft], d2, n - cut)
                same = [a[i + cut] == frozenset(r + cut for r in b_rows[i]) for i in range(n - cut)]
                first_same = next((i for i in range(len(same)) if all(same[i:])), None)
                where = "never" if first_same is None else (
                    f"seed row {first_same + cut}" if first_same + cut < s else f"frame idx {first_same + cut - s}")
                lines.append(f"      full vs drop-1: the windows hold the same rows from {where} on")
            k += 1
        lines.append("")
    longer_seeds(raws, seeds, lines)
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
