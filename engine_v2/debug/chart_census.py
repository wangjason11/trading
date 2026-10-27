"""Per-element / per-sub trace census of a saved M15 chart .html.

Usage: PYTHONPATH=. python engine_v2/debug/chart_census.py <chart.html> [<chart_b.html>]
With two files it prints a side-by-side census diff.
"""
import json, re, sys
from collections import Counter


def load_fig(path):
    html = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r'Plotly\.newPlot\(', html)
    i = html.index('[', m.end())
    dec = json.JSONDecoder()
    data, end = dec.raw_decode(html, i)
    j = html.index('{', end)
    layout, _ = dec.raw_decode(html, j)
    return data, layout


# A sub hover's identity, every format to ONE key `(parent_sid, parent_cycle, sub_id)` (strings, like the trace-name
# keys) so a census diff across a hover-format change compares equal: since the user's hover specs (dots 2026-10-22,
# KL / POI / fib / wave candles 2026-10-23) `sub_id=X | cycle_id=Y` + `first record: lens=… | parent_sid=P |
# parent_cycle_id=C`; before them positional customdata ints (KL / POI / dots) or the wave candle's `first record
# parent_sid=P parent_cycle=C` line (unmatched since the 2026-09-30 `Role:` line → `?` until 2026-10-23).
_FIRST_RECORD = re.compile(r"first record: lens=\w+ \| parent_sid=([^ |<]+) \| parent_cycle_id=([^ |<]+)")
_SUB_ID = re.compile(r"sub_id=(\d+) \| cycle_id=")
_OLD_WAVE = re.compile(r"sub_id=(\d+) cycle=-?\d+<br>Role: [^<]*<br>first record parent_sid=([^ <]+) "
                       r"parent_cycle=([^ <]+)")


def _key(p_sid, p_cycle, sub_id):
    return (str(p_sid), str(p_cycle), str(sub_id))


def _key_from_lines(text):
    fr, sub = _FIRST_RECORD.search(text), _SUB_ID.search(text)
    return _key(fr.group(1), fr.group(2), sub.group(1)) if fr and sub else "?"


def _zone_key(r, old_p_sid_slot):
    """KL / POI: the identity lines in slot 1 since 2026-10-23, else `(r[slot], r[slot + 1], r[1])`."""
    if isinstance(r[1], str) and r[1].startswith("sub_id="):
        return _key_from_lines(r[1])
    return _key(r[old_p_sid_slot], r[old_p_sid_slot + 1], r[1])


def census(path):
    data, layout = load_fig(path)
    c = Counter()
    for tr in data:
        name = tr.get("name") or ""
        ht = tr.get("hovertemplate") or ""
        cd = tr.get("customdata") or []
        if name == "M15 KL zone":
            c[("KL_hover", _zone_key(cd[0], 9))] += 1
        elif name == "M15 POI zone":
            c[("POI_hover", _zone_key(cd[0], 10))] += 1
        elif name.startswith("M15 KL outline"):
            mm = re.search(r'h1s(\S+)c(\S+)_sub(\d+)', name)
            c[("KL_outline", (mm.group(1), mm.group(2), mm.group(3)) if mm else name)] += 1
        elif re.match(r'^M15 (CTS \(unconf\)|CTS|BOS|PB) ', name):
            kind = re.match(r'^M15 (CTS \(unconf\)|CTS|BOS|PB) ', name).group(1)
            r = cd[0]
            fr = _FIRST_RECORD.search(str(r[6]))           # the `first record:` line since 2026-10-22
            c[(f"dots_{kind}", _key(fr.group(1), fr.group(2), r[3]) if fr else _key(r[6], r[7], r[3]))] += 1
        elif name.startswith("M15 swing"):
            mm = re.search(r'h1s(\S+)c(\S+)_m15s(\d+)', name)
            c[("swing", (mm.group(1), mm.group(2), mm.group(3)) if mm else name)] += 1
        elif name.startswith("M15 PB"):
            c[("pb_to_bos", name)] += 1
        elif name.startswith("M15 Prev BOS"):
            c[("prev_bos", name)] += 1
        elif not name and ht.startswith("TF=15M<br><b>Wave Candle"):
            old = _OLD_WAVE.search(ht)
            c[("wave_hover_M15", _key(old.group(2), old.group(3), old.group(1)) if old else _key_from_lines(ht))] += 1
        elif not name and ht.startswith("TF=1H<br><b>Wave Candle"):
            c[("wave_hover_H1", "-")] += 1
        elif "WVMI LP " in name:      # one trace per side, one mark per per-LP row (CHARTING_SPEC §13)
            c[("wvmi_lp_marks", name)] += len(cd)
        elif name.startswith("H1 ") or name in ("zone_proximity:trigger",):
            c[("H1_overlay", name.split(" sid=")[0].split(" (sid")[0])] += 1
        else:
            c[("base", name)] += 1
    shapes = layout.get("shapes", []) or []
    for sh in shapes:
        if sh.get("type") == "line" and sh.get("yref") == "paper":
            c[("shape_wave_vline", "-")] += 1
        elif sh.get("type") == "line":
            c[("shape_confirm_line", (sh.get("line") or {}).get("color"))] += 1
        elif sh.get("type") == "rect":
            fc = sh.get("fillcolor")
            c[("shape_rect_outline" if fc == "rgba(0,0,0,0)" else "shape_rect_fill", fc)] += 1
        else:
            c[("shape_other", sh.get("type"))] += 1
    c[("TOTAL_TRACES", "-")] = len(data)
    c[("TOTAL_SHAPES", "-")] = len(shapes)
    return c


cs = [census(p) for p in sys.argv[1:]]
keys = sorted(set().union(*[set(c) for c in cs]), key=lambda k: (str(k[0]), str(k[1])))
if len(cs) == 1:
    for k in keys:
        print(f"{cs[0][k]:6d}  {k[0]:20s} {k[1]}")
else:
    print(f"{'A':>6} {'B':>6} {'d':>6}  element / (parent_sid,parent_cycle,sub_id)")
    for k in keys:
        a, b = cs[0][k], cs[1][k]
        flag = "" if a == b else "   <<<"
        print(f"{a:6d} {b:6d} {b-a:6d}  {k[0]:20s} {k[1]}{flag}")
