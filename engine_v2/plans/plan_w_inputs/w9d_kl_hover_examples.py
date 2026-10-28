"""PLAN_W §4 — the KL zone hover before the `kind=` line, rendered as text for one BOS and one CTS zone of the H1 chart (S_W9c), with the
proposed kind line in each option's place. The hover = `kl_zone_hovertemplate("1H")` filled with the trace's customdata.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9d_kl_hover_examples.py SAVE_DIR [OUT.txt]
"""
import glob
import re
import sys

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
from engine_v2.charting._zone_render import kl_zone_hovertemplate  # noqa: E402
sys.argv = _argv

save = sys.argv[1]
html = [p for p in glob.glob(save + "/*.html") if "_M15_" not in p][0]
data, _ = load_fig(html)
tmpl = kl_zone_hovertemplate("1H")


def render(cd):
    def sub(m):
        i, fmt = int(m.group(1)), m.group(2)
        v = cd[i]
        return format(float(v), fmt[1:]) if fmt else str(v)
    txt = re.sub(r"%\{customdata\[(\d+)\](:[^}]*)?\}", sub, tmpl)
    return txt.replace("<extra></extra>", "").replace("<br>", "\n").replace("<b>", "").replace("</b>", "")


seen = {}
for t in data:
    if t.get("name") != "KL zone" or not t.get("customdata"):
        continue
    cd = t["customdata"][0]
    side = cd[0]
    if side not in seen:
        seen[side] = cd
    if len(seen) == 2:
        break
out = []
for side, cd in seen.items():
    kind = "BOS" if (side == "buy") == (int(cd[9]) == 1) else "CTS"
    out.append(f"--- today ({kind} zone; the kind is readable only from side={side} vs struct_direction={cd[9]}):")
    out.append(render(cd))
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write(text)
