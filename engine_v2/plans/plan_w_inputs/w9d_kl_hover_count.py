"""PLAN_W §4 — the KL zone hover traces per chart of a save (or a charts dir): every trace whose hovertemplate is the
KL zone hover (`KL Zone<br>`), by name, and how many carry a CTS zone (a hidden `sub_cts_zones` trace).

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9d_kl_hover_count.py DIR
"""
import collections
import glob
import os
import sys

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.html"))):
    data, _ = load_fig(p)
    kl = [t for t in data if "KL Zone<br>" in (t.get("hovertemplate") or "")]
    print(f"{os.path.basename(p)}: {len(kl)} KL hover traces of {len(data)} — "
          f"{dict(collections.Counter(t.get('name') for t in kl))}; in sub_cts_zones: "
          f"{sum(1 for t in kl if t.get('legendgroup') == 'sub_cts_zones')}")
