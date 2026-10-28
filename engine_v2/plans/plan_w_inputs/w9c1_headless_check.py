"""PLAN_W 9c-1 — the `Sub CTS zones` button in a real (headless) browser: loads an M15 chart HTML (choreographer,
kaleido's driver — as `qa_chart_edits_inputs/proto_headless_check.py`), centres it on an idx, and prints per step the
button's text + class and how many `sub_cts_zones` / `parent_overlay` traces and shapes are hidden; screenshots each
state.

    python engine_v2/plans/plan_w_inputs/w9c1_headless_check.py CHART_HTML SHOT_DIR IDX
"""
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import choreographer as choreo

HTML, SHOTS, IDX = sys.argv[1], sys.argv[2], sys.argv[3]

STATE_JS = r"""
(function () {
  var gd = document.querySelector('.plotly-graph-div');
  function count(group) {
    var tr = gd.data.filter(function (t) { return t.legendgroup === group; });
    var sh = (gd.layout.shapes || []).filter(function (s) { return s.legendgroup === group; });
    return {traces_hidden: tr.filter(function (t) { return t.visible === false; }).length + '/' + tr.length,
            shapes_hidden: sh.filter(function (s) { return s.visible === false; }).length + '/' + sh.length};
  }
  var b = document.getElementById('fxsubcts'), p = document.getElementById('fxparent');
  return JSON.stringify({sub_cts_button: b ? b.textContent + ' [' + b.className + ']' : null,
                         parent_button: p ? p.textContent : null,
                         sub_cts: count('sub_cts_zones'), parent: count('parent_overlay')});
})()
"""


async def ev(tab, expr):
    r = await tab.send_command("Runtime.evaluate", params={"expression": expr, "awaitPromise": True,
                                                           "returnByValue": True})
    res = r.get("result", {})
    if "exceptionDetails" in res:
        raise RuntimeError(json.dumps(res["exceptionDetails"])[:800])
    return res.get("result", {}).get("value")


async def shot(tab, name):
    s = await tab.send_command("Page.captureScreenshot", params={"format": "png"})
    path = os.path.join(SHOTS, name + ".png")
    open(path, "wb").write(base64.b64decode(s["result"]["data"]))
    print("shot", path)


async def main():
    os.makedirs(SHOTS, exist_ok=True)
    async with choreo.Browser(headless=True) as browser:
        tab = await browser.create_tab(Path(HTML).resolve().as_uri())
        await tab.send_command("Emulation.setDeviceMetricsOverride",
                               params={"width": 1600, "height": 900, "deviceScaleFactor": 1, "mobile": False})
        for _ in range(100):
            await asyncio.sleep(0.2)
            try:
                if await ev(tab, "!!document.getElementById('fxctl')"):
                    break
            except Exception:
                pass
        else:
            print("control bar never appeared")
            return
        await asyncio.sleep(0.8)
        await ev(tab, f"document.getElementById('fxidx').value = '{IDX}'; document.getElementById('fxgo').click(); 1")
        await asyncio.sleep(0.9)
        print("initial      ", await ev(tab, STATE_JS))
        await shot(tab, "1_hidden")
        for name in ("2_shown", "3_hidden_again"):
            await ev(tab, "document.getElementById('fxsubcts').click(); 1")
            await asyncio.sleep(0.9)
            print("click subcts ", await ev(tab, STATE_JS))
            await shot(tab, name)
        await ev(tab, "document.getElementById('fxparent').click(); 1")
        await asyncio.sleep(0.9)
        print("click parent ", await ev(tab, STATE_JS))


asyncio.run(main())
