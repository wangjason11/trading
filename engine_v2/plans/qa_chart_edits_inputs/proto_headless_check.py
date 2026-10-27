"""Headless check of the chart control bar (item 5's final part, 2026-10-27): loads a chart HTML in headless Chrome
(choreographer, kaleido's driver), drives the controls from script and prints what each step did; saves screenshots.

    python engine_v2/plans/qa_chart_edits_inputs/proto_headless_check.py CHART_HTML SHOT_DIR STEPS_JSON

STEPS_JSON: a list of steps — ["idx", "<value>", "15M"|"1H"], ["in"], ["out"], ["wheel", -100, 0.5],
["xonly", p0, p1], ["parent"], ["pan"], ["zoom"] (the drag-mode switch), ["drag", f0, f1] (a mouse drag across the
plot at mid height, f = fractions of the plot width), ["shot", "<name>"], ["wait", ms].
"""
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import choreographer as choreo

HTML, SHOTS, STEPS = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

STATE_JS = r"""
(function () {
  var gd = document.querySelector('.plotly-graph-div'), xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
  var a = xa.r2l(xa.range[0]), b = xa.r2l(xa.range[1]), idx = [];
  gd._fullData.filter(function (t) { return t.type === 'candlestick'; }).forEach(function (fd) {
    for (var p = 0; p < fd.x.length; p++) { var t = xa.d2l(fd.x[p]); if (t >= a && t <= b) idx.push(fd.customdata[p][0]); }
  });
  idx.sort(function (u, v) { return u - v; });
  var tr = gd.data.filter(function (t) { return t.legendgroup === 'parent_overlay'; });
  var sh = (gd.layout.shapes || []).filter(function (s) { return s.legendgroup === 'parent_overlay'; });
  return JSON.stringify({
    candles_visible: idx.length, first_idx: idx[0], last_idx: idx[idx.length - 1],
    x: xa.range, y: ya.range.map(function (v) { return +v.toFixed(5); }),
    y2: gd._fullLayout.yaxis2 ? gd._fullLayout.yaxis2.range.map(Math.round) : null,
    parent_traces_hidden: tr.filter(function (t) { return t.visible === false; }).length + '/' + tr.length,
    parent_shapes_hidden: sh.filter(function (s) { return s.visible === false; }).length + '/' + sh.length,
    mark: (gd.layout.shapes || []).filter(function (s) { return s.name === 'fx_idx_mark'; }).length,
    drag: gd._fullLayout.dragmode + (gd._fullLayout.yaxis.fixedrange ? '/yfixed' : ''),
    msg: (document.getElementById('fxmsg') || {}).textContent
  });
})()
"""


async def ev(tab, expr):
    r = await tab.send_command("Runtime.evaluate", params={"expression": expr, "awaitPromise": True,
                                                           "returnByValue": True})
    res = r.get("result", {})
    if "exceptionDetails" in res:
        raise RuntimeError(json.dumps(res["exceptionDetails"])[:800])
    return res.get("result", {}).get("value")


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
        print("initial  ", await ev(tab, STATE_JS))
        for step in STEPS:
            kind = step[0]
            if kind == "idx":
                tf_js = (f"if (document.getElementById('fxtf')) document.getElementById('fxtf').value = '{step[2]}';"
                         if len(step) > 2 else "")
                await ev(tab, f"document.getElementById('fxidx').value = '{step[1]}'; {tf_js}"
                              "document.getElementById('fxgo').click(); 1")
            elif kind in ("in", "out", "parent", "pan", "zoom"):
                await ev(tab, f"document.getElementById('fx{kind}').click(); 1")
            elif kind == "wheel":
                await ev(tab, f"""(function () {{
                    var gd = document.querySelector('.plotly-graph-div'), s = gd._fullLayout._size,
                        r = gd.getBoundingClientRect();
                    gd.dispatchEvent(new WheelEvent('wheel', {{deltaY: {step[1]}, bubbles: true, cancelable: true,
                        clientX: r.left + s.l + s.w * {step[2]}, clientY: r.top + s.t + s.h * 0.5}}));
                    return 1; }})()""")
            elif kind == "xonly":
                await ev(tab, f"""(function () {{
                    var gd = document.querySelector('.plotly-graph-div'), xa = gd._fullLayout.xaxis;
                    var fd = gd._fullData.filter(function (t) {{ return t.type === 'candlestick'; }})[0];
                    Plotly.relayout(gd, {{'xaxis.range[0]': fd.x[{step[1]}], 'xaxis.range[1]': fd.x[{step[2]}]}});
                    return 1; }})()""")
            elif kind == "drag":
                box = json.loads(await ev(tab, """(function () { var gd = document.querySelector('.plotly-graph-div'),
                    s = gd._fullLayout._size, r = gd.getBoundingClientRect();
                    return JSON.stringify([r.left + s.l, r.top + s.t, s.w, s.h]); })()"""))
                y = box[1] + box[3] * 0.4
                xs = [box[0] + box[2] * (step[1] + (step[2] - step[1]) * k / 10) for k in range(11)]
                await tab.send_command("Input.dispatchMouseEvent", params={
                    "type": "mousePressed", "x": xs[0], "y": y, "button": "left", "clickCount": 1})
                for x in xs[1:]:
                    await tab.send_command("Input.dispatchMouseEvent", params={
                        "type": "mouseMoved", "x": x, "y": y, "button": "left", "buttons": 1})
                    await asyncio.sleep(0.03)
                await tab.send_command("Input.dispatchMouseEvent", params={
                    "type": "mouseReleased", "x": xs[-1], "y": y, "button": "left", "clickCount": 1})
            elif kind == "wait":
                await asyncio.sleep(step[1] / 1000)
                continue
            elif kind == "shot":
                shot = await tab.send_command("Page.captureScreenshot", params={"format": "png"})
                path = os.path.join(SHOTS, step[1] + ".png")
                open(path, "wb").write(base64.b64decode(shot["result"]["data"]))
                print("shot     ", path)
                continue
            await asyncio.sleep(0.9)
            print(f"{' '.join(map(str, step)):9s}", await ev(tab, STATE_JS))


asyncio.run(main())
