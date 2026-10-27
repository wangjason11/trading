(function initChartControls() {
    // Item 5's final part — PROTOTYPE (2026-10-27): the chart control bar above the plot.
    //   Parent (1H) button (M15 charts only): flips every trace + shape tagged legendgroup == CFG.group.
    //   - / + : the time axis only, halve / double the visible candles about the centre.
    //   wheel  : the same about the cursor (inside the plot area only; outside it the page scrolls).
    //   Box zoom | Pan: plotly's dragmode; a pan moves the TIME axis only (`yaxis.fixedrange` while panning), the
    //          toolbar's Pan icon included (the switch tracks it) — round 2. Plotly pans by CLOCK time, so across a
    //          rangebreak (weekend / holiday) the window shrank or stuck; on release the window is re-set to the same
    //          candle count shifted by the candles dragged across (plotly's live drag kept for the motion).
    //   idx box (+ a 15M / 1H selector on the M15 charts): centre on the candle(s), +/- CFG.half candles, a mark that
    //          stays until the next jump; Go / Enter on an EMPTY box clears it (round 2).
    //   The price axis re-fits to the visible candles after every time-only change (these controls, a pan on release,
    //   an x-only drag-zoom, an x-axis drag); a box zoom keeps the price range it set.
    var CFG = __CFG__;
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullData || !window.Plotly) {
        setTimeout(initChartControls, 200);
        return;
    }
    // plotly REPLACES `_fullLayout.xaxis` on a full relayout (e.g. a `shapes` update) - never cache it.
    function X() { return gd._fullLayout.xaxis; }
    // The candles by POSITION (the rangebreaks space them evenly): time (plotly's linear ms), idx, idx_1H, low, high —
    // from EVERY candlestick trace ("OHLC" + the imbalance candles "imbalance:bullish" / "imbalance:bearish"; one
    // customdata layout: [0] idx, [12] idx_1H on the M15 charts), sorted by time.
    var rows = [];
    for (var i = 0; i < gd._fullData.length; i++) {
        var fd = gd._fullData[i];
        if (fd.type !== 'candlestick') continue;
        for (var r = 0; r < fd.x.length; r++) {
            var cd = fd.customdata[r];
            rows.push([X().d2l(fd.x[r]), cd[0], cd.length > 12 ? cd[12] : null, +fd.low[r], +fd.high[r]]);
        }
    }
    if (!rows.length) return;
    rows.sort(function (a, b) { return a[0] - b[0]; });
    var N = rows.length, T = [], IDX = [], IDXH = [], LO = [], HI = [];
    for (var p = 0; p < N; p++) {
        T.push(rows[p][0]); IDX.push(rows[p][1]); IDXH.push(rows[p][2]); LO.push(rows[p][3]); HI.push(rows[p][4]);
    }
    var STEP = Infinity;
    for (p = 1; p < N; p++) { var d = T[p] - T[p - 1]; if (d > 0 && d < STEP) STEP = d; }

    function lowerBound(v) {
        var lo = 0, hi = N;
        while (lo < hi) { var m = (lo + hi) >> 1; if (T[m] < v) lo = m + 1; else hi = m; }
        return lo;
    }
    function visible() {
        var xa = X(), a = xa.r2l(xa.range[0]), b = xa.r2l(xa.range[1]);
        var p0 = Math.min(N - 1, lowerBound(a)), p1 = Math.max(0, lowerBound(b + 1) - 1);
        return p1 < p0 ? [p0, p0] : [p0, p1];
    }
    function priceRange(p0, p1) {
        var lo = Infinity, hi = -Infinity;
        for (var q = p0; q <= p1; q++) { if (LO[q] < lo) lo = LO[q]; if (HI[q] > hi) hi = HI[q]; }
        var pad = (hi - lo) * CFG.pad;
        if (!(pad > 0)) pad = Math.abs(hi) * 1e-4;
        return [lo - pad, hi + pad];
    }
    function show(p0, p1) {
        var n = p1 - p0;
        if (p0 < 0) { p0 = 0; p1 = n; }
        if (p1 > N - 1) { p1 = N - 1; p0 = p1 - n; }
        p0 = Math.max(0, Math.round(p0));
        p1 = Math.min(N - 1, Math.round(p1));
        return Plotly.relayout(gd, {
            'xaxis.range': [X().l2r(T[p0] - STEP / 2), X().l2r(T[p1] + STEP / 2)],
            'yaxis.range': priceRange(p0, p1)
        });
    }
    function zoom(factor, frac) {
        var v = visible(), n = v[1] - v[0];
        var n2 = Math.max(CFG.minN, Math.min(N - 1, Math.round(n * factor)));
        if (n2 === n) return;
        var anchor = v[0] + frac * n;
        show(anchor - frac * n2, anchor - frac * n2 + n2);
    }

    // --- the bar ---
    var bar = document.createElement('div');
    bar.id = 'fxctl';
    bar.innerHTML = (
        '<style>#fxctl{font:13px "Segoe UI",Arial,sans-serif;display:flex;gap:6px;align-items:center;padding:6px 10px;}' +
        '#fxctl button,#fxctl select{font:inherit;padding:2px 9px;border:1px solid #999;border-radius:4px;' +
        'background:#f4f4f4;cursor:pointer;}#fxctl button:hover{background:#e6e6e6;}' +
        '#fxctl input{width:72px;font:inherit;padding:2px 5px;border:1px solid #999;border-radius:4px;}' +
        '#fxctl .sep{width:1px;height:20px;background:#ccc;margin:0 6px;}#fxctl .msg{color:#a33;margin-left:6px;}' +
        '#fxctl .off,#fxctl .on{background:#555;color:#fff;border-color:#555;}</style>' +
        (CFG.group ? '<button id="fxparent" title="Show / hide the parent (1H) overlay">Parent (1H): shown</button>' +
                     '<span class="sep"></span>' : '') +
        '<button id="fxout" title="Zoom out (time axis)">&minus;</button>' +
        '<button id="fxin" title="Zoom in (time axis)">+</button>' +
        '<span class="sep"></span>' +
        '<button id="fxzoom" title="Drag draws a box to zoom into">Box zoom</button>' +
        '<button id="fxpan" title="Drag slides the time window (same width)">Pan</button>' +
        '<span class="sep"></span>' +
        '<label for="fxidx">idx</label><input id="fxidx" type="text" inputmode="numeric">' +
        (CFG.tf === '15M' ? '<select id="fxtf"><option value="15M">15M</option><option value="1H">1H</option></select>' : '') +
        '<button id="fxgo">Go</button><span class="msg" id="fxmsg"></span>'
    );
    gd.parentNode.insertBefore(bar, gd);
    var $ = function (id) { return document.getElementById(id); };
    function msg(t) { $('fxmsg').textContent = t || ''; }

    // --- the parent (1H) toggle ---
    if (CFG.group) {
        var parentOn = true;
        $('fxparent').onclick = function () {
            parentOn = !parentOn;
            var tr = [], lay = {};
            for (var k = 0; k < gd.data.length; k++) if (gd.data[k].legendgroup === CFG.group) tr.push(k);
            (gd.layout.shapes || []).forEach(function (s, k2) {
                if (s.legendgroup === CFG.group) lay['shapes[' + k2 + '].visible'] = parentOn;
            });
            Plotly.update(gd, {visible: parentOn}, lay, tr);
            this.textContent = 'Parent (1H): ' + (parentOn ? 'shown' : 'hidden');
            this.className = parentOn ? '' : 'off';
        };
    }

    // --- +/- and the wheel ---
    $('fxin').onclick = function () { msg(); zoom(0.5, 0.5); };
    $('fxout').onclick = function () { msg(); zoom(2, 0.5); };
    var wheelAcc = 0, wheelFrac = 0.5, wheelPending = false;
    gd.addEventListener('wheel', function (e) {
        var s = gd._fullLayout._size, r = gd.getBoundingClientRect();
        var x = e.clientX - r.left - s.l, y = e.clientY - r.top - s.t;
        if (x < 0 || x > s.w || y < 0 || y > s.h) return;
        e.preventDefault();
        wheelAcc += e.deltaY;
        wheelFrac = x / s.w;
        if (wheelPending) return;
        wheelPending = true;
        requestAnimationFrame(function () {
            var f = Math.min(2, Math.max(0.5, Math.pow(CFG.wheel, wheelAcc / 100)));
            wheelAcc = 0;
            wheelPending = false;
            zoom(f, wheelFrac);
        });
    }, {passive: false});

    // --- Box zoom | Pan ---
    function setDrag(mode) { Plotly.relayout(gd, {dragmode: mode, 'yaxis.fixedrange': mode === 'pan'}); }
    function syncDrag() {
        var m = gd._fullLayout.dragmode;
        $('fxzoom').className = m === 'zoom' ? 'on' : '';
        $('fxpan').className = m === 'pan' ? 'on' : '';
    }
    $('fxzoom').onclick = function () { setDrag('zoom'); };
    $('fxpan').onclick = function () { setDrag('pan'); };
    syncDrag();
    var panStart = null;                        // {x, v (the visible positions), w} at a pan's mousedown in the plot
    gd.addEventListener('mousedown', function (e) {
        panStart = null;
        if (gd._fullLayout.dragmode !== 'pan' || e.button !== 0) return;
        var s = gd._fullLayout._size, r = gd.getBoundingClientRect();
        var x = e.clientX - r.left - s.l, y = e.clientY - r.top - s.t;
        if (x < 0 || x > s.w || y < 0 || y > s.h) return;
        panStart = {x: e.clientX, v: visible(), w: s.w};
    }, true);
    document.addEventListener('mouseup', function (e) {
        if (!panStart) return;
        panStart.x1 = e.clientX;
        setTimeout(function () { panStart = null; }, 600);   // a click without a drag fires no relayout
    }, true);

    // --- the idx box ---
    function withoutMark() { return (gd.layout.shapes || []).filter(function (s) { return s.name !== 'fx_idx_mark'; }); }
    function mark(a, b) {
        var keep = withoutMark();
        keep.push({type: 'rect', name: 'fx_idx_mark', xref: 'x', yref: 'paper',
                   x0: X().l2r(T[a] - STEP / 2), x1: X().l2r(T[b] + STEP / 2), y0: 0, y1: 1,
                   fillcolor: CFG.markColor, line: {width: 0}, layer: 'below'});
        Plotly.relayout(gd, {shapes: keep});
    }
    function jump() {
        var raw = $('fxidx').value.trim();
        if (raw === '') {                       // an empty box clears the mark (round 2)
            msg();
            if ((gd.layout.shapes || []).some(function (s) { return s.name === 'fx_idx_mark'; })) {
                Plotly.relayout(gd, {shapes: withoutMark()});
            }
            return;
        }
        var v = parseInt(raw, 10);
        if (isNaN(v)) { msg('not an idx: ' + raw); return; }
        var tf = $('fxtf') ? $('fxtf').value : CFG.tf;
        var key = (tf === CFG.tf) ? IDX : IDXH;
        var ps = [], best = -1, bestD = Infinity;
        for (var q = 0; q < N; q++) {
            if (key[q] === v) ps.push(q);
            var dd = Math.abs(key[q] - v);
            if (dd < bestD) { bestD = dd; best = q; }
        }
        var note = '';
        if (!ps.length) {
            var nv = key[best];
            for (q = 0; q < N; q++) if (key[q] === nv) ps.push(q);
            note = 'no ' + tf + ' candle at idx ' + v + ' — showing idx ' + nv;
        }
        var a = ps[0], b = ps[ps.length - 1], c = (a + b) / 2;
        show(c - CFG.half, c + CFG.half).then(function () { mark(a, b); });
        msg(note);
    }
    $('fxgo').onclick = jump;
    $('fxidx').addEventListener('keydown', function (e) { if (e.key === 'Enter') jump(); });

    // --- the price re-fit after a time-only change ---
    gd.on('plotly_relayout', function (ev) {
        if (ev.dragmode !== undefined) {        // the bar's switch or the toolbar's icons
            var pan = gd._fullLayout.dragmode === 'pan';
            if (!!gd._fullLayout.yaxis.fixedrange !== pan) Plotly.relayout(gd, {'yaxis.fixedrange': pan});
            syncDrag();
        }
        var xChanged = ev['xaxis.range[0]'] !== undefined || ev['xaxis.range'] !== undefined;
        var ySet = Object.keys(ev).some(function (k) { return k.indexOf('yaxis.') === 0; });
        if (xChanged && !ySet && panStart && panStart.x1 !== undefined) {   // a pan just released: candle space
            var ps = panStart, n = ps.v[1] - ps.v[0];
            panStart = null;
            var d = Math.round((ps.x1 - ps.x) / ps.w * (n + 1));
            show(ps.v[0] - d, ps.v[1] - d);
            return;
        }
        if (xChanged && !ySet) {
            var v = visible();
            Plotly.relayout(gd, {'yaxis.range': priceRange(v[0], v[1])});
        }
    });
})();
