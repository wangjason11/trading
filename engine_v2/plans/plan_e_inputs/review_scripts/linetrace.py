"""Line tracer for ONE source file — a stand-in for `coverage` (not installed here) to prove code dead or live.

Written for hygiene 5b (2026-09-28): it showed `FibTracker._update_fib_cts`'s cross branch body and the h1
create-on-fail single ran 0 times over the replay AND the full suite, and captured the cond1/cond2/cond3 values at
the decision line (all traces: `plans/PLAN_E_naming_event_convention.md` §7.2 note, commit 226d0c0).

Import it BEFORE the code under test; it installs `sys.settrace` + `threading.settrace` and dumps JSON at exit.
Only frames whose code file ends with LINETRACE_FILE get a line tracer, but the global hook still runs on every
call — expect the suite to take ~2x (49 s -> 113 s on 2026-09-28). Another tracer (a debugger, coverage) replaces it.

Environment:
  LINETRACE_FILE   required: path suffix of the traced file, e.g. engine_v2/zones/fib_tracker.py
  LINETRACE_WATCH  optional: "LINE:FUNC:var1,var2;LINE:FUNC:var" — record those locals (repr) each time LINE of
                   FUNC is about to run (the values BEFORE that line executes)
  LINETRACE_OUT    optional: the JSON path (default: <tempdir>/linetrace.json — never the repo)

Replay (repo root):
  python -c "import sys; sys.path.insert(0, r'<this dir>'); import linetrace, runpy; runpy.run_module('engine_v2.run_replay', run_name='__main__')" > <scratch>/run_trace.log 2>&1
  <this dir> must be a WINDOWS path under Git Bash (`cygpath -w`): a /c/... path gives ModuleNotFoundError: linetrace.
Suite (repo root; the plugin import is what installs the tracer):
  PYTHONPATH=<this dir> python -m pytest -q -p linetrace -p no:cacheprovider
Report:
  python <this dir>/linetrace.py report <out.json> 1600-1622 1730-1741 [...]
  — per range: the lines hit with counts, and the watched values grouped.

CAVEAT (the 5b slip): a branch's `if` line runs on EVERY call; to prove the BODY dead, start the range on the
body's first line, not on the `if`.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import tempfile
import threading
from collections import Counter

_TARGET = os.environ.get("LINETRACE_FILE", "").replace("\\", "/")
_OUT = os.environ.get("LINETRACE_OUT") or os.path.join(tempfile.gettempdir(), "linetrace.json")
_WATCH = {}
for spec in filter(None, os.environ.get("LINETRACE_WATCH", "").split(";")):
    line, func, names = spec.split(":")
    _WATCH[(int(line), func)] = tuple(n for n in names.split(",") if n)

HITS: dict = {}
VALS: list = []


def _local(frame, event, arg):
    if event == "line":
        ln = frame.f_lineno
        HITS[ln] = HITS.get(ln, 0) + 1
        names = _WATCH.get((ln, frame.f_code.co_name))
        if names:
            VALS.append((ln, frame.f_code.co_name, {n: repr(frame.f_locals.get(n, "<unset>")) for n in names}))
    return _local


def _global(frame, event, arg):
    # `not _TARGET`: at interpreter shutdown module globals are cleared to None and a late call (a logging weakref
    # callback) used to print a harmless "TypeError: endswith first arg must be str ... NoneType" after the suite.
    if _TARGET and frame.f_code.co_filename.replace("\\", "/").endswith(_TARGET):
        return _local
    return None


def _dump():
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump({"file": _TARGET, "hits": HITS, "vals": VALS}, fh)
    print(f"[linetrace] {_TARGET}: {len(HITS)} lines hit -> {_OUT}", file=sys.stderr)


def _install():
    if not _TARGET:
        raise SystemExit("[linetrace] set LINETRACE_FILE to the traced file's path suffix")
    sys.settrace(_global)
    threading.settrace(_global)
    atexit.register(_dump)


def report(path: str, ranges: list[str]) -> None:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    hits = {int(k): v for k, v in data["hits"].items()}
    print(f"{data['file']}: {len(hits)} distinct lines hit")
    for r in ranges:
        a, b = (int(x) for x in r.split("-"))
        live = {ln: hits[ln] for ln in range(a, b + 1) if ln in hits}
        print(f"  {a}-{b}: " + ("NEVER RUN" if not live else ", ".join(f"{ln}x{n}" for ln, n in live.items())))
    if data["vals"]:
        print("  watched values:")
        for (ln, func, vals), n in Counter((v[0], v[1], tuple(sorted(v[2].items()))) for v in data["vals"]).items():
            print(f"    {n} x {func}:{ln} {dict(vals)}")


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "report":
    report(sys.argv[2], sys.argv[3:])
elif __name__ != "__main__":
    _install()
