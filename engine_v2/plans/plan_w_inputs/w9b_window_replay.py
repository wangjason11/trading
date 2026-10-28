"""PLAN_W 9b — a full replay of ANOTHER window through a scratch tree (the main checkout is never touched).

    python engine_v2/plans/plan_w_inputs/w9b_window_replay.py TREE SOURCE START END     (from the repo root)

TREE (a SHORT path — Windows MAX_PATH, e.g. %TEMP%/pw/base) gets `engine_v2/` from SOURCE — a git ref (`git archive`)
or `WORKTREE` (the working tree's `engine_v2/`) — without `plans/` and `__pycache__`, a copy of `oanda.cfg` (DELETE
the tree when done), and `config.py`'s active window set to START / END (YYYY-MM-DD, UTC). Then runs
`python -m engine_v2.run_replay` in TREE (log: TREE/run.log; outputs: TREE/artifacts/) and prints the timing block.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile

tree, source, start, end = sys.argv[1:5]
tree = os.path.abspath(tree)
assert not os.path.exists(os.path.join(tree, "engine_v2")), f"{tree} already has engine_v2/ — use a fresh dir"
os.makedirs(tree, exist_ok=True)
if source == "WORKTREE":
    shutil.copytree("engine_v2", os.path.join(tree, "engine_v2"),
                    ignore=shutil.ignore_patterns("plans", "__pycache__", "*.pyc"))
else:
    blob = subprocess.run(["git", "archive", source, "engine_v2", ":(exclude)engine_v2/plans"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(tree)
shutil.copy("oanda.cfg", os.path.join(tree, "oanda.cfg"))

cfg = os.path.join(tree, "engine_v2", "config.py")
text = open(cfg, encoding="utf-8", newline="").read()
y0, m0, d0 = (int(x) for x in start.split("-"))
y1, m1, d1 = (int(x) for x in end.split("-"))
pat_s = re.compile(r"^    start=datetime\(2025, 12, 1, 0, 0, 0, tzinfo=timezone\.utc\),", re.M)
pat_e = re.compile(r"^    end=datetime\(2026, 1, 20, 0, 0, 0, tzinfo=timezone\.utc\),", re.M)
assert len(pat_s.findall(text)) == 1 and len(pat_e.findall(text)) == 1, "the active window lines moved"
text = pat_s.sub(f"    start=datetime({y0}, {m0}, {d0}, 0, 0, 0, tzinfo=timezone.utc),", text)
text = pat_e.sub(f"    end=datetime({y1}, {m1}, {d1}, 0, 0, 0, tzinfo=timezone.utc),", text)
open(cfg, "w", encoding="utf-8", newline="").write(text)

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=tree)
with open(os.path.join(tree, "run.log"), "w", encoding="utf-8") as log:
    rc = subprocess.run([sys.executable, "-m", "engine_v2.run_replay"], cwd=tree, env=env, stdout=log,
                        stderr=subprocess.STDOUT).returncode
lines = open(os.path.join(tree, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
i = next((k for k, l in enumerate(lines) if "=== Replay Timing ===" in l), None)
print(f"{tree}: exit {rc}")
print("\n".join(lines[i:i + 16]) if i is not None else "\n".join(lines[-15:]))
