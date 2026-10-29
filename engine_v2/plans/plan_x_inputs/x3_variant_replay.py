"""PLAN_X X3 — a full replay of one window through a scratch tree, the base or the re-ask VARIANT (the main checkout is
never touched). `plan_w_inputs/w9b_window_replay.py` + one step: with `--variant`, `x3_variant_patch.py` patches the
tree's engine copy before the replay.

    python engine_v2/plans/plan_x_inputs/x3_variant_replay.py TREE SOURCE START END [--variant] [--no-run]   (repo root)

TREE: a SHORT fresh path (Windows MAX_PATH); SOURCE: a git ref. The tree holds a copy of `oanda.cfg` — DELETE it when
done. `--no-run` builds (and patches) the tree only (for a pytest run on it).
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile

args = [a for a in sys.argv[1:] if not a.startswith("--")]
variant = "--variant" in sys.argv
no_run = "--no-run" in sys.argv
tree, source, start, end = args[:4]
tree = os.path.abspath(tree)
assert not os.path.exists(os.path.join(tree, "engine_v2")), f"{tree} already has engine_v2/ — use a fresh dir"
os.makedirs(tree, exist_ok=True)
blob = subprocess.run(["git", "archive", source, "engine_v2", ":(exclude)engine_v2/plans"],
                      check=True, capture_output=True).stdout
with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
    tf.extractall(tree)
shutil.copy("oanda.cfg", os.path.join(tree, "oanda.cfg"))
if variant:
    subprocess.run([sys.executable, os.path.join("engine_v2", "plans", "plan_x_inputs", "x3_variant_patch.py"), tree],
                   check=True)

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
if no_run:
    print(f"{tree}: built ({'variant' if variant else 'base'}), not run")
    sys.exit(0)

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=tree)
with open(os.path.join(tree, "run.log"), "w", encoding="utf-8") as log:
    rc = subprocess.run([sys.executable, "-m", "engine_v2.run_replay"], cwd=tree, env=env, stdout=log,
                        stderr=subprocess.STDOUT).returncode
lines = open(os.path.join(tree, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
i = next((k for k, l in enumerate(lines) if "=== Replay Timing ===" in l), None)
print(f"{tree} ({'variant' if variant else 'base'} {start} -> {end}): exit {rc}")
print("\n".join(lines[i:i + 16]) if i is not None else "\n".join(lines[-15:]))
