"""PLAN_W 9c — a variant replay of the CONFIG window through a scratch tree: the working tree's `engine_v2/` (without
`plans/`, `__pycache__`) + ONE text replacement (RELPATH under the tree, OLD must occur exactly once) + a copy of
`oanda.cfg` (DELETE the tree when done). Runs `python -m engine_v2.run_replay` in TREE (TREE/run.log,
TREE/artifacts/) and prints the timing block. Question 1's variant (the subs return their CTS KL zones):

    python engine_v2/plans/plan_w_inputs/w9c_variant_replay.py %TEMP%/pw9c/cts engine_v2/multitf/pooled_structure_build.py \
        'source_kinds: Tuple[str, ...] = ("BOS",),' 'source_kinds: Tuple[str, ...] = ("BOS", "CTS"),'
"""
import os
import shutil
import subprocess
import sys

tree, rel, old, new = sys.argv[1:5]
tree = os.path.abspath(tree)
assert not os.path.exists(os.path.join(tree, "engine_v2")), f"{tree} already has engine_v2/"
os.makedirs(tree, exist_ok=True)
shutil.copytree("engine_v2", os.path.join(tree, "engine_v2"),
                ignore=shutil.ignore_patterns("plans", "__pycache__", "*.pyc"))
shutil.copy("oanda.cfg", os.path.join(tree, "oanda.cfg"))
p = os.path.join(tree, rel)
text = open(p, encoding="utf-8", newline="").read()
assert text.count(old) == 1, f"{old!r} occurs {text.count(old)} times in {rel}"
open(p, "w", encoding="utf-8", newline="").write(text.replace(old, new))

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=tree)
with open(os.path.join(tree, "run.log"), "w", encoding="utf-8") as log:
    rc = subprocess.run([sys.executable, "-m", "engine_v2.run_replay"], cwd=tree, env=env, stdout=log,
                        stderr=subprocess.STDOUT).returncode
lines = open(os.path.join(tree, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
i = next((k for k, l in enumerate(lines) if "=== Replay Timing ===" in l), None)
print(f"{tree}: exit {rc}")
print("\n".join(lines[i:i + 16]) if i is not None else "\n".join(lines[-15:]))
