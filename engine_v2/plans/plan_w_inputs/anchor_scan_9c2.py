"""Every earlier mutants file's anchors in the files 9c-2 touched: matched once at BASE (pre-9c-2), and now?

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_w_inputs/anchor_scan_9c2.py 158d3d2 > engine_v2/plans/plan_w_inputs/anchor_scan_9c2.txt

The PLAN_V check (`plan_v_inputs/anchor_scan_8c.py`); 9c-2's own mutants file is skipped. A STALE anchor on code 9c-2
moved is listed and judged in PLAN_W §3.5.
"""
import glob, importlib.util, subprocess, sys
BASE = sys.argv[1]
TOUCHED = {"engine_v2/multitf/entity_df_mutation.py", "engine_v2/structure/reference_zone.py"}
old_src = {f: subprocess.run(["git", "show", f"{BASE}:{f}"], capture_output=True).stdout.decode("utf-8").replace("\r\n", "\n") for f in TOUCHED}
new_src = {f: open(f, encoding="utf-8").read() for f in TOUCHED}
stale = 0
checked = 0
files = 0
for path in sorted(glob.glob("engine_v2/plans/**/mutants*.py", recursive=True)):
    if path.replace("\\", "/").endswith("plan_w_inputs/mutants_9c2.py"):
        continue
    try:
        spec = importlib.util.spec_from_file_location("m", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        muts = list(m.MUT)
    except Exception as e:
        print(f"SKIP {path}: {e!r}"); continue
    files += 1
    for mid, rel, old, new in muts:
        if rel not in TOUCHED:
            continue
        a, b = old_src[rel].count(old), new_src[rel].count(old)
        checked += a == 1
        if a == 1 and b != 1:
            stale += 1
            print(f"STALE {path} {mid} ({rel}): {a} -> {b}")
            print(f"      anchor: {old.strip()[:150]!r}")
print(f"{stale} stale of {checked} anchors that matched once at the base ({files} mutants files read)")
