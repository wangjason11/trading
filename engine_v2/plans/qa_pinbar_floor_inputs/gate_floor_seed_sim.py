"""PLAN_U U2 — the fetch gate's floor-seed branches (PLAN_U §8 B7), simulated on doctored copies of a real run.log.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/gate_floor_seed_sim.py [RUN_LOG]

Extracts the snippet from the `/compare` skill §2b (the canonical copy), runs it with Git Bash in a scratch dir per
case (a `*_raw.csv` older than the case's run.log) and prints the verdict per case; writes the `.txt` next to itself.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "run.log"
skill = (REPO / ".claude/skills/compare/SKILL.md").read_text(encoding="utf-8")
snippet = re.search(r"```bash\n(RAW=\$\(ls -t.*?\nfi)\n```", skill.replace("\r\n", "\n"), re.S).group(1)
real = LOG.read_bytes().split(b"\n")


def drop(*subs):
    return [l for l in real if not any(s in l for s in subs)]


def crash():
    keep = [l for l in real if b"=== Replay Timing ===" not in l and b"[floor_seed]" not in l]
    k = next(i for i, l in enumerate(keep) if b"[auto_extend]" in l) + 1
    return keep[:k] + [b"Traceback (most recent call last):",
                       b"RuntimeError: [floor_seed] ERROR: H1 NZD_USD: 120 body-% pinbars in 3200 candles before "
                       b"2025-11-16 22:00:00+00:00 after 8 steps of 400 candles (need 180)"]


def no_lower_tf():
    return [l for l in real if b"[data_bridge]" not in l and b"[multi_tf:dual]" not in l
            and b"[floor_seed] M15" not in l]


def no_triggers(keep_m15=True):
    out = [l for l in real if b"[data_bridge]" not in l and b"[multi_tf:dual]" not in l
           and (keep_m15 or b"[floor_seed] M15" not in l)]
    return out + [b"[multi_tf:dual] no triggers \x97 no subs built"]


CASES = [
    ("the real log", "PASS", real),
    ("H1 seed line missing", "FAIL (floor seed)", drop(b"[floor_seed] H1")),
    ("M15 seed line missing", "FAIL (floor seed)", drop(b"[floor_seed] M15")),
    ("H1 seed with another count", "FAIL (floor seed)",
     [l.replace(b"232 body-%", b"231 body-%") for l in real]),
    ("a seed RETRY line (benign)", "PASS",
     real[:3] + [b"[floor_seed] RETRY 1/2 H1 fetch 2025-10-14 - 2025-11-16 in 5 s: OANDA candles request failed 504"]
     + real[3:]),
    ("crashed on [floor_seed] ERROR", "FAIL (stale log or crashed run", crash()),
    ("no lower TF, H1 seed present", "N/A (no lower timeframe)", no_lower_tf()),
    ("no lower TF, no H1 seed", "FAIL (floor seed)", [l for l in no_lower_tf() if b"[floor_seed] H1" not in l]),
    ("no H1 trigger, both seeds", "N/A (no H1 triggers", no_triggers()),
    ("no H1 trigger, no M15 seed", "FAIL (floor seed)", no_triggers(keep_m15=False)),
]

lines = []
ok = True
for name, want, content in CASES:
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "artifacts/debug").mkdir(parents=True)
        raw = dd / "artifacts/debug/x_raw.csv"
        raw.write_text("t\n")
        old = time.time() - 60
        os.utime(raw, (old, old))
        (dd / "run.log").write_bytes(b"\n".join(content))
        r = subprocess.run(["bash", "-c", snippet], cwd=dd, capture_output=True)
        got = r.stdout.decode("cp1252", "replace").splitlines()[0]
        hit = want in got
        ok &= hit
        lines.append(f"{'OK ' if hit else 'BAD'} {name:32} -> {got[:110]}")
lines.append("ALL AS EXPECTED" if ok else "UNEXPECTED VERDICT")
text = "\n".join(lines)
print(text)
Path(__file__).with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
sys.exit(0 if ok else 1)
