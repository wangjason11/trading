"""SESSION ORDER item 10 (Week 9 Part 1): the parked / deferred items sweep — the grep pass.

Greps every repo .md (minus .venv / artifacts / .git / .pytest_cache / the legacy side projects), the four project
skills, every memory file and every engine_v2 .py for deferral wording; prints each hit with its file, line and the
nearest markdown heading above it, then a per-file hit count. Each hit is then READ IN CONTEXT by hand — this script
only finds them; the classification lives in `part1_sweep.md`.

Usage (from the repo root):  python engine_v2/plans/part1_sweep_inputs/sweep_grep.py MEMORY_DIR > OUT.txt
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MEMORY_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None

SKIP_PARTS = {".venv", "artifacts", ".git", ".pytest_cache", "Forex_Algo", "classifier", "screener",
              "legacy_2025", "legacy_2025_port", "__pycache__", "node_modules"}

MD_RX = re.compile(
    r"defer|parked|\bpark\b|follow-?up|not folded|unscheduled|revisit|open question|out of scope|not built"
    r"|\bTODO\b|\bFIXME\b|\bXXX\b|latent|backlog|come back|change my mind|decide later|still open|left open"
    r"|unresolved|not yet|known (gap|limitation)|may revisit",
    re.IGNORECASE,
)
MD_RX_CS = re.compile(r"\bOPEN\b|TRIGGER\b|\bDECIDE\b|\bCONFIRM\b")
PY_RX = re.compile(r"#.*(\bTODO\b|\bFIXME\b|\bXXX\b|\bHACK\b|defer|latent|follow-?up|not yet|revisit)", re.IGNORECASE)


def wanted(p: Path) -> bool:
    return not any(part in SKIP_PARTS for part in p.parts)


def sources():
    out = []
    for p in sorted(ROOT.rglob("*.md")):
        if wanted(p.relative_to(ROOT)):
            out.append(("repo", p))
    if MEMORY_DIR is not None:
        for p in sorted(MEMORY_DIR.glob("*.md")):
            out.append(("memory", p))
    for p in sorted((ROOT / "engine_v2").rglob("*.py")):
        if wanted(p.relative_to(ROOT)):
            out.append(("code", p))
    return out


def main():
    counts = Counter()
    n_files = Counter()
    for kind, p in sources():
        n_files[kind] += 1
        label = (p.relative_to(ROOT) if kind != "memory" else Path("memory") / p.name).as_posix()
        heading = ""
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            if kind != "code" and line.startswith("#"):
                heading = line.strip()[:90]
            hit = PY_RX.search(line) if kind == "code" else (MD_RX.search(line) or MD_RX_CS.search(line))
            if hit:
                counts[label] += 1
                text = line.strip()
                print(f"{label}:{i}: [{heading}] {text[:240]}")
    print("\n=== per-file hit counts (descending) ===")
    for label, c in counts.most_common():
        print(f"{c:5d}  {label}")
    print(f"\nfiles scanned: {dict(n_files)}; files with hits: {len(counts)}; hits: {sum(counts.values())}")


if __name__ == "__main__":
    main()
