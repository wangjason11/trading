"""Exact-once string replacement preserving each file's line ends (Plan E E4; the JSON CLI 2026-10-09).

    from edit_once import edit; edit(path, [(old, new), ...])                       # one file, from Python
    python engine_v2/plans/plan_e_inputs/review_scripts/edit_once.py EDITS.json      # many files (repo root)

EDITS.json: [{"file": path, "old": text, "new": text}, ...] with LF line ends in `old` / `new`. Each `old` must occur
EXACTLY once in its LF-normalised file at its turn (edits apply in list order); a file with mixed line ends is refused;
nothing is written unless every edit of every file passes. A CRLF file is written back CRLF: with `core.autocrlf=true`
the working tree's docs and tests ARE CRLF (Git Bash `cat -A` hides the `^M`; `sed -i` rewrites the file LF). The JSON
file sidesteps a Bash heredoc of a Python script, which an apostrophe in the text breaks.
"""
import json
import sys


def _plan(edits):
    files = {}
    for path, old, new in edits:
        if path not in files:
            raw = open(path, "rb").read().decode("utf-8")
            crlf, lf = raw.count("\r\n"), raw.count("\n")
            assert crlf in (0, lf), f"{path}: mixed line ends ({crlf} CRLF / {lf} LF)"
            files[path] = [raw.replace("\r\n", "\n"), crlf > 0]
        n = files[path][0].count(old)
        assert n == 1, f"{path}: old occurs {n} times: {old[:70]!r}"
        files[path][0] = files[path][0].replace(old, new)
    return files


def _write(files):
    for path, (text, crlf) in files.items():
        open(path, "wb").write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        print(f"edited {path} ({'CRLF' if crlf else 'LF'})")


def edit(p, pairs):
    _write(_plan([(p, old, new) for old, new in pairs]))


if __name__ == "__main__":
    _write(_plan([(e["file"], e["old"], e["new"]) for e in json.load(open(sys.argv[1], encoding="utf-8"))]))
