"""Count-asserting regex renamer (CRLF-preserving, all-or-nothing per file). Plan E E5 (2026-09-25).

rename(path, [(regex, repl, expected_count), ...]) — every regex must match
exactly `expected_count` times in the file (checked sequentially, on the text
as modified by the previous pairs); otherwise nothing is written. Count the
expected hits first (`grep -o -w NAME FILE | wc -l`); for a quoted dict key vs a
local of the same name use a lookaround, e.g. r'(?<!["'])name(?!["'])'.
"""
import re


def rename(p, pairs):
    b = open(p, "rb").read()
    crlf = b"\r\n" in b
    s = b.decode("utf-8").replace("\r\n", "\n")
    for rx, repl, n_exp in pairs:
        s, n = re.subn(rx, repl, s)
        assert n == n_exp, (p, rx, n, n_exp)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "wb").write(s.encode("utf-8"))
