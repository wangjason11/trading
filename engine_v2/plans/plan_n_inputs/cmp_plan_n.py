"""PLAN_N §4: prove a replay's CSVs = the base save's CSVs under EXACTLY the declared migration (read-only).

    python engine_v2/plans/plan_n_inputs/cmp_plan_n.py BASE_SAVE CUR_DEBUG [--expect-cells 164]

For each of the base save's CSVs: the 7 declared ones are compared CELL BY CELL after applying the transform to the base
cells (every other cell must be equal as text); the others must be md5-equal. The transform (PLAN_N §4):
- `*kl_zones.csv` meta: every bounds_steps entry's `start_idx` -> `base_idx` (INIT) / `expanded_idx` (expansion), in place;
- `*structure_events.csv` meta: label-path RANGE_STARTED `start_idx` / `confirm_idx` -> `range_anchor_idx` /
  `confirmed_at` in place; pullback / proximity RANGE_STARTED gain `range_anchor_idx` (= its `cts_anchor_idx`) and
  `confirmed_at` (= its idx) as the FIRST two keys; STATE_CHANGED `range_confirmed`: `effective_idx` -> the row's idx;
- `*unresolved_triggers.csv` reason: `degenerate_parent_cycle` -> `collapsed_parent_cycle`.
Prints the per-file verdict and the changed-cell counts; exit 1 on any mismatch, a CSV present on only one side, or a
changed-cell total other than `--expect-cells` (PLAN_N landing review finding 8).
"""
import ast
import csv
import hashlib
import sys
from pathlib import Path


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def rows(p: Path):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def lit(cell: str) -> dict:
    d = ast.literal_eval(cell)
    assert str(d) == cell, f"meta cell does not round-trip: {cell[:80]}"
    return d


def t_kl(meta: dict) -> dict:
    steps = []
    for s in meta["bounds_steps"]:
        key = "base_idx" if s["event"] == "INIT" else "expanded_idx"
        steps.append({(key if k == "start_idx" else k): v for k, v in s.items()})
    return {**meta, "bounds_steps": steps}


def t_event(idx: int, typ: str, meta: dict) -> dict:
    if typ == "RANGE_STARTED":
        if "start_idx" in meta:
            ren = {"start_idx": "range_anchor_idx", "confirm_idx": "confirmed_at"}
            return {ren.get(k, k): v for k, v in meta.items()}
        return {"range_anchor_idx": meta["cts_anchor_idx"], "confirmed_at": idx, **meta}
    if typ == "STATE_CHANGED" and meta.get("reason") == "range_confirmed":
        return {**meta, "effective_idx": idx}
    return meta


def compare(base: Path, cur: Path) -> tuple[bool, int]:
    b, c = rows(base), rows(cur)
    if len(b) != len(c) or b[0] != c[0]:
        return False, -1
    hdr = b[0]
    changed = 0
    for rb, rc in zip(b[1:], c[1:]):
        exp = list(rb)
        name = base.name
        if name.endswith("kl_zones.csv"):
            i = hdr.index("meta")
            exp[i] = str(t_kl(lit(rb[i])))
        elif name.endswith("structure_events.csv"):
            i = hdr.index("meta")
            exp[i] = str(t_event(int(rb[hdr.index("idx")]), rb[hdr.index("type")], lit(rb[i])))
        elif name.endswith("unresolved_triggers.csv"):
            i = hdr.index("reason")
            exp[i] = "collapsed_parent_cycle" if rb[i] == "degenerate_parent_cycle" else rb[i]
        if exp != rc:
            return False, -1
        changed += sum(x != y for x, y in zip(rb, rc))
    return True, changed


def main():
    base_dir, cur_dir = Path(sys.argv[1]), Path(sys.argv[2])
    expect = int(sys.argv[sys.argv.index('--expect-cells') + 1]) if '--expect-cells' in sys.argv else None
    ok = True
    total = 0
    declared = ("kl_zones.csv", "structure_events.csv", "unresolved_triggers.csv")
    n_eq = n_tr = 0
    for bp in sorted(base_dir.glob("*.csv")):
        cp = cur_dir / bp.name
        if not cp.exists():
            print(f"MISSING  {bp.name}")
            ok = False
            continue
        if bp.name.endswith(declared):
            good, n = compare(bp, cp)
            print(f"{'TRANSFORM-EQUAL' if good else 'MISMATCH'}  {bp.name}  changed cells={n}")
            ok &= good
            n_tr += good
            total += max(n, 0)
        else:
            same = md5(bp) == md5(cp)
            print(f"{'md5-equal' if same else 'DIFF'}  {bp.name}")
            ok &= same
            n_eq += same
    extra = sorted({p.name for p in cur_dir.glob("*.csv")} - {p.name for p in base_dir.glob("*.csv")})
    if extra:
        print("EXTRA in current:", extra)
        ok = False
    if expect is not None and total != expect:
        print(f'changed cells {total} != expected {expect}')
        ok = False
    print(f"\n{n_eq} md5-equal, {n_tr} transform-equal, {total} changed cells; {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
