"""The broad exception scopes in the engine — a static guard (PLAN_H §8.5c, landing review C1 / LR2, 2026-10-06c).

The MS resolver re-raises ANY institutional-pattern error as `poi_zones.IPSelectionError`, a `RuntimeError`, so the
narrow `except (ValueError, IndexError)` scopes above it (`entity_df_mutation._build_geometry`) cannot swallow it
(LANDMINES "the resolver's fail-loud scope"). That holds only while no scope on an MS / IP path catches
`RuntimeError`, `Exception`, `BaseException` or everything (bare). This test lists every such handler in the engine
modules (not tests / plans / the unparseable `legacy_2025` archive) and fails on any handler outside the allowlist —
a new one must be checked against the MS / IP call paths and added here deliberately.
"""
from __future__ import annotations

import ast
import pathlib

ENGINE = pathlib.Path(__file__).resolve().parents[1]
BROAD = {"RuntimeError", "Exception", "BaseException", "IPSelectionError"}

# (module, enclosing function) -> the exact names each of its broad handlers catches. None of them encloses an MS run
# or the IP selector (checked 2026-10-06c against the landing review's 4 call paths to `select_ip_ic`).
ALLOWED = {
    # the resolver: the graceful anchor scope (anchor selection + fib construction only), then the re-raise
    ("zones/poi_zones.py", "compute_poi_inners_for_cycle"): [("Exception",), ("Exception",)],
    # the OANDA fetch retries — they wrap `get_history` only
    ("multitf/data_bridge.py", "_fetch_chunk"): [("RuntimeError", "requests.RequestException")],
    ("run_replay.py", "_get_history_retrying"): [("RuntimeError", "requests.RequestException")],
    # `export_sub_tables`, after the pipeline
    ("run_replay.py", "main"): [("Exception",)],
    # leaf helpers: a NaN check, the base-pattern zone helpers, one expansion guard
    ("structure/market_structure.py", "_isnan"): [("Exception",)],
    ("structure/reference_zone.py", "_derive_zone_ad_hoc"): [("Exception",)],
    ("zones/kl_zones_v1.py", "derive_kl_zones_v1"): [("Exception",)],
    ("zones/kl_zones_v1.py", "compute_bos_inner_from_event"): [("Exception",)],
}


def _broad_handlers(root=ENGINE):
    found = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("tests/", "plans/", "legacy_2025/")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(node, func):
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
                if isinstance(child, ast.ExceptHandler):
                    if child.type is None:
                        names = ("<bare>",)
                    else:
                        elts = child.type.elts if isinstance(child.type, ast.Tuple) else [child.type]
                        names = tuple(ast.unparse(e) for e in elts)
                    if names == ("<bare>",) or BROAD & {n.rsplit(".", 1)[-1] for n in names}:
                        found.setdefault((rel, func), []).append(names)
                visit(child, inner)

        visit(tree, "<module>")
    return found


def test_every_broad_except_is_on_the_allowlist():
    assert _broad_handlers() == ALLOWED


def test_the_guard_reports_a_runtime_error_or_bare_scope_and_not_a_narrow_one(tmp_path):
    (tmp_path / "zones").mkdir()
    (tmp_path / "zones" / "m.py").write_text(
        "def f():\n"
        "    try:\n        g()\n    except RuntimeError:\n        pass\n"
        "    try:\n        g()\n    except:\n        pass\n"
        "    try:\n        g()\n    except (ValueError, IndexError):\n        pass\n",
        encoding="utf-8")
    assert _broad_handlers(tmp_path) == {("zones/m.py", "f"): [("RuntimeError",), ("<bare>",)]}
