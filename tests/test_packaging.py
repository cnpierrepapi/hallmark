"""What the deployed function actually loads, checked against what it installs.

Two bugs hid here at once and neither showed up at runtime. A columnar engine
was installed into the function and imported by nothing, paying for itself on
every cold start. Three packages the request path imports directly were in no
requirements file at all and only resolved because an adapter happened to
depend on them, which is a constraint someone else is free to drop.

Both are invisible to the other tests, because the code works either way until
the day it does not. So they are asserted from the import graph instead.
"""

from __future__ import annotations

import ast
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Deliberately not "api": that is the stem of hallmark/api.py, and keying the
# entry point the same way silently overwrites the module it imports, leaving
# a graph that reaches nothing and a test that asserts nothing.
ENTRY = "__entry__"

# Bundled for the local pipeline but never reachable from a request. Vercel
# uploads hallmark/**, so these ship as files; what matters is that nothing on
# the request path imports them, because that is what pulls pyarrow in.
LOCAL_ONLY = {"campaign", "ledger"}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            if node.module == "hallmark":
                found.update(f"hallmark.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


def _graph() -> dict[str, set[str]]:
    graph = {p.stem: _imports(p) for p in (ROOT / "hallmark").glob("*.py")}
    graph[ENTRY] = _imports(ROOT / "api" / "index.py")
    return graph


def _reachable() -> tuple[set[str], set[str]]:
    """Every hallmark module and third-party package a request can reach."""
    graph = _graph()
    seen: set[str] = set()
    stack = [ENTRY]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        for dep in graph.get(module, ()):
            if dep.startswith("hallmark."):
                name = dep.split(".")[1]
                if name in graph:
                    stack.append(name)

    third = {
        dep.split(".")[0]
        for module in seen
        for dep in graph.get(module, ())
        if not dep.startswith(("hallmark", "__future__"))
        and dep.split(".")[0] not in sys.stdlib_module_names
    }
    return seen, third


def _declared() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name = line.split("==")[0].split("[")[0].split(">")[0].split("<")[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


class TestDeployedFunction:
    def test_the_ledger_never_reaches_the_request_path(self):
        """Reaching it would drag pyarrow into every cold start."""
        reachable, _ = _reachable()
        assert not (LOCAL_ONLY & reachable), (
            f"{sorted(LOCAL_ONLY & reachable)} became reachable from api/index.py. "
            "Attempts recorded live must go through hallmark.attempts, which writes "
            "JSON, so the Parquet writer stays on the machine that queries it."
        )

    def test_no_columnar_engine_is_installed_for_the_function(self):
        declared = _declared()
        assert "pyarrow" not in declared and "tzdata" not in declared, (
            "pyarrow is only needed where the ledger is written and read, which is "
            "requirements-dev.txt. Installing it here costs every cold start."
        )

    def test_everything_imported_at_runtime_is_declared(self):
        """A package that only arrives transitively can leave without warning."""
        _, third = _reachable()
        declared = _declared()

        missing = []
        for module in sorted(third):
            dists = packages_distributions().get(module)
            if not dists:
                continue  # not installed here, so nothing to check against
            if not any(d.lower().replace("_", "-") in declared for d in dists):
                missing.append(f"{module} (from {', '.join(sorted(set(dists)))})")

        assert not missing, (
            "imported on the request path but declared in no requirements file: "
            + "; ".join(missing)
        )
