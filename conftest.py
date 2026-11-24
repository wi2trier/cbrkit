"""Skips collecting the modules whose optional dependencies are not installed, so a partial install runs the rest."""

import ast
from importlib.util import find_spec
from pathlib import Path

from cbrkit._lazy import EXTRAS


def _imports(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module]

    return []


def _needs_missing_extra(path: Path) -> bool:
    """Whether a module imports an optional dependency that is missing at its top level."""
    roots = {
        name.partition(".")[0]
        for node in ast.parse(path.read_bytes()).body
        for name in _imports(node)
    }

    return any(root in EXTRAS and find_spec(root) is None for root in roots)


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Ignore the modules that need a missing extra, defer to pytest otherwise."""
    if collection_path.suffix == ".py" and _needs_missing_extra(collection_path):
        return True

    return None
