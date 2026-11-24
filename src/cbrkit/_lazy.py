"""Lazy exports for the names that need an optional dependency.

A module declares its lazy exports as plain `from .module import name` statements
in an `if TYPE_CHECKING:` block and calls `lazy_exports(__name__)` as its last statement.
Type checkers and pdoc read the block as ordinary imports,
while at runtime the module imports each name on first access.
So `import cbrkit` never loads an optional dependency,
and a missing extra surfaces as an `AttributeError` that names it,
caused by a `cbrkit.errors.MissingExtra`.
The module class is swapped when the module body runs,
so it also holds when a downstream package defers that with PEP 810 `lazy import`.

An exported name must not be the name of a submodule of its package,
because PEP 810 resolves `lazy from package import name` to the submodule `package.name` if one exists,
and importing such a submodule replaces the export on the package.
Backend modules carry a leading underscore,
so the package export is the one public path of every backend.

Migration to PEP 810 once cbrkit requires Python 3.15:
replace each `if TYPE_CHECKING:` block and its `lazy_exports(__name__)` call
with the same statements prefixed by `lazy`, then delete `lazy_exports` and `_LazyModule`.
Missing extras then raise the plain `ImportError` of the interpreter,
and `EXTRAS` only serves `extra_hint`.
"""

import ast
import inspect
import os
import pkgutil
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from functools import cache
from importlib import import_module
from types import ModuleType

from .errors import MissingExtra

__all__ = ["EXTRAS", "extra_hint", "lazy_exports"]

# Top-level import name of an optional dependency mapped to the extra that installs it.
EXTRAS: Mapping[str, str] = {
    "aiosqlite": "sqlite-vec",
    "bm25s": "bm25",
    "fastapi": "api",
    "fastexcel": "fastexcel",
    "fastmcp": "mcp",
    "httpx2": "http",
    "lancedb": "lancedb",
    "mcp_types": "mcp",
    "networkx": "graphs",
    "pgvector": "pgvector",
    "pydantic_ai": "pydantic-ai",
    "pygraphviz": "graphviz",
    "rustworkx": "graphs",
    "sentence_transformers": "transformers",
    "sqlalchemy": "sql",
    "sqlite_vec": "sqlite-vec",
    "Stemmer": "bm25",
    "typer": "cli",
    "uvicorn": "api",
    "xsdata": "bpmn",
    "zvec": "zvec",
}


def _install_hint(
    error: ModuleNotFoundError, feature: str, extra: str | None
) -> str | None:
    root = (error.name or "").partition(".")[0]

    if root == "cbrkit" or (extra := extra or EXTRAS.get(root)) is None:
        return None

    return (
        f"{feature} requires {root!r}, install it with `pip install 'cbrkit[{extra}]'`"
    )


@contextmanager
def extra_hint(feature: str, extra: str | None = None) -> Generator[None]:
    """Name the extra to install when an optional dependency is missing.

    Args:
        feature: The name that needs the dependency.
        extra: The extra that installs it, looked up by the import name of the dependency if None.
    """
    try:
        yield
    except ModuleNotFoundError as e:
        if (hint := _install_hint(e, feature, extra)) is None:
            raise

        raise MissingExtra(hint, name=e.name) from e


@cache
def _exports(module_name: str) -> Mapping[str, tuple[str, str]]:
    """Map each lazy export of a module to its source module and attribute."""
    module = sys.modules[module_name]
    exports: dict[str, tuple[str, str]] = {}

    try:
        source = inspect.getsource(module)
    except OSError as e:
        raise ImportError(
            f"{module_name}: lazy exports are read from the module source, which is missing"
        ) from e

    for stmt in ast.parse(source).body:
        if not (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "TYPE_CHECKING"
        ):
            continue

        for node in stmt.body:
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module
                and all(alias.name != "*" for alias in node.names)
            ):
                raise ImportError(
                    f"{module_name}: lazy exports must be `from module import name` statements"
                )

            for alias in node.names:
                exports[alias.asname or alias.name] = (
                    "." * node.level + node.module,
                    alias.name,
                )

    if hasattr(module, "__path__") and (
        clashes := exports.keys()
        & {info.name for info in pkgutil.iter_modules(module.__path__)}
    ):
        raise ImportError(f"{module_name}: lazy exports shadow submodules {clashes}")

    return exports


class _LazyModule(ModuleType):
    """Module whose lazy exports are imported on first access."""

    def __getattr__(self, name: str) -> object:
        if name.startswith("__") or name not in (exports := _exports(self.__name__)):
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}",
                name=name,
                obj=self,
            )

        source, attr = exports[name]

        try:
            with extra_hint(f"{self.__name__}.{name}"):
                module = import_module(source, self.__package__)
        except MissingExtra as e:
            raise AttributeError(str(e), name=name, obj=self) from e

        value = getattr(module, attr)
        setattr(self, name, value)

        return value

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *_exports(self.__name__)})


def lazy_exports(module_name: str) -> None:
    """Import the names of a module's `if TYPE_CHECKING:` block on first access.

    Setting the environment variable `CBRKIT_EAGER_IMPORTS` imports them right away,
    skipping those whose extra is missing, which tools like pdoc need.

    Args:
        module_name: `__name__` of the calling module.
    """
    module = sys.modules[module_name]
    module.__class__ = _LazyModule

    if os.environ.get("CBRKIT_EAGER_IMPORTS"):
        for name in _exports(module_name):
            with suppress(AttributeError):
                getattr(module, name)
