"""The package layers rule out import cycles by construction.

A module may only import from its own unit or from units of lower layers,
so sim and retrieve for instance never depend on the storage backends of indexable.
The contracts of `cbrkit.typing` form the lowest layer and depend on the standard library and numpy only.
"""

import ast
import sys
from pathlib import Path

import cbrkit

LAYERS: tuple[frozenset[str], ...] = (
    frozenset({"errors", "typing"}),
    frozenset({"_lazy", "schema"}),
    frozenset({"_util"}),
    frozenset({"helpers"}),
    frozenset({"model"}),
    frozenset({"io"}),
    frozenset({"sim"}),
    frozenset({"_phase", "adapt"}),
    frozenset({"retrieve", "reuse", "revise", "retain", "maintain"}),
    frozenset({"cycle", "eval", "indexable"}),
    frozenset({"spec"}),
    frozenset({"system"}),
    frozenset({"__init__"}),
    frozenset({"cli"}),
    frozenset({"__main__"}),
)
ROOT = Path(cbrkit.__file__).parent


def layer(unit: str) -> int:
    return next(index for index, units in enumerate(LAYERS) if unit in units)


def unit(module: list[str]) -> str:
    return module[1] if len(module) > 1 else "__init__"


def imports(path: Path) -> set[str]:
    """The cbrkit units a module imports."""
    parts = ["cbrkit", *path.relative_to(ROOT).with_suffix("").parts]
    package = parts[:-1]
    units: set[str] = set()

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            targets = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level:
            base = package[: len(package) - node.level + 1]
            module = [*base, *(node.module.split(".") if node.module else [])]
            targets = (
                [[*module, alias.name] for alias in node.names]
                if module == ["cbrkit"]
                else [module]
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module.split(".")]
        else:
            continue

        units |= {unit(target) for target in targets if target[0] == "cbrkit"}

    return units


def test_imports_point_down_the_layers() -> None:
    violations = [
        f"{path.relative_to(ROOT)} imports {target}"
        for path in sorted(ROOT.rglob("*.py"))
        for source in [unit(["cbrkit", *path.relative_to(ROOT).with_suffix("").parts])]
        for target in imports(path)
        if target != source and layer(target) >= layer(source)
    ]

    assert violations == []


def test_typing_depends_on_the_standard_library_and_numpy_only() -> None:
    tree = ast.parse((ROOT / "typing.py").read_text())
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        "." * node.level + (node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert {
        module
        for module in modules
        if module.split(".")[0] not in sys.stdlib_module_names | {"numpy"}
    } == set()
