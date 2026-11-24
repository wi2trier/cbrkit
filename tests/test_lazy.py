import os
import pickle
import pkgutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

import cbrkit
from cbrkit._lazy import EXTRAS


def _python(*args: str, **env: str) -> str:
    return subprocess.run(
        [sys.executable, *args],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | env,
    ).stdout.strip()


def test_import_loads_no_extra() -> None:
    loaded = _python(
        "-c",
        "import sys, cbrkit\n"
        "from cbrkit._lazy import EXTRAS\n"
        "print(sorted(name for name in EXTRAS if name in sys.modules))",
    )
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]

    assert loaded == "[]"
    assert set(EXTRAS.values()) <= extras.keys()


@pytest.mark.parametrize("eager", ["", "1"])
def test_exports_resolve(eager: str) -> None:
    # Eager mode must bind every name in the module namespace, which pdoc reads.
    unresolved = _python(
        "-c",
        "import os, pkgutil, cbrkit\n"
        "from importlib import import_module\n"
        "from cbrkit._lazy import EXTRAS\n"
        "eager = bool(os.environ['CBRKIT_EAGER_IMPORTS'])\n"
        "unresolved = []\n"
        "for info in pkgutil.walk_packages(cbrkit.__path__, 'cbrkit.'):\n"
        "    if info.name == 'cbrkit.__main__':\n"
        "        continue\n"
        "    try:\n"
        "        module = import_module(info.name)\n"
        "    except ModuleNotFoundError as e:\n"
        "        if (e.name or '').partition('.')[0] in EXTRAS:\n"
        "            continue\n"
        "        raise\n"
        "    for name in getattr(module, '__all__', []):\n"
        "        if name not in vars(module) if eager else not hasattr(module, name):\n"
        "            unresolved.append(f'{info.name}.{name}')\n"
        "print(unresolved)",
        CBRKIT_EAGER_IMPORTS=eager,
    )

    assert unresolved == "[]"


def test_package_exports_every_public_module() -> None:
    modules = {
        info.name
        for info in pkgutil.iter_modules(cbrkit.__path__)
        if not info.name.startswith("_")
    }

    assert set(cbrkit.__all__) == modules - {"cli"}
    assert all(isinstance(getattr(cbrkit, name), ModuleType) for name in cbrkit.__all__)


def test_missing_extra() -> None:
    result = _python(
        "-c",
        "import sys\n"
        "sys.modules['sentence_transformers'] = None\n"
        "import cbrkit\n"
        "rerank = cbrkit.retrieve.rerank\n"
        "assert hasattr(rerank, 'HTTP') and not hasattr(rerank, 'CrossEncoder')\n"
        "try:\n"
        "    rerank.CrossEncoder\n"
        "except AttributeError as e:\n"
        "    print(e)",
    )

    assert "pip install 'cbrkit[transformers]'" in result


def test_lazy_exports_behave_like_imports() -> None:
    import cbrkit

    graphs = cbrkit.sim.graphs
    star = _python("-c", "from cbrkit.sim.graphs import *\nprint(DFS.__module__)")

    assert isinstance(cbrkit.retrieve.rerank.HTTP, type)
    assert star == "cbrkit.sim.graphs._dfs"
    assert "VF2Rustworkx" in dir(graphs)

    with mock.patch("cbrkit.sim.graphs.VF2Rustworkx") as patched:
        assert graphs.VF2Rustworkx is patched

    func = graphs.VF2Networkx(node_measure=cbrkit.sim.generic.Equality())

    assert type(pickle.loads(pickle.dumps(func))) is graphs.VF2Networkx


FORKSERVER_SCRIPT = """
import multiprocessing

import cbrkit


def similarity(conn):
    graph = cbrkit.model.graph.from_dict({"nodes": {"a": 1, "b": 2}, "edges": {}})
    conn.send(conn.recv()(graph, graph).value)


if __name__ == "__main__":
    context = multiprocessing.get_context("forkserver")
    parent, child = context.Pipe()
    process = context.Process(target=similarity, args=(child,))
    process.start()
    parent.send(cbrkit.sim.graphs.VF2Networkx(node_measure=cbrkit.sim.generic.Equality()))
    print(parent.recv())
    process.join()
"""


def test_lazy_exports_in_forkserver(tmp_path: Path) -> None:
    script = tmp_path / "forkserver.py"
    script.write_text(FORKSERVER_SCRIPT)

    assert _python(str(script)) == "1.0"


@pytest.mark.skipif(sys.version_info < (3, 15), reason="PEP 810 needs Python 3.15")
def test_pep810_lazy_imports() -> None:
    result = _python(
        "-c",
        "import sys\n"
        "lazy import cbrkit\n"
        "lazy from cbrkit.retrieve.rerank import HTTP\n"
        "lazy from cbrkit.sim.graphs import VF2Rustworkx\n"
        "assert 'cbrkit' not in sys.modules\n"
        "assert VF2Rustworkx is cbrkit.sim.graphs.VF2Rustworkx\n"
        "print(HTTP.__module__)",
    )

    assert result == "cbrkit.retrieve.rerank._http"
