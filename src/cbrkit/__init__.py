"""
**[🌟 GitHub Project 🌟](https://github.com/wi2trier/cbrkit)**

.. include:: ../../README.md
   :start-after: <!-- PDOC_START -->

"""

import logging
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        adapt,
        cycle,
        errors,
        eval,
        helpers,
        indexable,
        io,
        model,
        retain,
        retrieve,
        reuse,
        revise,
        schema,
        sim,
        system,
        typing,
    )

__all__ = [
    "adapt",
    "cycle",
    "errors",
    "eval",
    "helpers",
    "indexable",
    "io",
    "model",
    "retain",
    "retrieve",
    "reuse",
    "revise",
    "schema",
    "sim",
    "system",
    "typing",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name: str) -> ModuleType:
    """Imports a submodule on first access, so importing one submodule never loads the others."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return import_module(f".{name}", __name__)
