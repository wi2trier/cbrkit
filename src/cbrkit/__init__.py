"""
**[🌟 GitHub Project 🌟](https://github.com/wi2trier/cbrkit)**

.. include:: ../../README.md
   :start-after: <!-- PDOC_START -->

"""

import logging

from . import (
    adapt,
    constants,
    cycle,
    dumpers,
    eval,
    helpers,
    indexable,
    loaders,
    model,
    retain,
    retrieval,
    reuse,
    revise,
    sim,
    synthesis,
    system,
    typing,
)

__all__ = [
    "adapt",
    "constants",
    "cycle",
    "dumpers",
    "eval",
    "helpers",
    "indexable",
    "loaders",
    "model",
    "retain",
    "retrieval",
    "reuse",
    "revise",
    "sim",
    "synthesis",
    "system",
    "typing",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
