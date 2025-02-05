"""
**[🌟 GitHub Project 🌟](https://github.com/wi2trier/cbrkit)**

.. include:: ../../README.md
   :start-after: <!-- PDOC_START -->

"""

import atexit
import logging

from . import (
    adapt,
    constants,
    cycle,
    dumpers,
    eval,
    helpers,
    loaders,
    model,
    retrieval,
    reuse,
    sim,
    synthesis,
    typing,
)

__all__ = [
    "adapt",
    "cycle",
    "dumpers",
    "eval",
    "helpers",
    "loaders",
    "model",
    "retrieval",
    "reuse",
    "sim",
    "synthesis",
    "typing",
    "constants",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
atexit.register(helpers.event_loop.close)
