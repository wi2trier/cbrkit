"""
**[🌟 GitHub Project 🌟](https://github.com/wi2trier/cbrkit)**

.. include:: ../README.md

"""

from . import global_sim, load, sim, typing
from .retrieve import (
    RetrievalResult,
    retrieve,
    retriever,
)

__all__ = (
    "load",
    "sim",
    "global_sim",
    "typing",
    "retrieve",
    "retriever",
    "RetrievalResult",
)
