"""
CBRkit contains retain phase functions for deciding whether to store
solved cases back into the casebase.
The retain phase uses storage functions (from ``cbrkit.retain.storage``)
to integrate cases into the casebase.
"""

from ..model import QueryResultStep, Result, ResultStep
from .apply import (
    apply_batches,
    apply_pair,
    apply_queries,
    apply_query,
    apply_result,
)
from .build import build, dropout
from .storage import KeyFunc, indexable, static

__all__ = [
    "build",
    "dropout",
    "apply_result",
    "apply_batches",
    "apply_queries",
    "apply_query",
    "apply_pair",
    "KeyFunc",
    "indexable",
    "static",
    "Result",
    "ResultStep",
    "QueryResultStep",
]
