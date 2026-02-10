"""
CBRkit contains revise phase functions for assessing solution quality
and optionally repairing solutions.
The revise phase uses similarity functions for assessment
(from ``cbrkit.sim``) and adaptation functions for repair
(from ``cbrkit.adapt``).
"""

from ..model import QueryResultStep, Result, ResultStep
from .apply import (
    apply_batches,
    apply_pair,
    apply_queries,
    apply_query,
    apply_result,
)
from .build import build

__all__ = [
    "build",
    "apply_result",
    "apply_batches",
    "apply_queries",
    "apply_query",
    "apply_pair",
    "Result",
    "ResultStep",
    "QueryResultStep",
]
