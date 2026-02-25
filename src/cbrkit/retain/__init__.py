"""Retain phase for deciding whether to store solved cases back into the casebase.

The retain phase evaluates solved cases and integrates them into the casebase
using storage functions.
Cases are assessed with a similarity function and stored via a storage backend.

Building Retainers:
- `build`: Creates a retainer from an assessment function (`assess_func`)
  and a storage function (`storage_func`).
- `dropout`: Wraps a retainer to filter cases below a `min_similarity` threshold.

Storage Functions:
- `static`: Generates keys from a fixed reference casebase to avoid key collisions.
  Takes a `key_func` that computes the next key from existing keys.
- `indexable`: Keeps an `IndexableFunc`'s index in sync with the casebase
  when new cases are retained.

Applying Retainers:
- `apply_result`: Applies the retainer to a revise result.
- `apply_query`: Applies the retainer to a single query against a casebase.
- `apply_queries`: Applies the retainer to multiple queries.
- `apply_batches`: Applies the retainer to batches of (casebase, query) pairs.
- `apply_pair`: Applies the retainer to a single (case, query) pair.

Types:
- `KeyFunc`: Protocol for functions that generate new casebase keys.

Example:
    >>> from cbrkit.sim.generic import equality
    >>> casebase = {0: "a", 1: "b"}
    >>> retainer = build(
    ...     assess_func=equality(),
    ...     storage_func=static(
    ...         key_func=lambda keys: max(keys, default=-1) + 1,
    ...         casebase=casebase,
    ...     ),
    ... )
    >>> result = apply_query(casebase, "a", retainer)
    >>> len(result.casebase)
    4
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
