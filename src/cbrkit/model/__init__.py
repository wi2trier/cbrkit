"""Core data models for CBR result structures and graph representations.

This module provides the data classes used throughout CBRkit to represent
the outputs of each CBR phase and full cycle results.

Result Classes:
    ``QueryResultStep``: Holds the result of a single query against a casebase,
    including ``similarities``, ``ranking``, ``casebase``, and ``query``.
    ``ResultStep``: Aggregates ``QueryResultStep`` instances across multiple queries.
    ``Result``: Wraps a sequence of ``ResultStep`` instances representing a
    multi-step pipeline (e.g., sequential retrievers).
    ``CycleResult``: Contains the results from all four CBR phases
    (retrieval, reuse, revise, retain).

Graph Module:
    The ``cbrkit.model.graph`` submodule provides a ``Graph`` data structure
    with ``Node`` and ``Edge`` types, serialization helpers (``to_dict``,
    ``from_dict``), and conversions to NetworkX, RustWorkX, and Graphviz.

Example:
    Accessing results after retrieval::

        result = cbrkit.retrieval.apply_query(casebase, query, retriever)
        print(result.ranking)       # case keys sorted by similarity
        print(result.similarities)  # {key: score, ...}
        print(result.casebase)      # filtered casebase with retrieved cases
"""

from . import graph, result
from .result import CycleResult, QueryResultStep, Result, ResultStep

__all__ = [
    "result",
    "graph",
    "QueryResultStep",
    "ResultStep",
    "Result",
    "CycleResult",
]
