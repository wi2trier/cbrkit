"""Reuse phase for adapting retrieved cases and scoring the adapted results.

The reuse phase takes the output of retrieval and applies adaptation functions
to modify retrieved cases, then scores the adapted cases using a similarity
function to assess adaptation quality.

Building Reusers:
    ``build``: Creates a reuser from an adaptation function and a similarity function.
    The adaptation function transforms cases, and the similarity function scores the result.

Applying Reusers:
    ``apply_result``: Applies the reuser to a retrieval result.
    ``apply_query``: Applies the reuser to a single query against a casebase.
    ``apply_queries``: Applies the reuser to multiple queries.
    ``apply_batches``: Applies the reuser to batches of (casebase, query) pairs.
    ``apply_pair``: Applies the reuser to a single (case, query) pair.

Multiple reusers can be composed by passing them as a list or tuple,
producing a multi-step pipeline with ``final_step`` and ``steps`` attributes.

Example:
    Build and apply a reuser::

        import cbrkit

        reuser = cbrkit.reuse.build(
            adaptation_func=cbrkit.adapt.attribute_value(...),
            similarity_func=cbrkit.sim.attribute_value(...),
        )
        result = cbrkit.reuse.apply_result(retrieval_result, reuser)
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
