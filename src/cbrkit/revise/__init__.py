"""Revise phase for assessing solution quality and optionally repairing solutions.

The revise phase takes the output of reuse and evaluates the quality of
adapted solutions using a similarity function.
Optionally, a repair function can be applied to fix solutions before assessment.

Building Revisers:
    ``build``: Creates a reviser from an assessment function (``assess_func``)
    and an optional repair function (``repair_func``).
    The assessment function is a similarity function that scores solutions.
    The repair function is an adaptation function applied before assessment.

Applying Revisers:
    ``apply_result``: Applies the reviser to a reuse result.
    ``apply_query``: Applies the reviser to a single query against a casebase.
    ``apply_queries``: Applies the reviser to multiple queries.
    ``apply_batches``: Applies the reviser to batches of (casebase, query) pairs.
    ``apply_pair``: Applies the reviser to a single (case, query) pair.

Multiple revisers can be composed by passing them as a list or tuple.

Example:
    Build and apply a reviser::

        import cbrkit

        reviser = cbrkit.revise.build(
            assess_func=cbrkit.sim.attribute_value(...),
            repair_func=cbrkit.adapt.attribute_value(...),
        )
        result = cbrkit.revise.apply_result(reuse_result, reviser)
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
