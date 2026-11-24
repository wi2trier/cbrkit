"""Revise phase: assessing solutions and repairing the failing ones.

A reviser assesses every solution for its query with a measure,
repairs those below its minimal quality with an adaptation, and assesses them again,
so it follows a reuser in `cbrkit.cycle` and precedes a `cbrkit.retain.Retainer`.

- `Reviser`: Assesses solutions with `measure` and repairs failing ones with `repair`.

Example:
    >>> from cbrkit.cycle import run_query
    >>> from cbrkit.sim.generic import Equality
    >>> reviser = Reviser(measure=Equality())
    >>> run_query({0: "a", 1: "b"}, "a", reviser).similarities
    {0: 1.0, 1: 0.0}
"""

from .build import Reviser

__all__ = ["Reviser"]
