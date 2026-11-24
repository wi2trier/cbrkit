"""Reuse phase for adapting retrieved cases and scoring the adapted results.

A reuser adapts the retrieved cases to the query and scores the adapted cases
with a similarity function, so it follows a retriever in `cbrkit.cycle`.

Building Reusers:
- `Reuser`: Adapts every case with a pairwise adaptation function such as those of `cbrkit.adapt`
  and scores the adapted cases with a similarity function.
  An adaptation of the whole casebase is a custom `cbrkit.typing.CbrFunc`.

Example:
    >>> from cbrkit.cycle import run_query
    >>> from cbrkit.sim.generic import Equality
    >>> reuser = Reuser(
    ...     adaptation=lambda case, query: query,
    ...     measure=Equality(),
    ... )
    >>> run_query({"a": "hello"}, "world", reuser).casebase
    {'a': 'world'}
"""

from .build import Reuser

__all__ = ["Reuser"]
