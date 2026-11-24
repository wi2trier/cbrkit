"""Retain phase: storing the solutions that revise accepted in the case base.

Revise assesses solutions and retain stores those it accepted, without assessing them again,
so a `Retainer` is the last function of the cycle in `cbrkit.cycle`.
A run plans the changes of the case base, additions of solutions or merges with the stored cases they duplicate,
as one `cbrkit.model.Changes` of typed changes with their reasons,
and applies them as one mutation of a storage implementing `cbrkit.typing.Storage`,
the protocol of the storages of `cbrkit.indexable` that allocate the keys of new cases.
The storage applies changes only if it still holds the cases they were planned on,
so concurrent runs never overwrite each other, and a refused run is planned again.
Retained cases are immediately visible to runs that read the storage,
such as `cbrkit.cycle.run_query(storage, query, ...)` or the retrievers of the database storages.
Maintaining the whole case base, such as removing cases by competence, is a separate activity, see `cbrkit.maintain`.

- `Retainer`: Stores the cases of every query, a `cbrkit.model.Planner` whose `plan` plans the changes of a dry run purely,
  and `cbrkit.model.QueryResult.changes` collects the changes of a run.

Example:
    >>> from cbrkit.cycle import run_query
    >>> from cbrkit.indexable import InMemory
    >>> from cbrkit.revise import Reviser
    >>> from cbrkit.sim.generic import Equality
    >>> storage = InMemory({0: "a", 1: "b"})
    >>> retainer = Retainer(storage=storage, measure=Equality())
    >>> result = run_query({"x": "a", "y": "c"}, "a", Reviser(measure=Equality(), min_similarity=1.0), retainer)
    >>> [(change.op, change.key, change.value) for change in result.changes.items]
    [('add', 2, 'a')]
"""

from .build import Retainer, changed

__all__ = ["Retainer", "changed"]
