from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from .._phase import Ranking, problems, required, score_pairs
from ..schema import PythonOnly, serializable
from ..sim import Measure
from ..typing import Batches, Casebase, Float, GroupCasebase, Results

__all__ = ["Retriever", "as_groups"]


def as_groups[K, V](casebase: Casebase[K, V]) -> GroupCasebase[K, V]:
    """View a casebase as a casebase of single-case groups.

    Grouped retrieval represents a group of source cases as a single case, which
    lets a retriever score a relation over several cases at once, for instance
    whether a query lies between a pair of cases.
    This function is the entry point into that representation and
    `cbrkit.retrieve.Group` then combines them into wider groups.
    Queries have to be wrapped in a single-case group as well.

    Args:
        casebase: Source casebase to view as groups.

    Returns:
        A casebase mapping each source key to a group holding only that case.

    Examples:
        >>> as_groups({"a": 1, "b": 2})
        {('a',): (1,), ('b',): (2,)}
    """
    return {(key,): (case,) for key, case in casebase.items()}


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Retriever[V, S: Float](Ranking):
    """Retrieves the cases most similar to the query, the retrieve phase of the CBR cycle.

    It compares the problems of the cases and the query, see `cbrkit.model.Case`.
    It flattens all (casebase, query) batches into a single list of pairwise
    comparisons and evaluates them together using the similarity function.
    With an executor, parallelism occurs at the level of
    chunks of similarity computations within batches.
    To parallelize across batches instead (i.e., process each (casebase, query)
    pair independently), wrap it with `Distribute`.
    It is the `retrieve` section of a `cbrkit.spec.Config`, which fills in a missing measure.
    It keeps the best cases by the fields of `cbrkit.retrieve.Ranking`.

    Args:
        measure: Similarity of the problem of a case to that of the query, needed to retrieve,
            the measure of the configuration in a `cbrkit.spec.Config` if None.
        executor: Runs chunks of pairs in parallel, such as a `ProcessPoolExecutor`,
            `ThreadPoolExecutor` or `InterpreterPoolExecutor` that the caller owns,
            in the current thread if None.
        chunksize: Number of pairs per call of the function, all pairs in one call
            without an executor and one chunk per available CPU with one if None.

    Examples:
        >>> import cbrkit
        >>> retriever = cbrkit.retrieve.Retriever(
        ...     measure=cbrkit.sim.AttributeValue(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.Linear(max_distance=100000),
        ...             "year": cbrkit.sim.numbers.Linear(max_distance=50),
        ...         },
        ...     ),
        ...     limit=10,
        ... )
    """

    measure: Measure[Any, S] | None = None
    executor: PythonOnly[Executor | None] = None
    chunksize: PythonOnly[int | None] = None

    def __call__[K](self, batches: Batches[K, V], /) -> Results[K, V, S]:
        scores = score_pairs(
            problems(batches),
            required(self.measure, "measure"),
            self.executor,
            self.chunksize,
        )

        return self.select_scores(batches, scores)
