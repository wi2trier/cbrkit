from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from .._phase import Ranking, adapt_pairs, problems, required, reused, score_pairs
from ..adapt import Adaptation
from ..schema import PythonOnly, serializable
from ..sim import Measure
from ..typing import Batches, Float, Results

__all__ = ["Reuser"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Reuser[V, S: Float](Ranking):
    """Adapts every case to the query and scores the adapted cases, the reuse phase of the CBR cycle.

    An unsplit case is scored as adapted.
    A split `cbrkit.model.Case` reuses the adapted solution for the problem of the query
    and is scored by the similarity of its source problem to the query,
    since its new problem is the one of the query.
    It is the `reuse` section of a `cbrkit.spec.Config`, which fills in a missing measure.
    It keeps the best adapted cases by the fields of `cbrkit.retrieve.Ranking`,
    the one best by default, since reuse proposes the solution of the new problem,
    which revise assesses and retain learns, after Aamodt and Plaza (1994).

    Args:
        adaptation: Adapts a case to the query, any adaptation of `cbrkit.adapt`.
        limit: Number of solutions it proposes per query, every adapted case if None.
        measure: Similarity of a problem to the one of the query, needed to reuse,
            the measure of the configuration in a `cbrkit.spec.Config` if None.
        executor: Runs chunks of pairs in parallel, such as a `ProcessPoolExecutor`,
            `ThreadPoolExecutor` or `InterpreterPoolExecutor` that the caller owns,
            in the current thread if None.
        chunksize: Number of pairs per call of the function, all pairs in one call
            without an executor and one chunk per available CPU with one if None.

    Examples:
        >>> from cbrkit.adapt.generic import Null
        >>> from cbrkit.model import Case
        >>> from cbrkit.sim.generic import Equality
        >>> Reuser(adaptation=Null(target="query"), measure=Equality())([({"a": "x"}, "y")])
        [({'a': 'y'}, {'a': 1.0})]
        >>> upper = lambda case, query: Case(problem=case.problem, solution=case.solution.upper())
        >>> Reuser(adaptation=upper, measure=Equality())([({"a": Case(problem="x", solution="fix")}, Case(problem="y"))])
        [({'a': Case(problem='y', solution='FIX')}, {'a': 0.0})]
    """

    adaptation: Adaptation[V]
    limit: int | None = 1
    measure: Measure[Any, S] | None = None
    executor: PythonOnly[Executor | None] = None
    chunksize: PythonOnly[int | None] = None

    def __call__[K](self, batches: Batches[K, V], /) -> Results[K, V, S]:
        adapted = adapt_pairs(batches, self.adaptation, self.executor, self.chunksize)
        reuses = [
            {key: reused(case, query, cases[key]) for key, case in casebase.items()}
            for (casebase, query), cases in zip(batches, adapted, strict=True)
        ]

        scores = score_pairs(
            problems(
                [
                    ({key: scored for key, (_, scored) in entries.items()}, query)
                    for entries, (_, query) in zip(reuses, batches, strict=True)
                ]
            ),
            required(self.measure, "measure"),
            self.executor,
            self.chunksize,
        )

        return self.select(
            zip(
                [
                    {key: case for key, (case, _) in entries.items()}
                    for entries in reuses
                ],
                scores,
                strict=True,
            )
        )
