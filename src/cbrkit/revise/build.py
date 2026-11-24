from concurrent.futures import Executor
from dataclasses import dataclass

from .._phase import Ranking, adapt_pairs, required, score_pairs
from ..adapt import Adaptation
from ..schema import PythonOnly, serializable
from ..sim import Measure
from ..typing import Batches, Float, Results

__all__ = ["Reviser"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Reviser[V, S: Float](Ranking):
    """Assesses solutions and repairs the failing ones, the revise phase of the CBR cycle.

    After Aamodt and Plaza (1994), every solution is assessed first,
    those below `min_similarity` are repaired and assessed again,
    and those that still fail are left out.
    The measure compares whole cases, so it sees the solution of a split `cbrkit.model.Case` and the problem of the query.
    It is the `revise` section of a `cbrkit.spec.Config`, which fills in a missing measure.
    It keeps the best solutions by the fields of `cbrkit.retrieve.Ranking`.

    Args:
        measure: Quality of a solution for its query, needed to revise,
            the measure of the configuration in a `cbrkit.spec.Config` if None.
        repair: Adapts a failing solution to its query before it is assessed again, any adaptation of `cbrkit.adapt`,
            needs a `min_similarity` that tells the failing solutions.
        min_similarity: Minimal quality of an accepted solution, every solution is accepted if None.
        executor: Runs chunks of pairs in parallel, such as a `ProcessPoolExecutor`,
            `ThreadPoolExecutor` or `InterpreterPoolExecutor` that the caller owns,
            in the current thread if None.
        chunksize: Number of pairs per call of the function, all pairs in one call
            without an executor and one chunk per available CPU with one if None.

    Raises:
        ValueError: If a repair is given without a `min_similarity`.

    Examples:
        >>> from cbrkit.adapt.generic import Null
        >>> from cbrkit.sim.generic import Equality
        >>> Reviser(measure=Equality(), min_similarity=1.0)([({0: "a", 1: "b"}, "a")])
        [({0: 'a', 1: 'b'}, {0: 1.0})]
        >>> Reviser(measure=Equality(), repair=Null(target="query"), min_similarity=1.0)([({0: "a", 1: "b"}, "a")])
        [({0: 'a', 1: 'a'}, {0: 1.0, 1: 1.0})]
    """

    measure: Measure[V, S] | None = None
    repair: Adaptation[V] | None = None
    executor: PythonOnly[Executor | None] = None
    chunksize: PythonOnly[int | None] = None

    def __post_init__(self) -> None:
        if self.repair is not None and self.min_similarity is None:
            raise ValueError(
                "A repair needs a min_similarity that tells the failing solutions"
            )

    def __call__[K](self, batches: Batches[K, V], /) -> Results[K, V, S]:
        measure = required(self.measure, "measure")
        scores = score_pairs(batches, measure, self.executor, self.chunksize)
        casebases = [dict(casebase) for casebase, _ in batches]

        if self.repair is not None and self.min_similarity is not None:
            failing = [
                (
                    {
                        key: case
                        for key, case in casebase.items()
                        if float(similarities[key]) < self.min_similarity
                    },
                    query,
                )
                for (casebase, query), similarities in zip(batches, scores, strict=True)
            ]
            repaired = [
                (cases, query)
                for cases, (_, query) in zip(
                    adapt_pairs(failing, self.repair, self.executor, self.chunksize),
                    failing,
                    strict=True,
                )
            ]
            rescored = score_pairs(repaired, measure, self.executor, self.chunksize)

            for casebase, similarities, (cases, _), new in zip(
                casebases, scores, repaired, rescored, strict=True
            ):
                casebase.update(cases)
                similarities.update(new)

        return self.select(zip(casebases, scores, strict=True))
