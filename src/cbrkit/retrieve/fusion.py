"""Fusion of several rankings into one."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .._phase import Ranking, call_all
from .._util import RRF_K, fuse_rankings
from ..model import Hit
from ..typing import AnyCbrFunc, AsyncCbrFunc, Batches, Float, Results

__all__ = ["Fuse", "fuse"]


def fuse[K, V, S: Float](
    rankings: Iterable[Sequence[Hit[K, V, S]]],
    /,
    *,
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[Hit[K, V, float]]:
    """Fuses several rankings by weighted reciprocal rank fusion.

    Every ranking orders its hits by similarity,
    and a case scores the sum of `weight / (k + rank)` over the rankings containing it.
    Ranks stay comparable when similarities are not,
    for instance across the results of several queries, retrievers, or memories.

    Args:
        rankings: The rankings to fuse, such as the `hits()` of the query results of a run.
        weights: Weight of every ranking, 1 for each if None.
        k: Smoothing constant, larger values narrow the gap between top ranks.
        limit: Maximum number of fused hits, all if None.

    Returns:
        The fused hits by descending score, holding the first case seen for every key.

    Examples:
        >>> fused = fuse(
        ...     [
        ...         [Hit(key="a", value=1, score=0.9), Hit(key="b", value=2, score=0.1)],
        ...         [Hit(key="b", value=2, score=0.8), Hit(key="c", value=3, score=0.5)],
        ...     ],
        ...     k=0,
        ... )
        >>> [(hit.key, hit.score) for hit in fused]
        [('b', 1.5), ('a', 1.0), ('c', 0.5)]
    """
    casebase, scores = fuse_rankings(
        (
            (
                {hit.key: hit.value for hit in ranking},
                {hit.key: hit.score for hit in ranking},
            )
            for ranking in rankings
        ),
        weights,
        k,
        limit,
    )

    return [
        Hit.model_construct(key=key, value=casebase[key], score=score)
        for key, score in scores.items()
    ]


@dataclass(slots=True, frozen=True, kw_only=True)
class Fuse[K, V](Ranking, AsyncCbrFunc[K, V, float]):
    """Fuses the rankings of several retriever functions by weighted reciprocal rank fusion, see `fuse`.

    Unlike `cbrkit.retrieve.Combine`, which aggregates similarities and thus assumes them comparable,
    it merges ranks, so retrievers of incomparable scores fuse well, such as a dense and a sparse one.
    The retrievers run concurrently, and the fused cases are those that the retrievers return,
    the value of the first retriever in order winning for a key.

    Args:
        retrievers: The retriever functions whose rankings are fused.
        weights: Weight of every ranking, 1 for each if None.
        k: Smoothing constant, larger values narrow the gap between top ranks.

    Examples:
        >>> from cbrkit.cycle import run_query
        >>> from cbrkit.retrieve import Retriever
        >>> first = Retriever(measure=lambda case, query: float(case == query))
        >>> second = Retriever(measure=lambda case, query: float(case > query))
        >>> retriever = Fuse(retrievers=[first, second], weights=[2, 1], k=0)
        >>> run_query({"a": 1, "b": 2}, 1, retriever).similarities
        {'a': 2.5, 'b': 2.0}
    """

    retrievers: Sequence[AnyCbrFunc[K, V, Float]]
    weights: Sequence[float] | None = None
    k: int = RRF_K

    async def __call__(self, batches: Batches[K, V], /) -> Results[K, V]:
        return self.select(
            fuse_rankings(rankings, self.weights, self.k)
            for rankings in await call_all(self.retrievers, batches)
        )
