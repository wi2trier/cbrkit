"""Wrappers that convert, expand, combine, and distribute the batches of functions of the cycle.

Every wrapper is asynchronous and wraps synchronous and asynchronous functions alike,
awaiting those with effects and running pure ones in a thread of their own like `cbrkit.cycle.run_async`.
"""

import asyncio
import itertools
import os
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, override

from .._phase import Ranking, call, call_all, is_async
from .._util import get_value
from ..helpers import bounded_gather, to_thread
from ..sim.aggregator import Aggregator
from ..typing import (
    AnyCbrFunc,
    AsyncCbrFunc,
    Batches,
    Casebase,
    ConversionFunc,
    Float,
    GroupCasebase,
    HasValue,
    Results,
    SimMap,
)

__all__ = [
    "Combine",
    "Distribute",
    "Group",
    "Transpose",
    "transpose_value",
]


def _per_casebase[C, R](casebases: Sequence[C], func: Callable[[C], R]) -> list[R]:
    """Applies a function once per distinct casebase object, since queries commonly share one."""
    results: dict[int, R] = {}

    for casebase in casebases:
        if id(casebase) not in results:
            results[id(casebase)] = func(casebase)

    return [results[id(casebase)] for casebase in casebases]


@dataclass(slots=True, frozen=True, kw_only=True)
class Transpose[K, V1, V2, S: Float](AsyncCbrFunc[K, V1, S]):
    """Transforms a retriever function from one type to another.

    Useful when the input values need to be converted before retrieval,
    for instance, when the cases are nested and you only need to compare a subset of the values.
    The retriever has to score the converted cases it is given,
    so a retriever that searches its own storage cannot be transposed.

    Args:
        retriever: The retriever function to be used on the converted values.
        conversion: A function that converts the input values from one type to another.

    Examples:
        >>> from cbrkit.cycle import run_query
        >>> from cbrkit.retrieve import Retriever
        >>> from cbrkit.sim.generic import Equality
        >>> retriever = Transpose(retriever=Retriever(measure=Equality()), conversion=str.lower)
        >>> run_query({"a": "X", "b": "y"}, "x", retriever).similarities
        {'a': 1.0, 'b': 0.0}
    """

    retriever: AnyCbrFunc[K, V2, S]
    conversion: ConversionFunc[V1, V2]

    @override
    async def __call__(self, batches: Batches[K, V1], /) -> Results[K, V1, S]:
        casebases = _per_casebase(
            [casebase for casebase, _ in batches],
            lambda casebase: {
                key: self.conversion(value) for key, value in casebase.items()
            },
        )
        inner_results = await call(
            self.retriever,
            [
                (casebase, self.conversion(query))
                for casebase, (_, query) in zip(casebases, batches, strict=True)
            ],
        )

        return [
            (original_casebase, sim_map)
            for (original_casebase, _), (_, sim_map) in zip(
                batches, inner_results, strict=True
            )
        ]


def transpose_value[K, V, S: Float](
    retriever: AnyCbrFunc[K, V, S],
) -> AsyncCbrFunc[K, HasValue[V], S]:
    """Wrap a retriever to extract values from structured value inputs before retrieval."""
    return Transpose(retriever=retriever, conversion=get_value)


def _combine_groups[K, V](
    casebase: GroupCasebase[K, V],
    size: int,
    ordered: bool,
    filter: Callable[[tuple[V, ...]], bool] | None,
) -> dict[tuple[K, ...], tuple[V, ...]]:
    """Concatenate the groups of a grouped casebase into wider groups."""
    if size < 1:
        raise ValueError("size must be at least 1")

    combinator = itertools.permutations if ordered else itertools.combinations
    combined: dict[tuple[K, ...], tuple[V, ...]] = {}

    for group in combinator(casebase.items(), size):
        key_group: tuple[K, ...] = ()
        case_group: tuple[V, ...] = ()

        for key, case in group:
            key_group += key
            case_group += case

        if filter is None or filter(case_group):
            combined[key_group] = case_group

    return combined


@dataclass(slots=True, frozen=True, kw_only=True)
class Group[K, V, S: Float](AsyncCbrFunc[tuple[K, ...], tuple[V, ...], S]):
    """Combines the cases of a grouped casebase into wider groups before retrieval.

    Takes the casebase produced by `cbrkit.retrieve.as_groups` and enumerates
    every group of `size` cases, which the wrapped retriever then scores against
    the query.
    Because its input and output casebases have the same type, it composes
    inside a retriever chain, so a cheap retriever can narrow down the source
    cases before this combinatorial expansion happens.
    The expansion is materialized per query, producing `n choose size`
    groups, or `n! / (n - size)!` when ordered.

    Args:
        retriever: The retriever function applied to the combined groups.
        size: Number of cases combined into each group.
        ordered: Whether different orders of the same cases produce separate
            groups, i.e. permutations instead of combinations.
        filter: Optional predicate that decides whether a group is kept.
            Groups are discarded as they are created, so a selective predicate
            also bounds the memory needed to hold the expansion.

    Examples:
        >>> import cbrkit
        >>> casebase = cbrkit.retrieve.as_groups({"a": 0.0, "b": 10.0})
        >>> def between(case, query):
        ...     return float(min(case) <= cbrkit.helpers.singleton(query) <= max(case))
        >>> retriever = Group(retriever=cbrkit.retrieve.Retriever(measure=between), size=2)
        >>> result = cbrkit.cycle.run_query(casebase, (4.0,), retriever)
        >>> result.ranking
        (('a', 'b'),)
    """

    # the keys are left open, since the retrievers are generic in them
    retriever: AnyCbrFunc[Any, tuple[V, ...], S]
    size: int = 2
    ordered: bool = False
    filter: Callable[[tuple[V, ...]], bool] | None = None

    @override
    async def __call__(
        self, batches: Batches[tuple[K, ...], tuple[V, ...]], /
    ) -> Results[tuple[K, ...], tuple[V, ...], S]:
        expanded = _per_casebase(
            [casebase for casebase, _ in batches],
            lambda casebase: _combine_groups(
                casebase, self.size, self.ordered, self.filter
            ),
        )

        return await call(
            self.retriever,
            [
                (casebase, query)
                for casebase, (_, query) in zip(expanded, batches, strict=True)
            ],
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class Combine[K, V, S: Float](Ranking, AsyncCbrFunc[K, V, float]):
    """Combines multiple retriever functions into one by aggregating their similarities.

    To merge rankings rather than similarities, use `Fuse`.
    The retrievers run concurrently, and the combined cases are those that the retrievers return,
    so retrievers that search their own storages can be combined.
    If multiple retrievers return different values for the same key,
    the value of the last retriever in order wins.

    Args:
        retrievers: The retriever functions to be combined.
        aggregator: Aggregates the similarities of a case from the retrievers,
            weighted by the positions of the retrievers.
        default: Similarity of a case that a retriever did not return,
            only cases that every retriever returns are kept if None.

    Examples:
        >>> from cbrkit.cycle import run_query
        >>> from cbrkit.retrieve import Retriever
        >>> first = Retriever(measure=lambda case, query: 1.0, min_similarity=1.0)
        >>> second = Retriever(measure=lambda case, query: float(case == query), min_similarity=1.0)
        >>> run_query({"a": 1, "b": 2}, 1, Combine(retrievers=[first, second])).similarities
        {'a': 1.0, 'b': 0.5}
        >>> run_query({"a": 1, "b": 2}, 1, Combine(retrievers=[first, second], default=None)).similarities
        {'a': 1.0}
    """

    retrievers: Sequence[AnyCbrFunc[K, V, S]]
    aggregator: Aggregator[int] = field(default_factory=Aggregator[int])
    default: float | None = 0.0

    @override
    async def __call__(self, batches: Batches[K, V], /) -> Results[K, V]:
        results: list[tuple[Casebase[K, V], SimMap[K, float]]] = []

        for rankings in await call_all(self.retrievers, batches):
            sim_maps = [sim_map for _, sim_map in rankings]
            keys = list(dict.fromkeys(key for sim_map in sim_maps for key in sim_map))

            if self.default is None:
                keys = [key for key in keys if all(key in m for m in sim_maps)]

            combined: dict[K, float] = {}

            for key in keys:
                sims: list[S | float] = [
                    sim_map.get(key, self.default or 0.0) for sim_map in sim_maps
                ]
                combined[key] = self.aggregator(sims)

            casebase = {
                key: value
                for cases, _ in rankings
                for key, value in cases.items()
                if key in combined
            }
            results.append((casebase, combined))

        return self.select(results)


@dataclass(slots=True, frozen=True, kw_only=True)
class Distribute[K, V, S: Float](AsyncCbrFunc[K, V, S]):
    """Wrapper that parallelizes retrieval across batches.

    While `Retriever` flattens all batches into a single list of pairwise
    comparisons and parallelizes individual similarity computations,
    `Distribute` calls the wrapped retriever separately for each
    (casebase, query) pair and runs those calls concurrently.

    Use this when you have many independent queries and want to process
    them in parallel as separate retrieval tasks.

    Args:
        retriever: The retriever function to be used, typically a `Retriever`.
        executor: Runs the calls of a synchronous retriever in parallel, such as a `ProcessPoolExecutor`
            that the caller owns, every call in a thread of its own if None,
            at most one per available CPU at a time.
            The calls of an asynchronous retriever are awaited concurrently.
    """

    retriever: AnyCbrFunc[K, V, S]
    executor: Executor | None = None

    @override
    async def __call__(self, batches: Batches[K, V], /) -> Results[K, V, S]:
        retriever = self.retriever

        if is_async(retriever):
            results = await asyncio.gather(*(retriever([batch]) for batch in batches))
        elif self.executor is None:
            results = await bounded_gather(
                batches,
                lambda batch: to_thread(retriever, [batch]),
                limit=os.process_cpu_count() or 1,
            )
        else:
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(
                *(
                    loop.run_in_executor(self.executor, retriever, [batch])
                    for batch in batches
                )
            )

        return [result[0] for result in results]
