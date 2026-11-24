"""Shared scoring, adaptation, selection, and calls of the retrieval, reuse, revise, and retain functions."""

import asyncio
import inspect
import itertools
import math
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import Any, TypeIs, cast

from ._util import sim_map2ranking
from .helpers import batchify, to_thread
from .model import Case, problem
from .typing import (
    AnyAdaptationFunc,
    AnyCbrFunc,
    AnySimFunc,
    AsyncCbrFunc,
    Batches,
    Casebase,
    Float,
    Results,
    Sim,
    SimMap,
)


def required[T](value: T | None, name: str) -> T:
    """The value of a field that a phase needs to run.

    Raises:
        ValueError: If the value is None.
    """
    if value is None:
        raise ValueError(f"No {name} is configured")

    return value


def map_pairs[K, V, R](
    batches: Batches[K, V],
    func: Callable[[Sequence[tuple[V, V]]], Sequence[R]],
    executor: Executor | None = None,
    chunksize: int | None = None,
) -> list[dict[K, R]]:
    """Apply a batch function to every (case, query) pair and regroup the results per batch.

    All pairs of all batches are flattened into one list,
    so the function sees as many pairs at once as possible.
    With an executor or a chunksize, the list is split into chunks
    that are processed independently, by default one chunk per available CPU.
    """
    index = [
        (idx, key) for idx, (casebase, _) in enumerate(batches) for key in casebase
    ]
    pairs = [(case, query) for casebase, query in batches for case in casebase.values()]

    if executor is None and chunksize is None:
        values: Sequence[R] = func(pairs)
    else:
        cpus = os.process_cpu_count() or 1
        size = chunksize or max(1, math.ceil(len(pairs) / cpus))
        chunks = list(itertools.batched(pairs, size))
        mapped = map(func, chunks) if executor is None else executor.map(func, chunks)
        values = list(itertools.chain.from_iterable(mapped))

    results: list[dict[K, R]] = [{} for _ in batches]

    for (idx, key), value in zip(index, values, strict=True):
        results[idx][key] = value

    return results


def score_pairs[K, V, S: Float](
    batches: Batches[K, V],
    func: AnySimFunc[V, S],
    executor: Executor | None = None,
    chunksize: int | None = None,
) -> list[dict[K, S]]:
    """Score every case against the query of its batch."""
    return map_pairs(batches, batchify(func), executor, chunksize)


def adapt_pairs[K, V](
    batches: Batches[K, V],
    func: AnyAdaptationFunc[V],
    executor: Executor | None = None,
    chunksize: int | None = None,
) -> list[dict[K, V]]:
    """Adapt every case to the query of its batch."""
    return map_pairs(batches, batchify(func), executor, chunksize)


def problems[K](batches: Batches[K, Any]) -> list[tuple[dict[K, Any], Any]]:
    """The batches with the problems of their cases and queries, see `cbrkit.model.problem`."""
    return [
        ({key: problem(case) for key, case in casebase.items()}, problem(query))
        for casebase, query in batches
    ]


def reused[V](case: V, query: V, adapted: V) -> tuple[V, Any]:
    """The case that reuses an adapted case for a query and the problem that is scored against the query.

    A split case solves the problem of the query with the adapted solution and is scored by its source problem,
    since its new problem is the one of the query,
    while an unsplit case is its own problem and solution.
    """
    if isinstance(case, Case) and isinstance(query, Case) and isinstance(adapted, Case):
        return replace(adapted, problem=query.problem), case.problem

    return adapted, adapted


def normalize[K, S: Float](similarities: SimMap[K, S]) -> dict[K, S]:
    """Scale the similarities min-max to between 0 and 1, keeping the details of every `cbrkit.typing.Sim`.

    When all similarities coincide, such as for a single case, every one becomes 1.0, the best score,
    rather than 0.0, which would mark a perfect single match as a non-match.

    Examples:
        >>> normalize({"a": 2.0, "b": 4.0, "c": 3.0})
        {'a': 0.0, 'b': 1.0, 'c': 0.5}
        >>> normalize({"a": Sim(value=0.4)})
        {'a': Sim(value=1.0)}
    """
    values = [float(similarity) for similarity in similarities.values()]
    low, high = (min(values), max(values)) if values else (0.0, 0.0)

    return {
        key: _rescaled(similarity, 1.0 if low == high else (value - low) / (high - low))
        for (key, similarity), value in zip(similarities.items(), values, strict=True)
    }


def _rescaled[S: Float](similarity: S, value: float) -> S:
    """The similarity with another value, keeping its type and details."""
    if isinstance(similarity, Sim):
        return replace(similarity, value=value)

    # a plain number, whose widest type is float
    return cast(S, value)


@dataclass(slots=True, frozen=True, kw_only=True)
class Ranking:
    """Selection of the best cases, shared by every ranking function of the cycle.

    Retrievers, storage retrievers, rerankers, reusers, and revisers inherit it and pass their scored cases to `select`.

    Args:
        limit: Maximal number of kept cases per query, all if None.
        min_similarity: Minimal (normalized) similarity of a kept case, any if None.
        normalize_scores: Scale the similarities of every query min-max to between 0 and 1 before the selection,
            all to 1 if they coincide, keeping the details of every `cbrkit.typing.Sim`.
    """

    limit: int | None = None
    min_similarity: float | None = None
    normalize_scores: bool = False

    def select[K, V, S: Float](
        self, results: Iterable[tuple[Casebase[K, V], SimMap[K, S]]]
    ) -> Results[K, V, S]:
        """Keep the best cases of every result, normalized first if `normalize_scores` is set.

        Args:
            results: Cases with their similarities, one pair per batch.

        Examples:
            >>> Ranking(limit=1).select([({"a": 1, "b": 2}, {"a": 0.2, "b": 0.8})])
            [({'a': 1, 'b': 2}, {'b': 0.8})]
        """
        selected: list[tuple[Casebase[K, V], SimMap[K, S]]] = []

        for casebase, similarities in results:
            if self.normalize_scores:
                similarities = normalize(similarities)

            if self.limit is not None or self.min_similarity is not None:
                ranking = sim_map2ranking(similarities, self.limit, self.min_similarity)
                similarities = {key: similarities[key] for key in ranking}

            selected.append((casebase, similarities))

        return selected

    def select_scores[K, V, S: Float](
        self, batches: Batches[K, V], scores: Iterable[SimMap[K, S]]
    ) -> Results[K, V, S]:
        """Keep the best cases of every batch by their scores, given in the order of the batches, see `select`."""
        return self.select(
            zip((casebase for casebase, _ in batches), scores, strict=True)
        )


def is_async[K, V, S: Float](
    func: AnyCbrFunc[K, V, S],
) -> TypeIs[AsyncCbrFunc[K, V, S]]:
    """Whether the function is a coroutine function or an object with an async `__call__`."""
    return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
        type(func).__call__
    )


async def call[K, V, S: Float](
    func: AnyCbrFunc[K, V, S], batches: Batches[K, V]
) -> Results[K, V, S]:
    """Run a function of the cycle, awaiting one with effects and running a pure one with `cbrkit.helpers.to_thread`.

    Every pure function runs in a thread of its own,
    from which its measures may await the models of the running event loop.

    Examples:
        >>> import asyncio
        >>> asyncio.run(call(lambda batches: [(casebase, {}) for casebase, _ in batches], [({}, "q")]))
        [({}, {})]
    """
    if is_async(func):
        return await func(batches)

    return await to_thread(func, batches)


async def call_all[K, V, S: Float](
    funcs: Iterable[AnyCbrFunc[K, V, S]], batches: Batches[K, V]
) -> list[list[tuple[Casebase[K, V], SimMap[K, S]]]]:
    """Run functions of the cycle concurrently with `call`, giving the results of all of them for every batch."""
    results = await asyncio.gather(*(call(func, batches) for func in funcs))

    return [[result[idx] for result in results] for idx in range(len(batches))]
