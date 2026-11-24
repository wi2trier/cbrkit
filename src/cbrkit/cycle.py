"""Runs the functions of the CBR cycle on queries, one function after another.

Retrievers, reusers, revisers, and retainers share one protocol, `cbrkit.typing.CbrFunc`:
a function takes (casebase, query) batches and returns the cases it keeps with their scores,
so any of them can follow any other, for instance a reranker after a retriever
or a reviser after a reuser.
Each function works on the cases its predecessor kept, and the result records one step per function.
A `cbrkit.retain.Retainer` closes the cycle by storing the cases that reach it,
so it follows a reuser, since only reuse turns retrieved cases into solutions of the query,
and whose changes `cbrkit.model.QueryResult.changes` collects,
while the maintenance of the whole case base is no step of the cycle, see `cbrkit.maintain`.
Cases and queries are of the same type, split into problem and solution if they are `cbrkit.model.Case`s.

`run_async` is the one engine: it awaits the functions with effects (`cbrkit.typing.AsyncCbrFunc`),
such as the retrievers of databases, rerankers, and retainers,
and runs the pure ones (`cbrkit.typing.CbrFunc`) in a worker thread with `cbrkit.helpers.to_thread`,
from which measures that embed or generate await their models on the event loop of the run.
`run` is `asyncio.run` around it for synchronous code, and `run_query` and `run_query_async` run one query.

The source of the cases is a casebase, a storage (`cbrkit.typing.Index`) whose current cases are read once per run,
or None if the first function searches its own storage, such as a retriever of `cbrkit.indexable`.
An empty casebase holds no cases.

Example:
    >>> from cbrkit.retrieve import Retriever
    >>> from cbrkit.sim.generic import Equality
    >>> result = run_query(
    ...     {"a": "x", "b": "y", "c": "x"},
    ...     "x",
    ...     Retriever(measure=Equality(), min_similarity=0.5),
    ... )
    >>> result.ranking
    ('a', 'c')
    >>> [hit.key for hit in result.hits(limit=1)]
    ['a']
"""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import fields, is_dataclass
from functools import cache
from timeit import default_timer
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError

from ._phase import call
from ._util import get_name
from .adapt import ADAPTATIONS
from .model import QueryResult, Result, Step
from .schema import using
from .sim import MEASURES
from .typing import AnyCbrFunc, Casebase, Float, Index, JsonEntry

__all__ = [
    "Source",
    "get_metadata",
    "run",
    "run_async",
    "run_query",
    "run_query_async",
]

logger = logging.getLogger(__name__)

type Source[K, V] = Casebase[K, V] | Index[K, V] | None
"""The cases of a run: a casebase, a storage whose current cases are read once,
or None if the first function searches its own storage."""


@cache
def _adapter(cls: type) -> TypeAdapter[Any]:
    # the built-in measures and adaptations that the nested fields of the phases are declared by name for
    with using(MEASURES, ADAPTATIONS):
        return TypeAdapter(cls)


def get_metadata(obj: object) -> JsonEntry:
    """The name and configuration of a function as JSON, such as the metadata of the steps of a result.

    A `cbrkit.schema.serializable` function, such as a similarity measure or a phase, gives its serialized form,
    other dataclasses their name and fields, other functions their name,
    and wrappers such as `batchify` the metadata of the wrapped function.

    Examples:
        >>> from cbrkit.helpers import batchify
        >>> from cbrkit.sim.numbers import Linear
        >>> get_metadata(batchify(Linear(max_distance=10)))
        {'max_distance': 10.0, 'min_distance': 0.0, 'kind': 'linear'}
        >>> get_metadata(len)
        {'name': 'len'}
    """
    if (wrapped := getattr(obj, "__wrapped__", None)) is not None:
        return get_metadata(wrapped)

    if is_dataclass(obj) and "__pydantic_config__" in vars(type(obj)):
        # values that do not serialize by the type of the class, such as custom functions, fall back to the fields
        with suppress(PydanticSerializationError, ValueError):
            return _adapter(type(obj)).dump_python(obj, mode="json", warnings="error")

    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            "name": get_name(obj),
            "metadata": {
                field.name: get_metadata(getattr(obj, field.name))
                for field in fields(obj)
                if field.repr
            },
        }

    if callable(obj):
        return {"name": get_name(obj)}

    if isinstance(obj, Mapping):
        return {str(key): get_metadata(value) for key, value in obj.items()}

    if isinstance(obj, list | tuple):
        return [get_metadata(value) for value in obj]

    if isinstance(obj, str | int | float | bool):
        return obj

    return None


async def _casebase[K, V](source: Source[K, V]) -> Casebase[K, V]:
    if source is None:
        return {}

    if isinstance(source, Mapping):
        return source

    return await source.get_index()


def run[Q, K, V, S: Float](
    source: Source[K, V],
    queries: Mapping[Q, V],
    *funcs: AnyCbrFunc[K, V, S],
) -> Result[Q, K, V, S]:
    """Runs the functions on several queries against the same cases, `asyncio.run` around `run_async`.

    Args:
        source: The cases, a storage whose current cases are read once,
            or None if the first function searches its own storage.
        queries: The queries by their keys.
        funcs: The functions of the cycle in the order in which they run.

    Raises:
        ValueError: If no function is given.
        RuntimeError: If an event loop runs in the current thread, use `run_async` there.

    Examples:
        >>> import cbrkit
        >>> import polars as pl
        >>> casebase = cbrkit.io.Polars(pl.read_csv("./data/cars-1k.csv"))
        >>> retriever = cbrkit.retrieve.Retriever(
        ...     measure=cbrkit.sim.AttributeValue(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.Linear(max_distance=100000),
        ...             "year": cbrkit.sim.numbers.Linear(max_distance=50),
        ...         },
        ...     ),
        ...     limit=3,
        ... )
        >>> result = run(casebase, {"a": casebase[42], "b": casebase[7]}, retriever)
        >>> result["a"].ranking[0], len(result), len(result.steps)
        (42, 2, 1)
    """
    return asyncio.run(run_async(source, queries, *funcs))


async def run_async[Q, K, V, S: Float](
    source: Source[K, V],
    queries: Mapping[Q, V],
    *funcs: AnyCbrFunc[K, V, S],
) -> Result[Q, K, V, S]:
    """Runs the functions on several queries, awaiting those with effects and running pure ones in a worker thread.

    Args:
        source: The cases, a storage whose current cases are read once,
            or None if the first function searches its own storage.
        queries: The queries by their keys.
        funcs: The functions of the cycle in the order in which they run.

    Raises:
        ValueError: If no function is given.
    """
    if not funcs:
        raise ValueError("At least one function is required")

    casebase = await _casebase(source)
    batches = {key: (casebase, query) for key, query in queries.items()}
    steps: list[Step[Q, QueryResult[K, V, S]]] = []
    start = default_timer()

    # cases without a score are dropped, so filtering functions shrink the casebase
    for i, func in enumerate(funcs, start=1):
        logger.info("Running %s (%d/%d)", get_name(func), i, len(funcs))
        step_start = default_timer()
        inputs = list(batches.values())
        outputs = await call(func, inputs)
        # the outputs are already typed, so they skip the validation
        results: dict[Q, QueryResult[K, V, S]] = {
            key: QueryResult.model_construct(
                query=query,
                casebase={case_key: cases[case_key] for case_key in similarities},
                similarities=similarities,
            )
            for (key, (_, query)), (cases, similarities) in zip(
                batches.items(), outputs, strict=True
            )
        }
        steps.append(
            Step(
                outputs=results,
                metadata=get_metadata(func),
                duration=default_timer() - step_start,
            )
        )
        batches = {key: (entry.casebase, entry.query) for key, entry in results.items()}

    return Result(steps=tuple(steps), duration=default_timer() - start)


def run_query[K, V, S: Float](
    source: Source[K, V],
    query: V,
    *funcs: AnyCbrFunc[K, V, S],
) -> QueryResult[K, V, S]:
    """Runs the functions on one query and returns the result of the last one.

    Use `run` to keep the results of the intermediate functions.

    Args:
        source: The cases, a storage whose current cases are read once,
            or None if the first function searches its own storage.
        query: The query.
        funcs: The functions of the cycle in the order in which they run.

    Raises:
        ValueError: If no function is given.

    Examples:
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.retrieve import Retriever
        >>> from cbrkit.sim.generic import Equality
        >>> run_query(InMemory({0: "a", 1: "b"}), "a", Retriever(measure=Equality())).similarities
        {0: 1.0, 1: 0.0}
        >>> run_query({}, "a", Retriever(measure=Equality())).similarities
        {}
    """
    return run(source, {0: query}, *funcs)[0]


async def run_query_async[K, V, S: Float](
    source: Source[K, V],
    query: V,
    *funcs: AnyCbrFunc[K, V, S],
) -> QueryResult[K, V, S]:
    """Runs the functions on one query and returns the result of the last one, see `run_async`.

    Args:
        source: The cases, a storage whose current cases are read once,
            or None if the first function searches its own storage.
        query: The query.
        funcs: The functions of the cycle in the order in which they run.

    Raises:
        ValueError: If no function is given.
    """
    return (await run_async(source, {0: query}, *funcs))[0]
