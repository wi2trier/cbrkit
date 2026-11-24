"""Helpers to build functions for cbrkit, such as batch functions, lazily built functions, and immutable mappings.

Similarity and adaptation functions take the case first and the query second, `f(case, query)`,
and batch functions take a list of pairs instead, `f([(case1, query1), (case2, query2), ...])`.
Classes deriving from a batch protocol such as `cbrkit.typing.BatchSimFunc` are batch functions,
plain functions become ones with `batched`, and cbrkit never guesses this from the signature.
`lazy` builds an expensive function, such as a model that loads its weights, on its first call,
and pickles only its factory, so every worker of an executor builds its own function.
`frozendict` is an immutable and hashable mapping that guards a casebase against accidental changes.
Functions with effects, such as embedding and language models, are asynchronous,
while measures and adaptations are synchronous:
`to_thread` runs a synchronous function in a thread of its own,
from which `wait` awaits an asynchronous function on the event loop of the caller,
so a measure such as `cbrkit.sim.embed.Embedding` embeds through the clients of that loop.
`bounded_gather` runs a coroutine function on many items with bounded concurrency,
and `next_keys` gives the default keys of new cases of a storage.
"""

import asyncio
import logging
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from copy import copy
from dataclasses import dataclass, field, is_dataclass, replace
from functools import partial
from threading import Lock
from typing import Any, ClassVar, Self, TypeIs, cast, get_args, override

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import (
    CoreSchema,
    SchemaSerializer,
    core_schema,
)

from ._util import get_attribute, get_name, getitem_or_getattr, next_keys
from .typing import (
    Batched,
    BatchPairFunc,
    ColumnSimFunc,
    PairFunc,
)

logger = logging.getLogger(__name__)

__all__ = [
    "batched",
    "batchify",
    "bounded_gather",
    "broadcast_getter",
    "dist2sim",
    "frozendict",
    "get_attribute",
    "get_name",
    "getitem_or_getattr",
    "is_batched",
    "lazy",
    "next_keys",
    "replace_attributes",
    "singleton",
    "to_thread",
    "unbatchify",
    "wait",
]


class frozendict[K, V](Mapping[K, V]):
    """Immutable, hashable mapping, a stand-in for the builtin of PEP 814.

    Equal to any mapping with the same items and hashable if its values are.
    Pydantic serializes it as a dict in JSON mode, also in fields typed `Any`.

    Examples:
        >>> d = frozendict({"a": 1}, b=2)
        >>> d
        frozendict({'a': 1, 'b': 2})
        >>> d == {"a": 1, "b": 2}, hash(d) == hash(frozendict(b=2, a=1))
        (True, True)
    """

    __slots__ = ("_data", "_hash")

    _data: dict[K, V]
    _hash: int | None
    __pydantic_serializer__: ClassVar[SchemaSerializer]

    def __init__(
        self, data: Mapping[K, V] | Iterable[tuple[K, V]] = (), /, **kwargs: V
    ) -> None:
        self._data = dict(data, **kwargs)
        self._hash = None

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._data

    @override
    def __eq__(self, other: object) -> bool:
        return self._data == other

    @override
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))

        return self._hash

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __or__(self, other: Mapping[K, V]) -> Self:
        return type(self)({**self._data, **other})

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[dict[K, V]]]:
        return type(self), (self._data,)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        key, value = get_args(source) or (Any, Any)
        schema = core_schema.dict_schema(
            handler.generate_schema(key), handler.generate_schema(value)
        )

        return core_schema.no_info_after_validator_function(
            cls,
            schema,
            serialization=core_schema.wrap_serializer_function_ser_schema(
                lambda data, serializer: serializer(dict(data)), schema=schema
            ),
        )


# pydantic serializes values of unknown type, such as cases in fields typed `Any`, with this serializer
frozendict.__pydantic_serializer__ = SchemaSerializer(
    core_schema.is_instance_schema(
        frozendict,
        serialization=core_schema.plain_serializer_function_ser_schema(
            dict, info_arg=False, when_used="json"
        ),
    )
)


def singleton[K, T](x: Mapping[K, T] | Collection[T]) -> T:
    """
    Return the only element of the input, or raise an error if there are multiple elements.

    Args:
        x: The input collection or mapping.

    Returns:
        The only element of the input.

    Examples:
        >>> singleton([1])
        1
        >>> singleton({1: "a"})
        'a'

    Raises:
        ValueError: If the input has more than one element.
        TypeError: If the input is not a collection or mapping.
    """
    if len(x) != 1:
        raise ValueError(f"Expected exactly one element, but got {len(x)}")

    if isinstance(x, Mapping):
        return next(iter(x.values()))

    if isinstance(x, Collection):
        return next(iter(x))

    raise TypeError(f"Expected a Mapping or Collection, but got {type(x)}")


def dist2sim(distance: float) -> float:
    """Convert a distance to a similarity.

    Args:
        distance: The distance to convert

    Examples:
        >>> dist2sim(1.0)
        0.5
    """
    return 1 / (1 + distance)


@dataclass(slots=True)
class _Lazy[**P, R]:
    """Builds a function on its first call, see `lazy`.

    It is the one mutable helper of cbrkit, since it memoizes the built function,
    such as a model whose weights are loaded on the first call.
    A lock builds it once when several threads call it first.
    """

    factory: Callable[[], Callable[P, R]]
    func: Callable[P, R] | None = field(default=None, repr=False)
    lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    @property
    def function(self) -> Callable[P, R]:
        """The built function."""
        if self.func is None:
            with self.lock:
                if self.func is None:
                    self.func = self.factory()

        return self.func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.function(*args, **kwargs)

    def __getstate__(self) -> Callable[[], Callable[P, R]]:
        # worker processes build their own function instead of unpickling it
        return self.factory

    def __setstate__(self, factory: Callable[[], Callable[P, R]]) -> None:
        self.factory = factory
        self.func = None
        self.lock = Lock()


def lazy[**P, R](factory: Callable[[], Callable[P, R]]) -> Callable[P, R]:
    """Builds a function on its first call, for instance a model that takes long to load.

    Pickling keeps only the factory,
    so every worker process of a process or interpreter pool builds its own function.
    The built function decides whether the result is a batch function.

    Args:
        factory: Builds the function without arguments.

    Examples:
        >>> from cbrkit.sim.generic import Equality
        >>> sim = lazy(Equality)
        >>> sim("a", "a"), is_batched(sim)
        (1.0, False)
    """
    return _Lazy(factory)


_loop: ContextVar[asyncio.AbstractEventLoop | None] = ContextVar(
    "cbrkit_loop", default=None
)
"""The event loop of `to_thread`, which its threads inherit with their context."""


async def to_thread[**P, R](
    func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    """Runs a synchronous function in a thread of its own, from which `wait` awaits on the running event loop.

    `cbrkit.cycle.run_async` runs the synchronous functions of the cycle with it,
    so their measures may embed or generate through the clients of this loop.
    The thread is not one of the default executor, since a function blocked in `wait`
    would hold that thread while the awaited work, such as an embedder, needs one of the same pool,
    so enough concurrent calls would starve the pool for good.

    Examples:
        >>> asyncio.run(to_thread(sum, [1, 2]))
        3
    """
    loop = asyncio.get_running_loop()
    context = copy_context()
    _ = context.run(_loop.set, loop)
    executor = ThreadPoolExecutor(1, thread_name_prefix="cbrkit")

    try:
        return await loop.run_in_executor(
            executor, partial(context.run, func, *args, **kwargs)
        )
    finally:
        executor.shutdown(wait=False)


def wait[T](coroutine: Coroutine[Any, Any, T], /) -> T:
    """Awaits a coroutine from synchronous code, such as a measure that embeds or generates.

    In a worker thread of `to_thread`, the coroutine runs on the event loop of its caller,
    otherwise on a new event loop.

    Raises:
        RuntimeError: If an event loop runs in the current thread, which the call would block.

    Examples:
        >>> async def double(value):
        ...     return 2 * value
        >>> wait(double(2))
        4
        >>> async def run():
        ...     return await to_thread(wait, double(3))
        >>> asyncio.run(run())
        6
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = _loop.get()

        if loop is None:
            return asyncio.run(coroutine)

        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

    coroutine.close()

    raise RuntimeError(
        "An event loop runs in this thread, run the function with cbrkit.helpers.to_thread"
    )


def _bounded[T, R](
    run: Callable[[T], Awaitable[R]], limit: int
) -> Callable[[T], Coroutine[Any, Any, R]]:
    """Wraps a coroutine function so that at most `limit` of its calls are in flight at once.

    Raises:
        ValueError: If `limit` is less than one, which would never start a call.
    """
    if limit < 1:
        raise ValueError(f"The concurrency limit must be at least 1, not {limit}")

    semaphore = asyncio.Semaphore(limit)

    async def guarded(item: T) -> R:
        async with semaphore:
            return await run(item)

    return guarded


async def _unwind(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Cancels the unfinished tasks and awaits all of them, even through the cancellation of the caller."""
    for task in tasks:
        _ = task.cancel()

    _ = await _shield_to_completion(asyncio.gather(*tasks, return_exceptions=True))


async def bounded_gather[T, R](
    items: Iterable[T],
    run: Callable[[T], Awaitable[R]],
    /,
    *,
    limit: int,
) -> list[R]:
    """Returns `run(item)` for every item in input order, at most `limit` in flight.

    The first exception cancels the remaining calls and propagates,
    so `run` should return its expected failures rather than raise them.

    Examples:
        >>> async def double(value):
        ...     return 2 * value
        >>> asyncio.run(bounded_gather([1, 2, 3], double, limit=2))
        [2, 4, 6]
        >>> async def fail(value):
        ...     raise ValueError(value)
        >>> asyncio.run(bounded_gather([1, 2], fail, limit=2))
        Traceback (most recent call last):
        ...
        ValueError: 1
    """
    guarded = _bounded(run, limit)
    tasks = [asyncio.ensure_future(guarded(item)) for item in items]

    try:
        return await asyncio.gather(*tasks)
    finally:
        await _unwind(tasks)


async def _shield_to_completion[T](work: Awaitable[T], /) -> T:
    """Runs an awaitable to completion, surviving the cancellation of the calling task.

    `asyncio.shield` protects the wrapped coroutine, but its awaiter still raises `CancelledError` at once,
    leaving the work running detached.
    For a critical section that mutates shared state under a lock, that is a bug,
    since the lock releases before the work finishes and whatever runs next races the orphaned task.
    This instead keeps awaiting the shielded task through the cancellation of the caller,
    and re-raises a deferred cancellation afterwards, which takes precedence even over an error of the work.
    Such an error is logged and chained as the context of the cancellation, so it is never lost.

    Examples:
        >>> asyncio.run(_shield_to_completion(asyncio.sleep(0, result="done")))
        'done'
    """
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None

    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.done():
                raise

            # Our own cancellation while the work is still in flight, so wait for it.
            cancelled = exc
        except BaseException:
            # The work raised, but a deferred cancellation of the caller takes precedence.
            if cancelled is not None:
                logger.exception(
                    "A shielded task failed after its caller was cancelled"
                )

                raise cancelled

            raise
        else:
            if cancelled is not None:
                raise cancelled

            return result


@dataclass(slots=True, frozen=True)
class _Batchify[V, R](Batched):
    """Calls a pair function on every pair of a batch."""

    __wrapped__: PairFunc[V, R]

    def __call__(self, batches: Sequence[tuple[V, V]], /) -> Sequence[R]:
        return [self.__wrapped__(case, query) for case, query in batches]


@dataclass(slots=True, frozen=True)
class _Columns[V](Batched):
    """Scores a batch of pairs as two columns."""

    __wrapped__: ColumnSimFunc[V]

    def __call__(self, batches: Sequence[tuple[V, V]], /) -> list[float]:
        cases = [case for case, _ in batches]
        queries = [query for _, query in batches]

        return self.__wrapped__.columns(cases, queries).tolist()


@dataclass(slots=True, frozen=True)
class _Unbatchify[V, R]:
    """Calls a batch function on a batch with a single pair."""

    __wrapped__: BatchPairFunc[V, R]

    def __call__(self, case: V, query: V, /) -> R:
        return self.__wrapped__([(case, query)])[0]


@dataclass(slots=True, frozen=True)
class _Batched[V, R](Batched):
    """A function marked as a batch function, see `batched`."""

    __wrapped__: BatchPairFunc[V, R]

    def __call__(self, batches: Sequence[tuple[V, V]], /) -> Sequence[R]:
        return self.__wrapped__(batches)


def batched[V, R](func: BatchPairFunc[V, R]) -> BatchPairFunc[V, R]:
    """Marks a function as a batch function that takes a sequence of pairs.

    Classes deriving from a batch protocol such as `cbrkit.typing.BatchSimFunc` need no marking.

    Examples:
        >>> @batched
        ... def equality(batches):
        ...     return [1.0 if case == query else 0.0 for case, query in batches]
        >>> unbatchify(equality)("a", "a")
        1.0
    """
    return _Batched(func)


def is_batched[V, R](
    func: PairFunc[V, R] | BatchPairFunc[V, R],
) -> TypeIs[BatchPairFunc[V, R]]:
    """Whether a function is a batch function, see `batched`.

    A function of `lazy` is one if the function that it builds is one.
    """
    if isinstance(func, _Lazy):
        return is_batched(func.function)

    return Batched in type(func).__mro__


def batchify[V, R](func: PairFunc[V, R] | BatchPairFunc[V, R]) -> BatchPairFunc[V, R]:
    """Turns a pair function such as a similarity into a batch function.

    Batch functions are returned as they are,
    and a `cbrkit.typing.ColumnSimFunc` scores the batch as two columns.

    Examples:
        >>> batchify(lambda case, query: case == query)([(1, 1), (1, 2)])
        [True, False]
        >>> from cbrkit.sim.numbers import Linear
        >>> batchify(Linear(max_distance=10))([(1, 6), (2, 2)])
        [0.5, 1.0]
    """
    if is_batched(func):
        return func

    if isinstance(func, ColumnSimFunc):
        return cast(BatchPairFunc[V, R], _Columns(func))

    return _Batchify(func)


def unbatchify[V, R](func: PairFunc[V, R] | BatchPairFunc[V, R]) -> PairFunc[V, R]:
    """Turns a batch function into a pair function such as a similarity.

    Pair functions are returned as they are.

    Examples:
        >>> from cbrkit.sim.generic import Equality
        >>> unbatchify(batchify(Equality()))("a", "b")
        0.0
    """
    if is_batched(func):
        return _Unbatchify(func)

    return func


def broadcast_getter(obj: Iterable[Any], key: str) -> tuple[Any, ...]:
    """Retrieve a value from every case of a group, returning them as a tuple.

    Use as the `value_getter` of `cbrkit.sim.AttributeValue` for grouped cases.
    Because the result is a tuple rather than a value, it is never `None`, so the
    `default` of `AttributeValue` does not apply to individual group members.

    Examples:
        >>> broadcast_getter(({"price": 10}, {"price": 20}), "price")
        (10, 20)
    """
    return tuple([getitem_or_getattr(case, key) for case in obj])


def replace_attributes[V](obj: V, updates: Mapping[str, Any]) -> V:
    """Return a copy of an object with the given attributes replaced.

    Dispatches on the case representation so that immutable cases are supported
    as well: pydantic models, dataclasses (including frozen ones), and mappings
    (including `frozendict`).

    Raises:
        TypeError: If the object is none of these.

    Examples:
        >>> replace_attributes({"price": 10, "year": 2000}, {"price": 20})
        {'price': 20, 'year': 2000}
        >>> from dataclasses import dataclass
        >>> @dataclass(slots=True, frozen=True)
        ... class Car:
        ...     price: int
        >>> replace_attributes(Car(10), {"price": 20})
        Car(price=20)
    """
    if isinstance(obj, BaseModel):
        return cast(V, obj.model_copy(update=dict(updates)))

    if is_dataclass(obj) and not isinstance(obj, type):
        return cast(V, replace(obj, **updates))

    if isinstance(obj, MutableMapping):
        adapted = copy(obj)
        adapted.update(updates)

        return cast(V, adapted)

    if isinstance(obj, Mapping):
        mapping_type = cast(Callable[[Mapping[str, Any]], V], type(obj))

        return mapping_type({**obj, **updates})

    raise TypeError(f"Cannot replace the attributes of {type(obj).__name__}")
