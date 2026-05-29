import asyncio
import hashlib
import inspect
import logging
import math
import os
import warnings
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from io import BytesIO
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Coroutine, Literal, TypeIs, cast, override

from pydantic import BaseModel, Field, create_model

from .typing import (
    AdaptationFunc,
    AnyConversionFunc,
    AnyNamedFunc,
    AnyPositionalFunc,
    AnySimFunc,
    BatchAdaptationFunc,
    BatchConversionFunc,
    BatchNamedFunc,
    BatchPositionalFunc,
    BatchSimFunc,
    ConversionFunc,
    Factory,
    Float,
    HasMetadata,
    JsonEntry,
    MaybeFactories,
    MaybeFactory,
    MaybeSequence,
    NamedFunc,
    PositionalFunc,
    SimFunc,
    SimMap,
    SimpleAdaptationFunc,
    SimSeq,
    StructuredValue,
    WrappedObject,
)

__all__ = [
    "BATCH_LOGGING_LEVEL",
    "batchify_adaptation",
    "batchify_conversion",
    "batchify_named",
    "batchify_positional",
    "batchify_sim",
    "callable2model",
    "chain_map_chunks",
    "chunkify",
    "chunkify_overlap",
    "dispatch_batches",
    "dispatch_batches_async",
    "dist2sim",
    "get_hash",
    "get_logger",
    "get_metadata",
    "get_name",
    "get_optional_name",
    "get_value",
    "getitem_or_getattr",
    "setitem_or_setattr",
    "identity",
    "is_factory",
    "load_callable",
    "load_callables",
    "load_callables_map",
    "load_object",
    "log_batch",
    "mp_count",
    "mp_map",
    "mp_pool",
    "mp_starmap",
    "normalize",
    "normalize_and_scale",
    "optional_dependencies",
    "produce_factories",
    "produce_factory",
    "produce_sequence",
    "reverse_batch_positional",
    "reverse_positional",
    "round_int",
    "round_nearest",
    "run_coroutine",
    "run_threaded",
    "scale",
    "sim_map2ranking",
    "sim_seq2ranking",
    "singleton",
    "total_params",
    "unbatchify_adaptation",
    "unbatchify_conversion",
    "unbatchify_named",
    "unbatchify_positional",
    "unbatchify_sim",
    "unpack_float",
    "unpack_floats",
    "unpack_value",
    "unpack_values",
    "use_mp",
    "wrap_factory",
]


def run_coroutine[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code without returning a Task.

    When an event loop is already running in the current thread, the coroutine
    runs on a fresh loop in a background thread and this call blocks for its
    result. Otherwise the coroutine runs via `asyncio.run`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return cast(T, executor.submit(asyncio.run, coro).result())


async def run_threaded[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a blocking sync function in the default thread pool.

    Centralizes the sync-from-async bridge so callers do not have to
    reach for :func:`asyncio.to_thread` directly.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


@contextmanager
def optional_dependencies(
    error_handling: Literal["ignore", "warn", "raise"] = "ignore",
    extras_name: str | None = None,
) -> Generator[None, Any, None]:
    """Context manager that catches ImportError for optional dependencies."""
    try:
        yield None
    except (ImportError, ModuleNotFoundError) as e:
        match error_handling:
            case "raise":
                msg = str(e)
                if extras_name is not None:
                    msg += f". Please install `cbrkit[{extras_name}]`"
                raise ImportError(msg) from e
            case "warn":
                msg = f"Missing optional dependency: `{e.name}`"
                if extras_name is not None:
                    msg += f". Please install `cbrkit[{extras_name}]`"
                warnings.warn(msg, ImportWarning, stacklevel=2)
            case "ignore":
                pass


def get_name(obj: Any) -> str:
    """Return a human-readable name for the given object."""
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if hasattr(obj, "__name__"):
        return obj.__name__

    return type(obj).__name__


def get_optional_name(obj: Any | None) -> str | None:
    """Return a human-readable name for the given object, or None if the object is None."""
    if obj is None:
        return None

    return get_name(obj)


def get_metadata(obj: Any) -> JsonEntry:
    """Recursively extract metadata from an object as a JSON-serializable structure."""
    if isinstance(obj, WrappedObject):
        return get_metadata(obj.__wrapped__)

    if isinstance(obj, HasMetadata):
        return {
            "name": get_optional_name(obj),
            "doc": inspect.getdoc(obj),
            "metadata": obj.metadata,
        }

    if is_dataclass(obj):
        return {
            "name": get_optional_name(obj),
            "doc": inspect.getdoc(obj),
            "metadata": {
                field.name: get_metadata(getattr(obj, field.name))
                for field in fields(obj)
                if field.repr
            },
        }

    if callable(obj):
        return {
            "name": get_optional_name(obj),
            "doc": inspect.getdoc(obj),
        }

    if isinstance(obj, dict):
        return {str(key): get_metadata(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [get_metadata(value) for value in obj]

    if isinstance(obj, str | int | float | bool):
        return obj

    return None


def singleton[T](x: Mapping[Any, T] | Collection[T]) -> T:
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
        return cast(T, next(iter(x.values())))
    elif isinstance(x, Collection):
        return next(iter(x))

    raise TypeError(f"Expected a Mapping or Collection, but got {type(x)}")


def chunkify[V](val: Sequence[V], size: int) -> Iterator[Sequence[V]]:
    """Yield chunks with a fixed size.

    Examples:
        >>> list(chunkify([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """

    if size < 1:
        raise ValueError("Chunk size must be greater than 0")

    for i in range(0, len(val), size):
        yield val[i : i + size]


def chunkify_overlap[V](
    val: Sequence[V],
    size: int,
    overlap: int,
    direction: Literal["left", "right", "both"] = "both",
) -> Iterator[Sequence[V]]:
    """Yield fixed-size chunks with overlapping elements from adjacent chunks."""
    chunks = list(chunkify(val, size))
    take_left = overlap > 0 and direction in ("left", "both")
    take_right = overlap > 0 and direction in ("right", "both")

    for i, chunk in enumerate(chunks):
        left = chunks[i - 1][-overlap:] if take_left and i > 0 else []
        right = chunks[i + 1][:overlap] if take_right and i < len(chunks) - 1 else []

        yield [*left, *chunk, *right]


def chain_map_chunks[U, V](
    batches: Sequence[Sequence[U]],
    func: AnyConversionFunc[U, V],
) -> Sequence[Sequence[V]]:
    """Apply a conversion function to flattened chunks and reshape back into batches."""
    batched_func = batchify_conversion(func)
    batch2chunk_indexes: list[list[int]] = []
    flat_batches: list[U] = []

    for batch in batches:
        last_idx = len(batch2chunk_indexes)
        batch2chunk_indexes.append(list(range(last_idx, last_idx + len(batch))))
        flat_batches.extend(batch)

    results = batched_func(flat_batches)

    return [
        [results[idx] for idx in chunk_indexes] for chunk_indexes in batch2chunk_indexes
    ]


def dispatch_batches[K, V, R](
    batches: Sequence[tuple[Mapping[K, V], V]],
    call_queries: Callable[[Sequence[V], Mapping[K, V]], Sequence[R]],
) -> Sequence[R]:
    """Dispatches batches, optimizing when all casebases are identical.

    When every batch entry shares the same casebase (by identity), all queries
    are forwarded in a single call.  Otherwise each batch is processed
    individually.

    Args:
        batches: Sequence of (casebase, query) pairs.
        call_queries: Callable that takes a sequence of queries and a single
            casebase, returning a sequence of results.

    Returns:
        A flat sequence of results, one per input batch entry.
    """
    if not batches:
        return []

    first_casebase = batches[0][0]

    if all(
        casebase is first_casebase or (not casebase and not first_casebase)
        for casebase, _ in batches
    ):
        return call_queries([query for _, query in batches], first_casebase)

    return [call_queries([query], casebase)[0] for casebase, query in batches]


async def dispatch_batches_async[K, V, R](
    batches: Sequence[tuple[Mapping[K, V], V]],
    call_queries: Callable[[Sequence[V], Mapping[K, V]], Awaitable[Sequence[R]]],
) -> Sequence[R]:
    """Async mirror of :func:`dispatch_batches`.

    When every batch entry shares the same casebase (by identity, or all
    empty), forwards all queries in a single ``call_queries`` call so the
    query-side work (e.g. embedding) batches; otherwise dispatches each
    batch individually.
    """
    if not batches:
        return []

    first_casebase = batches[0][0]

    if all(
        casebase is first_casebase or (not casebase and not first_casebase)
        for casebase, _ in batches
    ):
        return await call_queries([query for _, query in batches], first_casebase)

    return [(await call_queries([query], casebase))[0] for casebase, query in batches]


def dist2sim(distance: float) -> float:
    """Convert a distance to a similarity.

    Args:
        distance: The distance to convert

    Examples:
        >>> dist2sim(1.0)
        0.5
    """
    return 1 / (1 + distance)


BATCH_LOGGING_LEVEL: int = logging.DEBUG


def log_batch(
    logger: logging.Logger | None,
    i: int,
    total: int,
):
    """Log the progress of batch processing."""
    if logger is not None and total > 1:
        logger.log(BATCH_LOGGING_LEVEL, f"Processing batch {i}/{total}")


def total_params(func: Callable[..., Any]) -> int:
    """Return the total number of parameters in a callable's signature."""
    return len(inspect.signature(func).parameters)


def _resolve_callable[F: Callable[..., Any]](
    wrapped: MaybeFactory[F], parameters: int
) -> tuple[F, int]:
    """Resolve a possibly-factory callable, returning (callable, parameter count).

    A zero-parameter wrapped value is treated as a factory and invoked to obtain
    the real callable.
    """
    if parameters == 0:
        func = cast(Factory[F], wrapped)()
        return func, total_params(func)

    return cast(F, wrapped), parameters


def produce_sequence[T](obj: MaybeSequence[T]) -> list[T]:
    """Wrap a single value or sequence into a list."""
    if isinstance(obj, str):
        return cast(list[T], [obj])

    if isinstance(obj, Sequence):
        return cast(list[T], list(obj))

    return [obj]


def is_factory[T](obj: MaybeFactory[T]) -> TypeIs[Factory[T]]:
    """Check whether the given object is a zero-argument factory callable."""
    return callable(obj) and total_params(obj) == 0


def produce_factory[T](obj: T | Factory[T]) -> T:
    """Resolve a factory by calling it, or return the value as-is."""
    if is_factory(obj):
        return cast(T, obj())

    return cast(T, obj)


def produce_factories[T](obj: MaybeFactories[T]) -> list[T]:
    """Resolve one or more factories into a list of values."""
    if isinstance(obj, Sequence):
        return cast(list[T], [produce_factory(item) for item in obj])

    return cast(list[T], [produce_factory(obj)])


@dataclass(slots=True, frozen=True)
class wrap_factory[**P, T]:
    """Wraps a factory or callable, resolving the factory on each call."""

    func: MaybeFactory[Callable[P, T]]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        func = cast(Callable[P, T], produce_factory(self.func))

        return func(*args, **kwargs)


@dataclass(slots=True, init=False)
class batchify_positional[T](
    WrappedObject[MaybeFactory[AnyPositionalFunc[T]]], BatchPositionalFunc[T]
):
    """Normalizes a positional function to batch mode."""

    __wrapped__: MaybeFactory[AnyPositionalFunc[T]]
    parameters: int
    logger: logging.Logger | None

    def __init__(self, func: MaybeFactory[AnyPositionalFunc[T]]):
        self.__wrapped__ = func
        self.parameters = total_params(func)
        logger = get_logger(func)
        self.logger = logger if logger.isEnabledFor(BATCH_LOGGING_LEVEL) else None

    @override
    def __call__(self, batches: Sequence[Any]) -> Sequence[T]:
        func, parameters = _resolve_callable(self.__wrapped__, self.parameters)

        if parameters == 1:
            return cast(BatchPositionalFunc[T], func)(batches)

        positional = cast(PositionalFunc[T], func)
        values: list[T] = []

        for i, batch in enumerate(batches, start=1):
            log_batch(self.logger, i, len(batches))
            values.append(positional(*batch))

        return values


@dataclass(slots=True, init=False)
class unbatchify_positional[T](
    WrappedObject[MaybeFactory[AnyPositionalFunc[T]]], PositionalFunc[T]
):
    """Normalizes a batch positional function to single-item mode."""

    __wrapped__: MaybeFactory[AnyPositionalFunc[T]]
    parameters: int

    def __init__(self, func: MaybeFactory[AnyPositionalFunc[T]]):
        self.__wrapped__ = func
        self.parameters = total_params(func)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        func, parameters = _resolve_callable(self.__wrapped__, self.parameters)

        if parameters == 1:
            return cast(BatchPositionalFunc[T], func)([args])[0]

        return cast(PositionalFunc[T], func)(*args)


@dataclass(slots=True, init=False)
class batchify_named[T](
    WrappedObject[MaybeFactory[AnyNamedFunc[T]]], BatchNamedFunc[T]
):
    """Normalizes a named function to batch mode."""

    __wrapped__: MaybeFactory[AnyNamedFunc[T]]
    parameters: int
    logger: logging.Logger | None

    def __init__(self, func: MaybeFactory[AnyNamedFunc[T]]):
        self.__wrapped__ = func
        self.parameters = total_params(func)
        logger = get_logger(func)
        self.logger = logger if logger.isEnabledFor(BATCH_LOGGING_LEVEL) else None

    @override
    def __call__(self, batches: Sequence[Any]) -> Sequence[T]:
        func, parameters = _resolve_callable(self.__wrapped__, self.parameters)

        if parameters == 1:
            return cast(BatchNamedFunc[T], func)(batches)

        named = cast(NamedFunc[T], func)
        values: list[T] = []

        for i, batch in enumerate(batches, start=1):
            log_batch(self.logger, i, len(batches))
            values.append(named(**batch))

        return values


@dataclass(slots=True, init=False)
class unbatchify_named[T](WrappedObject[MaybeFactory[AnyNamedFunc[T]]], NamedFunc[T]):
    """Normalizes a batch named function to single-item mode."""

    __wrapped__: MaybeFactory[AnyNamedFunc[T]]
    parameters: int

    def __init__(self, func: MaybeFactory[AnyNamedFunc[T]]):
        self.__wrapped__ = func
        self.parameters = total_params(func)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        func, parameters = _resolve_callable(self.__wrapped__, self.parameters)

        if parameters == 1:
            return cast(BatchNamedFunc[T], func)([kwargs])[0]

        return cast(NamedFunc[T], func)(**kwargs)


def batchify_sim[V, S: Float](
    func: MaybeFactory[AnySimFunc[V, S]],
) -> BatchSimFunc[V, S]:
    """Normalize a similarity function to batch mode."""
    return batchify_positional(func)


def unbatchify_sim[V, S: Float](func: MaybeFactory[AnySimFunc[V, S]]) -> SimFunc[V, S]:
    """Normalize a batch similarity function to single-item mode."""
    return unbatchify_positional(func)


def batchify_conversion[P, R](
    func: MaybeFactory[AnyConversionFunc[P, R]],
) -> BatchConversionFunc[P, R]:
    """Normalize a conversion function to batch mode."""
    return batchify_positional(func)


def unbatchify_conversion[P, R](
    func: MaybeFactory[AnyConversionFunc[P, R]],
) -> ConversionFunc[P, R]:
    """Normalize a batch conversion function to single-item mode."""
    return unbatchify_positional(func)


def batchify_adaptation[V](
    func: MaybeFactory[SimpleAdaptationFunc[V]],
) -> BatchAdaptationFunc[V]:
    """Normalizes an adaptation function to batch mode.

    Args:
        func: An adaptation function or batch adaptation function.

    Returns:
        A batch adaptation function.
    """
    return batchify_positional(func)


def unbatchify_adaptation[V](
    func: MaybeFactory[SimpleAdaptationFunc[V]],
) -> AdaptationFunc[V]:
    """Normalizes an adaptation function to single-item mode.

    Args:
        func: An adaptation function or batch adaptation function.

    Returns:
        An adaptation function.
    """
    return unbatchify_positional(func)


@dataclass(slots=True)
class reverse_positional[T](WrappedObject[PositionalFunc[T]], PositionalFunc[T]):
    """Reverses the order of positional arguments before calling the wrapped function."""

    __wrapped__: PositionalFunc[T]

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self.__wrapped__(*reversed(args))


@dataclass(slots=True)
class reverse_batch_positional[T](
    WrappedObject[BatchPositionalFunc[T]], BatchPositionalFunc[T]
):
    """Reverses arguments in each batch entry before calling the wrapped function."""

    __wrapped__: BatchPositionalFunc[T]

    @override
    def __call__(self, batches: Sequence[Any]) -> Sequence[T]:
        return self.__wrapped__([[*reversed(batch)] for batch in batches])


def get_value[T](arg: StructuredValue[T]) -> T:
    """Extract the inner value from a StructuredValue."""
    return arg.value


def unpack_value[T](arg: T | StructuredValue[T]) -> T:
    """Unwrap a StructuredValue to its inner value, or return the argument unchanged."""
    if isinstance(arg, StructuredValue):
        return arg.value

    return arg


def unpack_values[T](args: Iterable[T | StructuredValue[T]]) -> list[T]:
    """Unwrap each element in an iterable using unpack_value."""
    return [unpack_value(arg) for arg in args]


unpack_float: Callable[[Float], float] = unpack_value
unpack_floats: Callable[[Iterable[Float]], list[float]] = unpack_values


def sim_map2ranking[K, S: Float](similarities: SimMap[K, S]) -> list[K]:
    """Sort similarity map keys by descending similarity score."""
    return sorted(
        similarities, key=lambda i: unpack_float(similarities[i]), reverse=True
    )


def sim_seq2ranking[S: Float](similarities: SimSeq[S]) -> list[int]:
    """Sort similarity sequence indices by descending similarity score."""
    return sorted(
        range(len(similarities)),
        key=lambda i: unpack_float(similarities[i]),
        reverse=True,
    )


def getitem_or_getattr(obj: Any, key: Any) -> Any:
    """Retrieve a value by key using item access or attribute access as fallback."""
    if hasattr(obj, "__getitem__"):
        return obj[key]

    return getattr(obj, key)


def setitem_or_setattr(obj: Any, key: Any, value: Any) -> None:
    """Set a value by key using item assignment or attribute assignment as fallback."""
    if hasattr(obj, "__setitem__"):
        obj[key] = value
    else:
        setattr(obj, key, value)


def round_nearest(value: float) -> int:
    """Round a float to the nearest integer, rounding up on a tie."""
    return math.floor(value + 0.5)


def round_int(value: float, mode: Literal["floor", "ceil", "nearest"] = "nearest") -> int:
    """Round a float to an integer using the specified rounding mode."""
    match mode:
        case "floor":
            return math.floor(value)
        case "ceil":
            return math.ceil(value)
        case "nearest":
            return round_nearest(value)
        case _:
            raise ValueError(f"Invalid rounding mode: {mode!r}")


def scale(value: float, lower: float, upper: float) -> float:
    """Scale a value from [0, 1] to [lower, upper]."""
    return value * (upper - lower) + lower


def normalize(value: float, value_min: float, value_max: float) -> float:
    """Normalize a value from [value_min, value_max] to [0, 1].

    Returns 0 when min and max coincide.
    """
    if value_max == value_min:
        return 0.0

    return (value - value_min) / (value_max - value_min)


def normalize_and_scale(
    value: float,
    value_min: float,
    value_max: float,
    target_min: float,
    target_max: float,
) -> float:
    """Normalize a value from [value_min, value_max] to [target_min, target_max]."""
    # First normalize to [0, 1]
    normalized = normalize(value, value_min, value_max)

    # Then scale to target range
    return scale(normalized, target_min, target_max)


def load_object(import_name: str) -> Any:
    """Import an object based on a string.

    Args:
        import_name: Can either be in in dotted notation (`module.submodule.object`)
            or with a colon as object delimiter (`module.submodule:object`).

    Returns:
        The imported object.
    """

    if ":" in import_name:
        module_name, obj_name = import_name.split(":", 1)
    elif "." in import_name:
        module_name, obj_name = import_name.rsplit(".", 1)
    else:
        raise ValueError(f"Failed to import {import_name!r}")

    module = import_module(module_name)

    return getattr(module, obj_name)


def load_callable(import_name: str) -> Callable[..., Any]:
    """Import a callable from a dotted or colon-separated import path."""
    return load_object(import_name)


def _iter_names(import_names: MaybeSequence[str]) -> list[str]:
    return [import_names] if isinstance(import_names, str) else list(import_names)


def load_callables(
    import_names: MaybeSequence[str],
) -> list[Callable[..., Any]]:
    """Import one or more callables from dotted or colon-separated import paths."""
    functions: list[Callable[..., Any]] = []

    for import_name in _iter_names(import_names):
        obj = load_object(import_name)

        if isinstance(obj, Sequence):
            assert all(callable(func) for func in obj)
            functions.extend(cast(Sequence[Callable[..., Any]], obj))
        elif callable(obj):
            functions.append(obj)

    return functions


def load_callables_map(
    import_names: MaybeSequence[str],
) -> dict[str, Callable[..., Any]]:
    """Import callables into a dict keyed by their import paths."""
    functions: dict[str, Callable[..., Any]] = {}

    for import_name in _iter_names(import_names):
        obj = load_object(import_name)

        if isinstance(obj, Mapping):
            assert all(callable(func) for func in obj.values())
            functions.update(obj)
        elif callable(obj):
            functions[import_name] = obj

    return functions


def identity[T](x: T) -> T:
    """Return the argument unchanged."""
    return x


def get_logger(obj: Any) -> logging.Logger:
    """Return a logger named after the given object's module and qualified name."""
    if isinstance(obj, str):
        return logging.getLogger(obj)

    if hasattr(obj, "__self__"):
        obj = obj.__self__

    if hasattr(obj, "__class__"):
        obj = obj.__class__

    name = obj.__module__

    if not name.endswith(obj.__qualname__):
        name += f".{obj.__qualname__}"

    return logging.getLogger(name)


def mp_count(pool_or_processes: Pool | int | bool) -> int:
    """Return the number of worker processes for the given pool or process specification."""
    if isinstance(pool_or_processes, bool):
        return os.cpu_count() or 1
    elif isinstance(pool_or_processes, int):
        return pool_or_processes
    elif isinstance(pool_or_processes, Pool):
        return getattr(pool_or_processes, "_processes")

    return os.cpu_count() or 1


def mp_pool(pool_or_processes: Pool | int | bool) -> Pool:
    """Return an existing Pool or create a new one from the given specification."""
    if isinstance(pool_or_processes, bool):
        return Pool()
    elif isinstance(pool_or_processes, int):
        return Pool(pool_or_processes)
    elif isinstance(pool_or_processes, Pool):
        return pool_or_processes

    raise TypeError(f"Invalid multiprocessing value: {pool_or_processes}")


def use_mp(pool_or_processes: Pool | int | bool) -> bool:
    """Determine whether multiprocessing should be used for the given specification."""
    if isinstance(pool_or_processes, bool):
        return pool_or_processes
    elif isinstance(pool_or_processes, int):
        return pool_or_processes > 1
    elif isinstance(pool_or_processes, Pool):
        return True

    return False


@dataclass(slots=True, frozen=True)
class mp_logging_wrapper[V]:
    """Wraps a function with batch progress logging for multiprocessing.

    When `star` is True, the batch is unpacked as positional arguments.
    """

    func: Callable[..., V]
    logger: logging.Logger | None
    star: bool = False

    def __call__(self, batch: Any, i: int, total: int) -> V:
        log_batch(self.logger, i, total)

        if self.star:
            return self.func(*batch)

        return self.func(batch)


def _mp_dispatch[V](
    wrapper: mp_logging_wrapper[V],
    batches: Sequence[Any],
    pool_or_processes: Pool | int | bool,
) -> list[V]:
    args = ((batch, i, len(batches)) for i, batch in enumerate(batches, start=1))

    if use_mp(pool_or_processes):
        with mp_pool(pool_or_processes) as p:
            return p.starmap(wrapper, args)

    return [wrapper(*a) for a in args]


def mp_map[U, V](
    func: Callable[[U], V],
    batches: Sequence[U],
    pool_or_processes: Pool | int | bool,
    logger: logging.Logger | None,
) -> list[V]:
    """Apply a function to each batch, optionally using multiprocessing."""
    if logger is None or not logger.isEnabledFor(BATCH_LOGGING_LEVEL):
        logger = None

    return _mp_dispatch(mp_logging_wrapper(func, logger, star=False), batches, pool_or_processes)


def mp_starmap[*Us, V](
    func: Callable[[*Us], V],
    batches: Sequence[tuple[*Us]],
    pool_or_processes: Pool | int | bool,
    logger: logging.Logger | None,
) -> list[V]:
    """Apply a multi-argument function to each batch, optionally using multiprocessing."""
    if logger is None or not logger.isEnabledFor(BATCH_LOGGING_LEVEL):
        logger = None

    return _mp_dispatch(mp_logging_wrapper(func, logger, star=True), batches, pool_or_processes)


def get_hash(file: Path | bytes | BytesIO) -> str:
    """Compute a SHA-256 hex digest for a file path, bytes, or BytesIO object."""
    if isinstance(file, Path):
        data = file.read_bytes()
    elif isinstance(file, BytesIO):
        data = file.getvalue()
    else:
        data = file

    return hashlib.sha256(data).hexdigest()


def callable2model(
    func: Callable[..., Any], with_default: bool = True
) -> type[BaseModel]:
    """Convert a callable's signature into a Pydantic BaseModel class."""
    sig = inspect.signature(func)
    model_fields: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        field_type = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
        field_config = (
            Field(default=param.default)
            if param.default is not inspect.Parameter.empty and with_default
            else Field(...)
        )
        model_fields[param.name] = (field_type, field_config)

    return create_model(get_name(func), **model_fields)
