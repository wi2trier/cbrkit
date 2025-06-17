import asyncio
import hashlib
import inspect
import logging
import math
import os
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import import_module
from io import BytesIO
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast, override

from pydantic import BaseModel, Field, create_model

from .typing import (
    AnyConversionFunc,
    AnyNamedFunc,
    AnyPositionalFunc,
    AnySimFunc,
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
    SimSeq,
    StructuredValue,
    WrappedObject,
)

__all__ = [
    "BATCH_LOGGING_LEVEL",
    "batchify_named",
    "batchify_positional",
    "callable2model",
    "chain_map_chunks",
    "chunkify_overlap",
    "chunkify",
    "dist2sim",
    "event_loop",
    "get_hash",
    "get_logger",
    "get_metadata",
    "get_name",
    "get_optional_name",
    "get_value",
    "getitem_or_getattr",
    "identity",
    "is_factory",
    "load_callables_map",
    "load_callables",
    "load_object",
    "normalize",
    "normalize_and_scale",
    "log_batch",
    "mp_count",
    "mp_map",
    "mp_pool",
    "mp_starmap",
    "optional_dependencies",
    "produce_factories",
    "produce_factory",
    "produce_sequence",
    "reverse_batch_positional",
    "reverse_positional",
    "round_nearest",
    "round",
    "scale",
    "sim_map2ranking",
    "sim_seq2ranking",
    "singleton",
    "total_params",
    "unbatchify_named",
    "unbatchify_positional",
    "unpack_float",
    "unpack_floats",
    "unpack_value",
    "unpack_values",
    "use_mp",
    "wrap_factory",
]


@dataclass(slots=True)
class EventLoop:
    _instance: asyncio.AbstractEventLoop | None = None

    def get(self) -> asyncio.AbstractEventLoop:
        if self._instance is None:
            self._instance = asyncio.new_event_loop()
            asyncio.set_event_loop(self._instance)

        return self._instance

    def close(self) -> None:
        if self._instance is not None:
            tasks = asyncio.all_tasks(self._instance)
            for task in tasks:
                task.cancel()

            try:
                self._instance.run_until_complete(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
            except asyncio.CancelledError:
                pass

            finally:
                self._instance.close()
                self._instance = None


event_loop = EventLoop()


@contextmanager
def optional_dependencies(
    error_handling: Literal["ignore", "warn", "raise"] = "ignore",
    extras_name: str | None = None,
) -> Generator[None, Any, None]:
    try:
        yield None
    except (ImportError, ModuleNotFoundError) as e:
        match error_handling:
            case "raise":
                if extras_name is not None:
                    print(f"Please install `cbrkit[{extras_name}]`")

                raise e
            case "warn":
                print(f"Missing optional dependency: `{e.name}`")

                if extras_name is not None:
                    print(f"Please install `cbrkit[{extras_name}]`")
            case "ignore":
                pass


def get_name(obj: Any) -> str:
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if hasattr(obj, "__name__"):
        return obj.__name__

    return type(obj).__name__


def get_optional_name(obj: Any | None) -> str | None:
    if obj is None:
        return None

    return get_name(obj)


def get_metadata(obj: Any) -> JsonEntry:
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

    if isinstance(obj, Callable):
        return {
            "name": get_optional_name(obj),
            "doc": inspect.getdoc(obj),
        }

    if isinstance(obj, dict):
        return {key: get_metadata(value) for key, value in obj.items()}

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
        return next(iter(x.values()))
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
    chunks = list(chunkify(val, size))

    for i, chunk in enumerate(chunks):
        # Get the previous, current, and next chunk based on the overlap
        prev_chunk = chunks[i - 1] if i > 0 and overlap > 0 else []
        next_chunk = chunks[i + 1] if i < len(chunks) - 1 and overlap > 0 else []

        if direction == "left" or direction == "both":
            prev_chunk = prev_chunk[-overlap:]

        if direction == "right" or direction == "both":
            next_chunk = next_chunk[:overlap]

        yield [*prev_chunk, *chunk, *next_chunk]


def chain_map_chunks[U, V](
    batches: Sequence[Sequence[U]],
    func: AnyConversionFunc[U, V],
) -> Sequence[Sequence[V]]:
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


def dist2sim(distance: float) -> float:
    """Convert a distance to a similarity.

    Args:
        distance: The distance to convert

    Examples:
        >>> dist2sim(1.)
        0.5
    """
    return 1 / (1 + distance)


BATCH_LOGGING_LEVEL: int = logging.DEBUG


def log_batch(
    logger: logging.Logger | None,
    i: int,
    total: int,
):
    if logger is not None and total > 1:
        logger.log(BATCH_LOGGING_LEVEL, f"Processing batch {i}/{total}")


def total_params(func: Callable) -> int:
    return len(inspect.signature(func).parameters)


def produce_sequence[T](obj: MaybeSequence[T]) -> list[T]:
    if isinstance(obj, Sequence):
        return list(obj)

    return [obj]


def is_factory[T](obj: MaybeFactory[T]) -> TypeGuard[Factory[T]]:
    return callable(obj) and total_params(obj) == 0


def produce_factory[T](obj: MaybeFactory[T]) -> T:
    if is_factory(obj):
        return obj()

    return cast(T, obj)


def produce_factories[T](obj: MaybeFactories[T]) -> list[T]:
    if isinstance(obj, Sequence):
        return [produce_factory(item) for item in obj]

    return [produce_factory(obj)]


@dataclass(slots=True, frozen=True)
class wrap_factory[**P, T]:
    func: MaybeFactory[Callable[P, T]]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        func: Callable[P, T] = produce_factory(self.func)

        return func(*args, **kwargs)


@dataclass(slots=True, init=False)
class batchify_positional[T](
    WrappedObject[MaybeFactory[AnyPositionalFunc[T]]], BatchPositionalFunc[T]
):
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
        if self.parameters == 0:
            func = cast(Factory[AnyPositionalFunc[T]], self.__wrapped__)()
            parameters = total_params(func)
        else:
            func = cast(AnyPositionalFunc[T], self.__wrapped__)
            parameters = self.parameters

        if parameters != 1:
            func = cast(PositionalFunc[T], func)
            values: list[T] = []

            for i, batch in enumerate(batches, start=1):
                log_batch(self.logger, i, len(batches))
                values.append(func(*batch))

            return values

        func = cast(BatchPositionalFunc[T], func)
        return func(batches)


@dataclass(slots=True, init=False)
class unbatchify_positional[T](
    WrappedObject[MaybeFactory[AnyPositionalFunc[T]]], PositionalFunc[T]
):
    __wrapped__: MaybeFactory[AnyPositionalFunc[T]]
    parameters: int

    def __init__(self, func: MaybeFactory[AnyPositionalFunc[T]]):
        self.__wrapped__ = func
        self.parameters = total_params(func)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        if self.parameters == 0:
            func = cast(Factory[AnyPositionalFunc[T]], self.__wrapped__)()
            parameters = total_params(func)
        else:
            func = cast(AnyPositionalFunc[T], self.__wrapped__)
            parameters = self.parameters

        if parameters == 1:
            func = cast(BatchPositionalFunc[T], func)
            return func([args])[0]

        func = cast(PositionalFunc[T], func)
        return func(*args)


@dataclass(slots=True, init=False)
class batchify_named[T](
    WrappedObject[MaybeFactory[AnyNamedFunc[T]]], BatchNamedFunc[T]
):
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
        if self.parameters == 0:
            func = cast(Factory[AnyNamedFunc[T]], self.__wrapped__)()
            parameters = total_params(func)
        else:
            func = cast(AnyNamedFunc[T], self.__wrapped__)
            parameters = self.parameters

        if parameters != 1:
            func = cast(NamedFunc[T], func)
            values: list[T] = []

            for i, batch in enumerate(batches, start=1):
                log_batch(self.logger, i, len(batches))
                values.append(func(**batch))

            return values

        func = cast(BatchNamedFunc[T], func)
        return func(batches)


@dataclass(slots=True)
class unbatchify_named[T](WrappedObject[MaybeFactory[AnyNamedFunc[T]]], NamedFunc[T]):
    __wrapped__: MaybeFactory[AnyNamedFunc[T]]
    parameters: int = field(init=False)

    def __init__(self, func: MaybeFactory[AnyNamedFunc[T]]):
        self.__wrapped__ = func
        self.parameters = total_params(func)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        if self.parameters == 0:
            func = cast(Factory[AnyNamedFunc[T]], self.__wrapped__)()
            parameters = total_params(func)
        else:
            func = cast(AnyNamedFunc[T], self.__wrapped__)
            parameters = self.parameters

        if parameters == 1:
            func = cast(BatchNamedFunc[T], func)
            return func([kwargs])[0]

        func = cast(NamedFunc[T], func)
        return func(**kwargs)


def batchify_sim[V, S: Float](
    func: MaybeFactory[AnySimFunc[V, S]],
) -> BatchSimFunc[V, S]:
    return batchify_positional(func)


def unbatchify_sim[V, S: Float](func: MaybeFactory[AnySimFunc[V, S]]) -> SimFunc[V, S]:
    return unbatchify_positional(func)


def batchify_conversion[P, R](
    func: MaybeFactory[AnyConversionFunc[P, R]],
) -> BatchConversionFunc[P, R]:
    return batchify_positional(func)


def unbatchify_conversion[P, R](
    func: MaybeFactory[AnyConversionFunc[P, R]],
) -> ConversionFunc[P, R]:
    return unbatchify_positional(func)


@dataclass(slots=True)
class reverse_positional[T](WrappedObject[PositionalFunc[T]], PositionalFunc[T]):
    __wrapped__: PositionalFunc[T]

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self.__wrapped__(*reversed(args))


@dataclass(slots=True)
class reverse_batch_positional[T](
    WrappedObject[BatchPositionalFunc[T]], BatchPositionalFunc[T]
):
    __wrapped__: BatchPositionalFunc[T]

    @override
    def __call__(self, batches: Sequence[Any]) -> Sequence[T]:
        return self.__wrapped__([[*reversed(batch)] for batch in batches])


def get_value[T](arg: StructuredValue[T]) -> T:
    return arg.value


def unpack_value[T](arg: T | StructuredValue[T]) -> T:
    if isinstance(arg, StructuredValue):
        return arg.value

    return arg


def unpack_values[T](args: Iterable[T | StructuredValue[T]]) -> list[T]:
    return [unpack_value(arg) for arg in args]


unpack_float: Callable[[Float], float] = unpack_value
unpack_floats: Callable[[Iterable[Float]], list[float]] = unpack_values


def sim_map2ranking[K, S: Float](similarities: SimMap[K, S]) -> list[K]:
    return sorted(
        similarities, key=lambda i: unpack_float(similarities[i]), reverse=True
    )


def sim_seq2ranking[S: Float](similarities: SimSeq[S]) -> list[int]:
    return sorted(
        range(len(similarities)),
        key=lambda i: unpack_float(similarities[i]),
        reverse=True,
    )


def getitem_or_getattr(obj: Any, key: Any) -> Any:
    if hasattr(obj, "__getitem__"):
        return obj[key]

    return getattr(obj, key)


def round_nearest(value: float) -> int:
    x = math.floor(value)

    if (value - x) < 0.50:
        return x

    return math.ceil(value)


def round(value: float, mode: Literal["floor", "ceil", "nearest"] = "nearest") -> int:
    if mode == "floor":
        return math.floor(value)
    elif mode == "ceil":
        return math.ceil(value)
    elif mode == "nearest":
        return round_nearest(value)

    raise ValueError(f"Invalid rounding mode: {mode}")


def scale(value: float, lower: float, upper: float) -> float:
    """Scale a value from [0, 1] to [lower, upper]."""
    if lower == 0 and upper == 1:
        return value

    return value * (upper - lower) + lower


def normalize(value: float, value_min: float, value_max: float) -> float:
    """Normalize a value from [value_min, value_max] to [0, 1]."""
    if value_max == value_min:
        # Handle edge case where all values are identical
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


def load_callable(import_name: str) -> Callable:
    return load_object(import_name)


def load_callables(
    import_names: MaybeSequence[str],
) -> list[Callable]:
    functions: list[Callable] = []

    for import_name in produce_sequence(import_names):
        obj = load_object(import_name)

        if isinstance(obj, Sequence):
            assert all(isinstance(func, Callable) for func in functions)
            functions.extend(obj)
        elif isinstance(obj, Callable):
            functions.append(obj)

    return functions


def load_callables_map(
    import_names: MaybeSequence[str],
) -> dict[str, Callable]:
    functions: dict[str, Callable] = {}

    for import_name in produce_sequence(import_names):
        obj = load_object(import_name)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            functions.update(obj)
        elif isinstance(obj, Callable):
            functions[import_name] = obj

    return functions


def identity[T](x: T) -> T:
    return x


def get_logger(obj: Any) -> logging.Logger:
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
    if isinstance(pool_or_processes, bool):
        return os.cpu_count() or 1
    elif isinstance(pool_or_processes, int):
        return pool_or_processes
    elif isinstance(pool_or_processes, Pool):
        return pool_or_processes._processes  # type: ignore

    return os.cpu_count() or 1


def mp_pool(pool_or_processes: Pool | int | bool) -> Pool:
    if isinstance(pool_or_processes, bool):
        return Pool()
    elif isinstance(pool_or_processes, int):
        return Pool(pool_or_processes)
    elif isinstance(pool_or_processes, Pool):
        return pool_or_processes

    raise TypeError(f"Invalid multiprocessing value: {pool_or_processes}")


def use_mp(pool_or_processes: Pool | int | bool) -> bool:
    if isinstance(pool_or_processes, bool):
        return pool_or_processes
    elif isinstance(pool_or_processes, int):
        return pool_or_processes > 1
    elif isinstance(pool_or_processes, Pool):
        return True

    return False


@dataclass(slots=True, frozen=True)
class mp_logging_wrapper[U, V]:
    func: Callable[[U], V]
    logger: logging.Logger | None

    def __call__(
        self,
        batch: U,
        i: int,
        total: int,
    ) -> V:
        log_batch(self.logger, i, total)

        return self.func(batch)


@dataclass(slots=True, frozen=True)
class mp_logging_starwrapper[*Us, V]:
    func: Callable[[*Us], V]
    logger: logging.Logger | None

    def __call__(
        self,
        batch: tuple[*Us],
        i: int,
        total: int,
    ) -> V:
        log_batch(self.logger, i, total)

        return self.func(*batch)


def mp_map[U, V](
    func: Callable[[U], V],
    batches: Sequence[U],
    pool_or_processes: Pool | int | bool,
    logger: logging.Logger | None,
) -> list[V]:
    if logger is None or not logger.isEnabledFor(BATCH_LOGGING_LEVEL):
        logger = None

    wrapper = mp_logging_wrapper(func, logger)

    if use_mp(pool_or_processes):
        pool = mp_pool(pool_or_processes)

        with pool as p:
            return p.starmap(
                wrapper,
                ((batch, i, len(batches)) for i, batch in enumerate(batches, start=1)),
            )

    return [wrapper(batch, i, len(batches)) for i, batch in enumerate(batches, start=1)]


def mp_starmap[*Us, V](
    func: Callable[[*Us], V],
    batches: Sequence[tuple[*Us]],
    pool_or_processes: Pool | int | bool,
    logger: logging.Logger | None,
) -> list[V]:
    if logger is None or not logger.isEnabledFor(BATCH_LOGGING_LEVEL):
        logger = None

    wrapper = mp_logging_starwrapper(func, logger)

    if use_mp(pool_or_processes):
        pool = mp_pool(pool_or_processes)

        with pool as p:
            return p.starmap(
                wrapper,
                ((batch, i, len(batches)) for i, batch in enumerate(batches, start=1)),
            )

    return [wrapper(batch, i, len(batches)) for i, batch in enumerate(batches, start=1)]


def get_hash(file: Path | bytes | BytesIO) -> str:
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
    sig = inspect.signature(func)
    fields: dict[str, Any] = {}

    for param in sig.parameters.values():
        # Skip *args/**kwargs
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
        fields[param.name] = (field_type, field_config)

    return create_model(func.__name__, **fields)
