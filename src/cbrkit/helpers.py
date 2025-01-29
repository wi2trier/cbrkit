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
    "event_loop",
    "optional_dependencies",
    "dist2sim",
    "batchify_positional",
    "unbatchify_positional",
    "batchify_named",
    "unbatchify_named",
    "get_value",
    "unpack_value",
    "unpack_values",
    "unpack_float",
    "unpack_floats",
    "singleton",
    "chunkify",
    "sim_map2ranking",
    "sim_seq2ranking",
    "load_object",
    "load_callables",
    "load_callables_map",
    "identity",
    "get_logger",
    "mp_count",
    "mp_pool",
    "use_mp",
    "mp_map",
    "mp_starmap",
    "getitem_or_getattr",
    "round",
    "round_nearest",
    "scale",
    "log_batch",
    "BATCH_LOGGING_LEVEL",
    "total_params",
    "get_hash",
    "wrap_factory",
    "produce_factory",
    "produce_factories",
    "is_factory",
    "produce_sequence",
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

    if isinstance(obj, type):
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


def chunkify[V](val: Sequence[V], k: int) -> Iterator[Sequence[V]]:
    """Yield chunks of size `k` from the input sequence.

    Examples:
        >>> list(chunkify([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """

    if k < 1:
        raise ValueError("Chunk size must be greater than 0")

    for i in range(0, len(val), k):
        yield val[i : i + k]


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
    def __call__(self, batches: Sequence[tuple[Any, ...]]) -> Sequence[T]:
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
    def __call__(self, *args: Any) -> T:
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
    def __call__(self, batches: Sequence[dict[str, Any]]) -> Sequence[T]:
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
    def __call__(self, **kwargs: Any) -> T:
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
    return batchify_positional(cast(MaybeFactory[AnyPositionalFunc[S]], func))


def unbatchify_sim[V, S: Float](func: MaybeFactory[AnySimFunc[V, S]]) -> SimFunc[V, S]:
    return cast(
        SimFunc[V, S],
        unbatchify_positional(cast(MaybeFactory[AnyPositionalFunc[S]], func)),
    )


def batchify_conversion[P, R](
    func: MaybeFactory[AnyConversionFunc[P, R]],
) -> BatchConversionFunc[P, R]:
    return cast(
        BatchConversionFunc[P, R],
        batchify_positional(cast(MaybeFactory[AnyPositionalFunc[R]], func)),
    )


def unbatchify_conversion[P, R](
    func: MaybeFactory[AnyConversionFunc[P, R]],
) -> ConversionFunc[P, R]:
    return cast(
        ConversionFunc[P, R],
        unbatchify_positional(cast(MaybeFactory[AnyPositionalFunc[R]], func)),
    )


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
    import_names: Collection[str] | str,
) -> dict[str, Callable]:
    if isinstance(import_names, str):
        import_names = [import_names]

    functions: dict[str, Callable] = {}

    for import_name in import_names:
        obj = load_object(import_name)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            functions.update(obj)

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


def callable2model(func: Callable[..., Any]) -> type[BaseModel]:
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
            cast(type[Any], param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else Any
        )
        field_config = (
            Field(...)
            if param.default is inspect.Parameter.empty
            else Field(default=param.default)
        )
        fields[param.name] = (field_type, field_config)

    return create_model(f"{func.__name__}Model", **fields)
