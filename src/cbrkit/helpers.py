import asyncio
import logging
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
from inspect import getdoc
from inspect import signature as inspect_signature
from multiprocessing.pool import Pool
from typing import Any, ClassVar, Literal, cast, override

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
    Float,
    HasMetadata,
    JsonEntry,
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


def get_name(obj: Any) -> str | None:
    if obj is None:
        return None

    elif isinstance(obj, type):
        return obj.__name__

    return type(obj).__name__


def get_metadata(obj: Any) -> JsonEntry:
    if isinstance(obj, WrappedObject):
        return get_metadata(obj.__wrapped__)

    if isinstance(obj, HasMetadata):
        return {
            "name": get_name(obj),
            "doc": getdoc(obj),
            "metadata": obj.metadata,
        }

    if is_dataclass(obj):
        return {
            "name": get_name(obj),
            "doc": getdoc(obj),
            "metadata": {
                field.name: get_metadata(getattr(obj, field.name))
                for field in fields(obj)
                if field.repr
            },
        }

    if isinstance(obj, Callable):
        return {
            "name": get_name(obj),
            "doc": getdoc(obj),
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


@dataclass(slots=True, init=False)
class batchify_positional[T](
    WrappedObject[AnyPositionalFunc[T]], BatchPositionalFunc[T]
):
    __wrapped__: AnyPositionalFunc[T]
    parameters: int
    logger: logging.Logger | None
    logger_level: ClassVar[int] = logging.DEBUG

    def __init__(self, func: AnyPositionalFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)
        logger = get_logger(self.__wrapped__)
        self.logger = logger if logger.isEnabledFor(self.logger_level) else None

    @override
    def __call__(self, batches: Sequence[tuple[Any, ...]]) -> Sequence[T]:
        if self.parameters != 1:
            func = cast(PositionalFunc[T], self.__wrapped__)
            values: list[T] = []

            for i, batch in enumerate(batches, start=1):
                if self.logger is not None:
                    self.logger.log(
                        self.logger_level, f"Processing batch {i}/{len(batches)}"
                    )

                values.append(func(*batch))

            return values

        func = cast(BatchPositionalFunc[T], self.__wrapped__)
        return func(batches)


@dataclass(slots=True, init=False)
class unbatchify_positional[T](WrappedObject[AnyPositionalFunc[T]], PositionalFunc[T]):
    __wrapped__: AnyPositionalFunc[T]
    parameters: int

    def __init__(self, func: AnyPositionalFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)

    @override
    def __call__(self, *args: Any) -> T:
        if self.parameters == 1:
            func = cast(BatchPositionalFunc[T], self.__wrapped__)
            return func([args])[0]

        func = cast(PositionalFunc[T], self.__wrapped__)
        return func(*args)


@dataclass(slots=True, init=False)
class batchify_named[T](WrappedObject[AnyNamedFunc[T]], BatchNamedFunc[T]):
    __wrapped__: AnyNamedFunc[T]
    parameters: int
    logger: logging.Logger | None
    logger_level: ClassVar[int] = logging.DEBUG

    def __init__(self, func: AnyNamedFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)
        logger = get_logger(self.__wrapped__)
        self.logger = logger if logger.isEnabledFor(self.logger_level) else None

    @override
    def __call__(self, batches: Sequence[dict[str, Any]]) -> Sequence[T]:
        if self.parameters != 1:
            func = cast(NamedFunc[T], self.__wrapped__)
            values: list[T] = []

            for i, batch in enumerate(batches, start=1):
                if self.logger is not None:
                    self.logger.log(
                        self.logger_level, f"Processing batch {i}/{len(batches)}"
                    )

                values.append(func(**batch))

            return values

        func = cast(BatchNamedFunc[T], self.__wrapped__)
        return func(batches)


@dataclass(slots=True)
class unbatchify_named[T](WrappedObject[AnyNamedFunc[T]], NamedFunc[T]):
    __wrapped__: AnyNamedFunc[T]
    parameters: int = field(init=False)

    def __init__(self, func: AnyNamedFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)

    @override
    def __call__(self, **kwargs: Any) -> T:
        if self.parameters == 1:
            func = cast(BatchNamedFunc[T], self.__wrapped__)
            return func([kwargs])[0]

        func = cast(NamedFunc[T], self.__wrapped__)
        return func(**kwargs)


def batchify_sim[V, S: Float](func: AnySimFunc[V, S]) -> BatchSimFunc[V, S]:
    return batchify_positional(cast(AnyPositionalFunc[S], func))


def unbatchify_sim[V, S: Float](func: AnySimFunc[V, S]) -> SimFunc[V, S]:
    return cast(SimFunc[V, S], unbatchify_positional(cast(AnyPositionalFunc[S], func)))


def batchify_conversion[P, R](
    func: AnyConversionFunc[P, R],
) -> BatchConversionFunc[P, R]:
    return cast(
        BatchConversionFunc[P, R],
        batchify_positional(cast(AnyPositionalFunc[R], func)),
    )


def unbatchify_conversion[P, R](
    func: AnyConversionFunc[P, R],
) -> ConversionFunc[P, R]:
    return cast(
        ConversionFunc[P, R],
        batchify_positional(cast(AnyPositionalFunc[R], func)),
    )


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
    import_names: Sequence[str] | str,
) -> list[Callable]:
    if isinstance(import_names, str):
        import_names = [import_names]

    functions: list[Callable] = []

    for import_name in import_names:
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

    if hasattr(obj, "__class__"):
        obj = obj.__class__

    name = obj.__module__

    if not name.endswith(obj.__qualname__):
        name += f".{obj.__qualname__}"

    return logging.getLogger(name)


def use_mp(pool_or_processes: Pool | int | None) -> bool:
    return pool_or_processes is not None and pool_or_processes != 1


def mp_map[U, V](
    func: Callable[[U], V],
    batches: Iterable[U],
    pool_or_processes: Pool | int | None,
) -> list[V]:
    if isinstance(pool_or_processes, int):
        pool_processes = None if pool_or_processes <= 0 else pool_or_processes
        pool = Pool(pool_processes)
    elif isinstance(pool_or_processes, Pool):
        pool = pool_or_processes
    else:
        raise TypeError(f"Invalid multiprocessing value: {pool_or_processes}")

    with pool as p:
        return p.map(func, batches)


def mp_starmap[*Us, V](
    func: Callable[[*Us], V],
    batches: Iterable[tuple[*Us]],
    pool_or_processes: Pool | int | None,
) -> list[V]:
    if isinstance(pool_or_processes, int):
        pool_processes = None if pool_or_processes <= 0 else pool_or_processes
        pool = Pool(pool_processes)
    elif isinstance(pool_or_processes, Pool):
        pool = pool_or_processes
    else:
        raise TypeError(f"Invalid multiprocessing value: {pool_or_processes}")

    with pool as p:
        return p.starmap(func, batches)
