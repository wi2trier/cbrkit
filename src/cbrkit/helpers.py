import dataclasses
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from importlib import import_module
from inspect import getdoc
from inspect import signature as inspect_signature
from typing import Any, cast, override

from .typing import (
    AnyGenerationFunc,
    AnyNamedFunc,
    AnyPositionalFunc,
    AnySimFunc,
    BatchGenerationFunc,
    BatchPositionalFunc,
    BatchSimFunc,
    Float,
    GenerationFunc,
    HasMetadata,
    JsonDict,
    JsonEntry,
    NamedBatchFunc,
    NamedFunc,
    PositionalFunc,
    SimFunc,
    SimMap,
    SimSeq,
    StructuredValue,
)

__all__ = [
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
]


def get_name(obj: Any) -> str | None:
    if obj is None:
        return None

    elif isinstance(obj, type):
        return obj.__name__

    return type(obj).__name__


def get_metadata(obj: Any) -> JsonEntry:
    if hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__

    if isinstance(obj, HasMetadata):
        value: JsonDict = {
            "metadata": obj.metadata,
        }

        if isinstance(obj, Callable) or isinstance(obj, type):
            value["name"] = get_name(obj)
            value["doc"] = getdoc(obj)

        return value

    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            "name": get_name(obj),
            "doc": getdoc(obj),
            "metadata": {
                field.name: get_metadata(getattr(obj, field.name))
                for field in dataclasses.fields(obj)
                if field.repr
            },
        }

    if isinstance(obj, Callable) or isinstance(obj, type):
        return {
            "name": get_name(obj),
            "doc": getdoc(obj),
        }

    if isinstance(obj, dict):
        return {key: get_metadata(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [get_metadata(value) for value in obj]

    if isinstance(obj, dict | list | str | int | float | bool | None):
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
    """Yield a total of k chunks from val.

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


@dataclass(slots=True)
class batchify_positional[T](BatchPositionalFunc[T]):
    __wrapped__: AnyPositionalFunc[T]
    parameters: int = field(init=False)

    def __init__(self, func: AnyPositionalFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)

    @override
    def __call__(self, batches: Sequence[tuple[Any, ...]]) -> Sequence[T]:
        if self.parameters != 1:
            func = cast(PositionalFunc[T], self.__wrapped__)
            return [func(*batch) for batch in batches]

        func = cast(BatchPositionalFunc[T], self.__wrapped__)
        return func(batches)


@dataclass(slots=True)
class unbatchify_positional[T](PositionalFunc[T]):
    __wrapped__: AnyPositionalFunc[T]
    parameters: int = field(init=False)

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


@dataclass(slots=True)
class batchify_named[T](NamedBatchFunc[T]):
    __wrapped__: AnyNamedFunc[T]
    parameters: int = field(init=False)

    def __init__(self, func: AnyNamedFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)

    @override
    def __call__(self, batches: Sequence[dict[str, Any]]) -> Sequence[T]:
        if self.parameters != 1:
            func = cast(NamedFunc[T], self.__wrapped__)
            return [func(**batch) for batch in batches]

        func = cast(NamedBatchFunc[T], self.__wrapped__)
        return func(batches)


@dataclass(slots=True)
class unbatchify_named[T](NamedFunc[T]):
    __wrapped__: AnyNamedFunc[T]
    parameters: int = field(init=False)

    def __init__(self, func: AnyNamedFunc[T]):
        self.__wrapped__ = func
        signature = inspect_signature(func)
        self.parameters = len(signature.parameters)

    @override
    def __call__(self, **kwargs: Any) -> T:
        if self.parameters == 1:
            func = cast(NamedBatchFunc[T], self.__wrapped__)
            return func([kwargs])[0]

        func = cast(NamedFunc[T], self.__wrapped__)
        return func(**kwargs)


def batchify_sim[V, S: Float](sim_func: AnySimFunc[V, S]) -> BatchSimFunc[V, S]:
    return batchify_positional(cast(AnyPositionalFunc[S], sim_func))


def unbatchify_sim[V, S: Float](sim_func: AnySimFunc[V, S]) -> SimFunc[V, S]:
    return cast(
        SimFunc[V, S], unbatchify_positional(cast(AnyPositionalFunc[S], sim_func))
    )


def batchify_generation[P, R](
    generation_func: AnyGenerationFunc[P, R],
) -> BatchGenerationFunc[P, R]:
    return cast(
        BatchGenerationFunc[P, R],
        batchify_positional(cast(AnyPositionalFunc[R], generation_func)),
    )


def unbatchify_generation[P, R](
    generation_func: AnyGenerationFunc[P, R],
) -> GenerationFunc[P, R]:
    return cast(
        GenerationFunc[P, R],
        batchify_positional(cast(AnyPositionalFunc[R], generation_func)),
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
