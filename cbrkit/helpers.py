from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import signature as inspect_signature
from typing import Any, Literal, cast, override

from cbrkit.typing import (
    AnySimFunc,
    Float,
    JsonDict,
    SimMap,
    SimMapFunc,
    SimPairFunc,
    SimSeqFunc,
    SupportsMetadata,
)

__all__ = [
    "dist2sim",
    "SimWrapper",
    "SimSeqWrapper",
    "SimMapWrapper",
    "SimPairWrapper",
    "unpack_sim",
    "unpack_sims",
    "singleton",
]


def get_name(obj: Any | None) -> str | None:
    if obj is None:
        return None
    elif isinstance(obj, type):
        return obj.__name__

    return type(obj).__name__


def get_metadata(obj: Any) -> JsonDict:
    if isinstance(obj, SimWrapper):
        obj = obj.func

    metadata = {}

    if isinstance(obj, SupportsMetadata):
        metadata = obj.metadata

    metadata["name"] = get_name(obj)

    return metadata


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
    else:
        raise TypeError(f"Expected a Mapping or Collection, but got {type(x)}")


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
class SimWrapper[V, S: Float](SupportsMetadata):
    func: AnySimFunc[V, S]
    kind: Literal["pair", "seq"] = field(init=False)

    def __post_init__(self):
        signature = inspect_signature(self.func)

        if len(signature.parameters) == 2:
            self.kind = "pair"
        else:
            self.kind = "seq"


class SimSeqWrapper[V, S: Float](SimWrapper, SimSeqFunc[V, S]):
    @override
    def __call__(self, pairs: Sequence[tuple[V, V]]) -> Sequence[S]:
        if self.kind == "pair":
            func = cast(SimPairFunc[V, S], self.func)
            return [func(x, y) for (x, y) in pairs]

        func = cast(SimSeqFunc[V, S], self.func)
        return func(pairs)


class SimMapWrapper[V, S: Float](SimWrapper, SimMapFunc[Any, V, S]):
    @override
    def __call__(self, x_map: Mapping[Any, V], y: V) -> SimMap[Any, S]:
        if self.kind == "seq":
            func = cast(SimSeqFunc[V, S], self.func)
            pairs = [(x, y) for x in x_map.values()]
            return {
                key: sim for key, sim in zip(x_map.keys(), func(pairs), strict=True)
            }

        func = cast(SimPairFunc[V, S], self.func)
        return {key: func(x, y) for key, x in x_map.items()}


class SimPairWrapper[V, S: Float](SimWrapper, SimPairFunc[V, S]):
    @override
    def __call__(self, x: V, y: V) -> S:
        if self.kind == "seq":
            func = cast(SimSeqFunc[V, S], self.func)
            return func([(x, y)])[0]

        func = cast(SimPairFunc[V, S], self.func)
        return func(x, y)


def unpack_sim(sim: Float) -> float:
    if isinstance(sim, float | int | bool):
        return sim
    else:
        return sim.value


def unpack_sims(sims: Iterable[Float]) -> list[float]:
    return [unpack_sim(sim) for sim in sims]
