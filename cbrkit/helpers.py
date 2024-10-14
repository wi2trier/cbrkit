from collections.abc import Collection, Iterable, Mapping, Sequence
from inspect import signature as inspect_signature
from typing import Any, cast

from cbrkit.typing import (
    Float,
    JsonDict,
    SimMap,
    SimMapFunc,
    SimPairFunc,
    SimSeq,
    SimSeqFunc,
    SupportsMetadata,
)

__all__ = [
    "dist2sim",
    "sim2seq",
    "sim2map",
    "sim2pair",
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


def sim2seq[V, S: Float](
    func: SimPairFunc[V, S] | SimSeqFunc[V, S],
) -> SimSeqFunc[V, S]:
    """
    Converts a similarity function that operates on pairs of values into a similarity function that operates on sequences of values.

    Args:
        func: The similarity function to be converted.

    Examples:
        >>> def sim_func(x: int, y: int) -> float:
        ...     return abs(x - y) / max(x, y)
        ...
        >>> seq_func = sim2seq(sim_func)
        >>> seq_func([(1, 2), (3, 4), (5, 6)])
        [0.5, 0.25, 0.16666666666666666]
    """
    signature = inspect_signature(func)

    if len(signature.parameters) == 2:
        casted_func = cast(SimPairFunc[V, S], func)

        def wrapped_sim_pair_func(pairs: Sequence[tuple[V, V]]) -> SimSeq:
            return [casted_func(x, y) for (x, y) in pairs]

        return wrapped_sim_pair_func

    elif len(signature.parameters) == 1:
        return cast(SimSeqFunc[V, S], func)

    raise TypeError(
        f"Invalid signature for similarity function: {signature.parameters}"
    )


def sim2map[K, V, S: Float](
    func: SimPairFunc[V, S] | SimSeqFunc[V, S] | SimMapFunc[K, V, S],
) -> SimMapFunc[K, V, S]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2 and signature.parameters.keys() in (
        {"x_map", "y"},
        {"casebase", "query"},
    ):
        return cast(SimMapFunc[K, V, S], func)

    elif len(signature.parameters) == 2:
        sim_pair_func = cast(SimPairFunc[V, S], func)

        def wrapped_sim_pair_func(
            x_map: Mapping[K, V],
            y: V,
        ) -> SimMap[K, S]:
            return {key: sim_pair_func(x, y) for key, x in x_map.items()}

        return wrapped_sim_pair_func

    elif len(signature.parameters) == 1:
        sim_seq_func = cast(SimSeqFunc[V, S], func)

        def wrapped_sim_seq_func(x_map: Mapping[Any, V], y: V) -> SimMap[K, S]:
            pairs = [(x, y) for x in x_map.values()]
            sims = sim_seq_func(pairs)

            return {key: sim for key, sim in zip(x_map.keys(), sims, strict=True)}

        return wrapped_sim_seq_func

    raise TypeError(
        f"Invalid signature for similarity function: {signature.parameters}"
    )


def sim2pair[V, S: Float](
    func: SimPairFunc[V, S] | SimSeqFunc[V, S] | SimMapFunc[Any, V, S],
) -> SimPairFunc[V, S]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2 and signature.parameters.keys() in (
        {"x_map", "y"},
        {"casebase", "query"},
    ):
        sim_map_func = cast(SimMapFunc[Any, V, S], func)

        def wrapped_sim_map_func(
            x: V,
            y: V,
        ) -> S:
            return sim_map_func({"x": x}, y)["x"]

        return wrapped_sim_map_func

    elif len(signature.parameters) == 2:
        return cast(SimPairFunc[V, S], func)

    elif len(signature.parameters) == 1:
        sim_seq_func = cast(SimSeqFunc[V, S], func)

        def wrapped_sim_seq_func(x: V, y: V) -> S:
            return sim_seq_func([(x, y)])[0]

        return wrapped_sim_seq_func

    raise TypeError(
        f"Invalid signature for similarity function: {signature.parameters}"
    )


def unpack_sim(sim: Float) -> float:
    if isinstance(sim, float | int | bool):
        return sim
    else:
        return sim.value


def unpack_sims(sims: Iterable[Float]) -> list[float]:
    return [unpack_sim(sim) for sim in sims]
