from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, override

from ..helpers import batchify_sim
from ..typing import (
    AnySimFunc,
    BatchSimFunc,
    ConversionFunc,
    Float,
    SimSeq,
)


@dataclass(slots=True)
class transpose[V1, V2, S: Float](BatchSimFunc[V1, S]):
    """Transforms a similarity function from one type to another.

    Args:
        conversion_func: A function that converts the input values from one type to another.
        similarity_func: The similarity function to be used on the converted values.

    Examples:
        >>> from cbrkit.sim.generic import equality
        >>> sim = transpose(
        ...     similarity_func=equality(),
        ...     conversion_func=lambda x: x.lower(),
        ... )
        >>> sim([("A", "a"), ("b", "B")])
        [1.0, 1.0]
    """

    similarity_func: BatchSimFunc[V2, S]
    conversion_func: ConversionFunc[V1, V2]

    def __init__(
        self,
        similarity_func: AnySimFunc[V2, S],
        conversion_func: ConversionFunc[V1, V2],
    ):
        self.similarity_func = batchify_sim(similarity_func)
        self.conversion_func = conversion_func

    @override
    def __call__(self, batches: Sequence[tuple[V1, V1]]) -> Sequence[S]:
        return self.similarity_func(
            [(self.conversion_func(x), self.conversion_func(y)) for x, y in batches]
        )


@dataclass(slots=True)
class cache[V, U, S: Float](BatchSimFunc[V, S]):
    similarity_func: BatchSimFunc[V, S]
    conversion_func: ConversionFunc[V, U] | None
    cache: dict[tuple[U, U], S]

    def __init__(
        self,
        similarity_func: AnySimFunc[V, S],
        conversion_func: ConversionFunc[V, U] | None = None,
    ):
        self.similarity_func = batchify_sim(similarity_func)
        self.conversion_func = conversion_func
        self.cache = {}

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[S]:
        transformed_batches = (
            [(self.conversion_func(x), self.conversion_func(y)) for x, y in batches]
            if self.conversion_func is not None
            else cast(list[tuple[U, U]], batches)
        )
        uncached_indexes = [
            idx
            for idx, pair in enumerate(transformed_batches)
            if pair not in self.cache
        ]

        uncached_sims = self.similarity_func([batches[idx] for idx in uncached_indexes])
        self.cache.update(
            {
                transformed_batches[idx]: sim
                for idx, sim in zip(uncached_indexes, uncached_sims, strict=True)
            }
        )

        return [self.cache[pair] for pair in transformed_batches]
