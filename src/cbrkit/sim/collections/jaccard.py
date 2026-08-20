from collections.abc import Collection
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import override

from ...helpers import dist2sim, optional_dependencies
from ...typing import SimFunc

__all__ = [
    "jaccard",
]


with optional_dependencies():
    from nltk.metrics import jaccard_distance

    @dataclass(slots=True, frozen=True)
    class jaccard[V](SimFunc[Collection[V], float]):
        """Jaccard similarity function.

        Examples:
            >>> sim = jaccard()
            >>> sim(["a", "b", "c", "d"], ["a", "b", "c"])
            0.8
        """

        @override
        def __call__(self, x: Collection[V], y: Collection[V]) -> float:
            if not isinstance(x, AbstractSet):
                x = set(x)
            if not isinstance(y, AbstractSet):
                y = set(y)

            return dist2sim(jaccard_distance(x, y))
