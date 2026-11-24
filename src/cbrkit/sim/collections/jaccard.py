from collections.abc import Collection
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Literal, override

from ...schema import serializable
from ...typing import SimFunc

__all__ = [
    "Jaccard",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Jaccard[V](SimFunc[Collection[V], float]):
    """Jaccard similarity of two collections interpreted as sets.

    Two empty collections are identical and have a similarity of 1.0.

    Examples:
        >>> sim = Jaccard()
        >>> sim(["a", "b", "c", "d"], ["a", "b", "c"])
        0.75
        >>> sim([], [])
        1.0
    """

    kind: Literal["jaccard"] = field(default="jaccard", repr=False)

    @override
    def __call__(self, case: Collection[V], query: Collection[V]) -> float:
        x_set = case if isinstance(case, AbstractSet) else set(case)
        y_set = query if isinstance(query, AbstractSet) else set(query)
        union = x_set | y_set

        if not union:
            return 1.0

        return len(x_set & y_set) / len(union)
