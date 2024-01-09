from collections.abc import Collection, Set
from typing import Any

from cbrkit.sim._helpers import dist2sim
from cbrkit.typing import SimPairFunc


def jaccard() -> SimPairFunc[Collection[Any], float]:
    from nltk.metrics import jaccard_distance

    def wrapped_func(x: Collection[Any], y: Collection[Any]) -> float:
        if not isinstance(x, Set):
            x = set(x)
        if not isinstance(y, Set):
            y = set(y)

        return dist2sim(jaccard_distance(x, y))

    return wrapped_func
