from collections.abc import Collection, Set
from typing import Any

from cbrkit.sim.helpers import dist2sim, sim2seq
from cbrkit.typing import SimSeqFunc


def jaccard() -> SimSeqFunc[Collection[Any]]:
    from nltk.metrics import jaccard_distance

    @sim2seq
    def wrapped_func(x: Collection[Any], y: Collection[Any]) -> float:
        if not isinstance(x, Set):
            x = set(x)
        if not isinstance(y, Set):
            y = set(y)

        return dist2sim(jaccard_distance(x, y))

    return wrapped_func
