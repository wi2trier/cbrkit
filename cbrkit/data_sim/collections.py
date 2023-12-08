from collections.abc import Collection, Set
from typing import Any

from cbrkit import model
from cbrkit.data_sim.helpers import apply, dist2sim


def jaccard() -> model.DataSimilarityBatchFunc[Collection[Any]]:
    from nltk.metrics import jaccard_distance

    @apply
    def wrapped_func(x: Collection[Any], y: Collection[Any]) -> float:
        if not isinstance(x, Set):
            x = set(x)
        if not isinstance(y, Set):
            y = set(y)

        return dist2sim(jaccard_distance(x, y))

    return wrapped_func