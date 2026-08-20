from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...helpers import unpack_float
from ...typing import Float, SimFunc

__all__ = [
    "mapping",
    "isolated_mapping",
]


@dataclass(slots=True, frozen=True)
class isolated_mapping[V](SimFunc[Sequence[V], float]):
    """
    Isolated Mapping similarity function that compares each element in 'y' (query)
    with all elements in 'x' (case)
    and takes the maximum similarity for each element in 'y', then averages
    these maximums. Assumes y -> x (query operated onto case).

    Args:
        element_similarity: A function that takes two elements (query_item, case_item)
        and returns a similarity score between them.

    Examples:
        >>> from cbrkit.sim.strings import levenshtein
        >>> sim = isolated_mapping(levenshtein())
        >>> sim(["kitten", "sitting"], ["sitting", "fitted"])
        0.8333333333333334
    """

    element_similarity: SimFunc[V, float]

    @override
    def __call__(self, x: Sequence[V], y: Sequence[V]) -> float:
        if not y:
            return 0.0

        return sum(
            max((self.element_similarity(yi, xi) for xi in x), default=0.0) for yi in y
        ) / len(y)


@dataclass(slots=True, frozen=True)
class mapping[V](SimFunc[Sequence[V], float]):
    """
    Computes the best possible mapping of the query items onto the case items.

    Each query item is mapped to at most one case item and each case item is used at
    most once, which is a maximum weight bipartite matching and is solved exactly.
    If the query is longer than the case, the surplus query items remain unmapped and
    contribute a similarity of zero.

    Args:
        element_similarity: A function that calculates the similarity between two elements.

    Returns:
        A similarity function for sequences.

    Examples:
        >>> def example_similarity_function(x, y) -> float:
        ...     return 1.0 if x == y else 0.0
        >>> sim_func = mapping(example_similarity_function)
        >>> sim_func(["Monday", "Tuesday", "Wednesday"], ["Monday", "Tuesday", "Sunday"])
        0.6666666666666666
        >>> sim_func(["Monday"], ["Monday", "Tuesday", "Wednesday"])
        0.3333333333333333
    """

    element_similarity: SimFunc[V, Float]

    @override
    def __call__(self, x: Sequence[V], y: Sequence[V]) -> float:
        if not x or not y:
            return 0.0

        sims = np.array(
            [[unpack_float(self.element_similarity(xi, yi)) for xi in x] for yi in y]
        )
        rows, cols = linear_sum_assignment(sims, maximize=True)

        return float(sims[rows, cols].sum() / len(y))
