from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import override

from ...typing import SimFunc

__all__ = [
    "sequence_correctness",
]


@dataclass(slots=True, frozen=True)
class sequence_correctness[V](SimFunc[Sequence[V], float]):
    """List Correctness similarity function.

    Parameters:
    worst_case_sim (float): The similarity value to use when all pairs are discordant. Default is 0.0.

    Examples:
        >>> sim = sequence_correctness(0.5)
        >>> sim(["Monday", "Tuesday", "Wednesday"], ["Monday", "Wednesday", "Tuesday"])
        0.3333333333333333
    """

    worst_case_sim: float = 0.0

    @override
    def __call__(self, x: Sequence[V], y: Sequence[V]) -> float:
        if len(x) != len(y):
            return 0.0

        # Looking every element up once keeps the pair loop free of scans over y.
        positions = [y.index(value) for value in x if value in y]
        count_concordant = 0
        count_discordant = 0

        # The elements are enumerated in the order of x, so a pair is concordant
        # exactly when y places them in that same order.
        for first, second in combinations(positions, 2):
            if first < second:
                count_concordant += 1
            else:
                count_discordant += 1

        if count_concordant + count_discordant == 0:
            return 0.0

        correctness = (count_concordant - count_discordant) / (
            count_concordant + count_discordant
        )

        if correctness >= 0:
            return correctness

        return abs(correctness) * self.worst_case_sim
