from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, override

from ...schema import Unit, serializable
from ...typing import SimFunc

__all__ = [
    "SequenceCorrectness",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SequenceCorrectness[V](SimFunc[Sequence[V], float]):
    """Similarity of two sequences by the order of their shared elements.

    Args:
        worst_case_similarity: The similarity when all pairs are discordant.

    Examples:
        >>> sim = SequenceCorrectness(worst_case_similarity=0.5)
        >>> sim(["Monday", "Tuesday", "Wednesday"], ["Monday", "Wednesday", "Tuesday"])
        0.3333333333333333
    """

    worst_case_similarity: Unit = 0.0
    kind: Literal["sequence_correctness"] = field(
        default="sequence_correctness", repr=False
    )

    @override
    def __call__(self, case: Sequence[V], query: Sequence[V]) -> float:
        if len(case) != len(query):
            return 0.0

        # Looking every element up once keeps the pair loop free of scans over y.
        positions = [query.index(value) for value in case if value in query]
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

        return abs(correctness) * self.worst_case_similarity
