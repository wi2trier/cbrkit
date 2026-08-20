from collections.abc import Sequence
from dataclasses import dataclass, field

from ...typing import Float, StructuredValue

__all__ = [
    "SequenceSim",
]


@dataclass(slots=True, frozen=True)
class SequenceSim[V, S: Float](StructuredValue[float]):
    """
    A class representing sequence similarity with optional mapping and similarity scores.

    Attributes:
        value: The overall similarity score as a float.
        similarities: Optional local similarity scores as a sequence of floats.
        mapping: Optional alignment information as a sequence of tuples.
    """

    similarities: Sequence[S] | None = field(default=None)
    mapping: Sequence[tuple[V | None, V | None]] | None = field(default=None)
