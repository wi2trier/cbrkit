"""Similarity measures for collections and sequences.

This module provides similarity functions for unordered collections as well as
for sequences, where the order of the elements matters.
Sequence measures based on an alignment additionally expose the computed
alignment through `cbrkit.model.SequenceSim` if their `alignment` field is set.
Measures of elements are similarities everywhere (`element_measure`),
the elastic measures `DTW` and `TWED` take `1 - similarity` as the distance of two elements.
The mapping measures keep the local similarities as returned by the element measure,
two empty sequences are identical and an empty sequence has nothing in common with any other.

Algorithms:
- `Jaccard`: Jaccard similarity of two collections interpreted as sets.
- `DTW`: Dynamic Time Warping, which warps the sequences along the time axis.
- `TWED`: Time Warp Edit Distance, a metric alternative to `DTW` that deletes
  unmatched elements instead of warping them.
- `SmithWaterman`: Smith-Waterman local alignment.
- `OptimalMapping`: Optimal one-to-one mapping of the query elements onto the case elements.
- `IsolatedMapping`: Maps each query element to its most similar case element.
- `SequenceMapping`: Element-wise comparison of sequences, optionally weighted and
  with a sliding window for sequences of different lengths.
- `SequenceCorrectness`: Similarity based on the order of the shared elements.

Types:
- `Weight`: Weighted interval used by `SequenceMapping`.

Example:
    >>> sim = DTW()
    >>> sim([1, 2, 3], [1, 2, 3, 4]).value
    0.5
"""

from .alignment import DTW, TWED, SmithWaterman
from .jaccard import Jaccard
from .mapping import IsolatedMapping, OptimalMapping
from .sequence_correctness import SequenceCorrectness
from .sequence_mapping import SequenceMapping, Weight

__all__ = [
    "DTW",
    "TWED",
    "IsolatedMapping",
    "Jaccard",
    "OptimalMapping",
    "SequenceCorrectness",
    "SequenceMapping",
    "SmithWaterman",
    "Weight",
]
