"""Similarity measures for collections and sequences.

This module provides similarity functions for unordered collections as well as
for sequences, where the order of the elements matters.
Sequence measures based on an alignment additionally expose the computed
alignment through `SequenceSim`.

Algorithms:
- `jaccard`: Jaccard similarity of two collections interpreted as sets.
- `dtw`: Dynamic Time Warping, which warps the sequences along the time axis.
- `twed`: Time Warp Edit Distance, a metric alternative to `dtw` that deletes
  unmatched elements instead of warping them.
- `smith_waterman`: Smith-Waterman local alignment.
- `mapping`: Optimal one-to-one mapping of the query elements onto the case elements.
- `isolated_mapping`: Maps each query element to its most similar case element.
- `sequence_mapping`: Element-wise comparison of sequences, optionally weighted and
  with a sliding window for sequences of different lengths.
- `sequence_correctness`: Similarity based on the order of the shared elements.

Types:
- `SequenceSim`: Similarity value with optional local similarities and alignment.
- `Weight`: Weighted interval used by `sequence_mapping`.

Example:
    >>> sim = dtw()
    >>> sim([1, 2, 3], [1, 2, 3, 4]).value
    0.5
"""

from ...helpers import optional_dependencies
from .alignment import dtw, smith_waterman, twed
from .common import SequenceSim
from .mapping import isolated_mapping, mapping
from .sequence_correctness import sequence_correctness
from .sequence_mapping import Weight, sequence_mapping

with optional_dependencies():
    from .jaccard import jaccard

__all__ = [
    "dtw",
    "twed",
    "smith_waterman",
    "jaccard",
    "mapping",
    "isolated_mapping",
    "sequence_mapping",
    "sequence_correctness",
    "SequenceSim",
    "Weight",
]
