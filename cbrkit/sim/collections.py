from collections.abc import Collection, Set
from typing import Any

from cbrkit.helpers import dist2sim
from cbrkit.typing import SimPairFunc
from typing import List

__all__ = ["jaccard"]


def jaccard() -> SimPairFunc[Collection[Any], float]:
    """Jaccard similarity function.

    Examples:
        >>> sim = jaccard()
        >>> sim(["a", "b", "c", "d"], ["a", "b", "c"])
        0.8
    """
    from nltk.metrics import jaccard_distance

    def wrapped_func(x: Collection[Any], y: Collection[Any]) -> float:
        if not isinstance(x, Set):
            x = set(x)
        if not isinstance(y, Set):
            y = set(y)

        return dist2sim(jaccard_distance(x, y))

    return wrapped_func


def smith_waterman(match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -1) -> SimPairFunc[str, float]:
    """
    Performs the Smith-Waterman alignment with configurable scoring parameters. If no element matches it returns 0.0.

    Args:
        match_score (int, optional): Score for matching characters. Defaults to 2.
        mismatch_penalty (int, optional): Penalty for mismatching characters. Defaults to -1.
        gap_penalty (int, optional): Penalty for gaps. Defaults to -1.

    Returns:
        float: Alignment score for the two sequences.

    Example:
        >>> sim = smith_waterman()
        >>> sim("abcde", "fghe")
        2
    """
    from minineedle import smith, core

    def wrapped_func(x: str, y: str) -> float:
        try:
            alignment = smith.SmithWaterman(x, y)
            alignment.change_matrix(core.ScoreMatrix(match=match_score, miss=mismatch_penalty, gap=gap_penalty))
            alignment.align()
            return alignment.get_score()
        except ZeroDivisionError:
            return 0.0

    return wrapped_func


def dtw_similarity() -> SimPairFunc[List[int], float]:
    """Dynamic Time Warping similarity function.

    Examples:
        >>> sim = dtw_similarity()
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        0.5
    """
    from dtaidistance import dtw
    import numpy as np

    def wrapped_func(x: List[float], y: List[float]) -> float:
        distance = dtw.distance(np.array(x), np.array(y))
        return float(dist2sim(distance))

    return wrapped_func
