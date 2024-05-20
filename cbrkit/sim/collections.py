from collections.abc import Collection, Sequence, Set
from typing import Any

from cbrkit.helpers import dist2sim
from cbrkit.typing import SimPairFunc, ValueType

Number = float | int

__all__ = ["jaccard", "smith_waterman", "dtw"]


def jaccard() -> SimPairFunc[Collection[Any], float]:
    """Jaccard similarity function.

    Examples:
        >>> sim = jaccard()
        >>> sim(["a", "b", "c", "d"], ["a", "b", "c"])
        0.8
    """
    from nltk.metrics import jaccard_distance

    def wrapped_func(x: Collection[ValueType], y: Collection[ValueType]) -> float:
        if not isinstance(x, Set):
            x = set(x)
        if not isinstance(y, Set):
            y = set(y)

        return dist2sim(jaccard_distance(x, y))

    return wrapped_func


def smith_waterman(
    match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -1
) -> SimPairFunc[Sequence[Any], float]:
    """
    Performs the Smith-Waterman alignment with configurable scoring parameters. If no element matches it returns 0.0.

    Args:
        match_score: Score for matching characters. Defaults to 2.
        mismatch_penalty: Penalty for mismatching characters. Defaults to -1.
        gap_penalty: Penalty for gaps. Defaults to -1.

    Example:
        >>> sim = smith_waterman()
        >>> sim("abcde", "fghe")
        2
    """
    from minineedle import core, smith

    def wrapped_func(x: Sequence[ValueType], y: Sequence[ValueType]) -> float:
        try:
            alignment = smith.SmithWaterman(x, y)
            alignment.change_matrix(
                core.ScoreMatrix(
                    match=match_score, miss=mismatch_penalty, gap=gap_penalty
                )
            )
            alignment.align()

            return alignment.get_score()
        except ZeroDivisionError:
            return 0.0

    return wrapped_func


def dtw() -> SimPairFunc[Collection[Number], float]:
    """Dynamic Time Warping similarity function.

    Examples:
        >>> sim = dtw()
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        0.5
    """
    import numpy as np
    from dtaidistance import dtw

    def wrapped_func(
        x: Collection[Number] | np.ndarray, y: Collection[Number] | np.ndarray
    ) -> float:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        distance = dtw.distance(x, y)

        return dist2sim(distance)

    return wrapped_func


def isolated_mapping(
    element_similarity: SimPairFunc[ValueType, float],
) -> SimPairFunc[Sequence[ValueType], float]:
    """
    Isolated Mapping similarity function that compares each element in 'x'
    with all elements in 'y'
    and takes the maximum similarity for each element in 'x', then averages
    these maximums.

    Args:
        element_similarity: A function that takes two elements and returns
        a similarity score between them.

    Examples:
        >>> from cbrkit.sim.strings import levenshtein
        >>> sim = isolated_mapping(levenshtein())
        >>> sim(["kitten", "sitting"], ["sitting", "fitted"])
        0.8333333333333334
    """

    def wrapped_func(x: Sequence[ValueType], y: Sequence[ValueType]) -> float:
        total_similarity = 0.0

        for xi in x:
            max_similarity = max(element_similarity(xi, yi) for yi in y)
            total_similarity += max_similarity

        average_similarity = total_similarity / len(x)

        return average_similarity

    return wrapped_func


def mapping(
    element_similarity: SimPairFunc[ValueType, float], max_queue_size: int = 1000
) -> SimPairFunc[Sequence[ValueType], float]:
    """
    Implements an A* algorithm to find the best matching between query items and case items
    based on the provided similarity function, maximizing the overall similarity score.

    Args:
        similarity_function: A function that calculates the similarity between two elements.
        max_queue_size: Maximum size of the priority queue. Defaults to 1000.

    Returns:
        A similarity function for sequences.

    Examples:
        >>> def example_similarity_function(x: Any, y: Any) -> float:
        ...     return 1.0 if x == y else 0.0
        >>> sim_func = mapping(example_similarity_function)
        >>> result = sim_func(["Monday", "Tuesday", "Wednesday"], ["Monday", "Tuesday", "Sunday"])
        >>> print(f"Normalized Similarity Score: {result}")
        Normalized Similarity Score: 0.6666666666666666
    """
    import heapq

    def wrapped_func(query: Sequence[ValueType], case: Sequence[ValueType]) -> float:
        # Priority queue to store potential solutions with their scores
        pq = []
        initial_solution = (0.0, set(), frozenset(query), frozenset(case))
        heapq.heappush(pq, initial_solution)

        best_score = 0.0

        while pq:
            current_score, current_mapping, remaining_query, remaining_case = (
                heapq.heappop(pq)
            )

            if not remaining_query:  # All query items are mapped
                best_score = max(best_score, current_score / len(query))
                continue  # Continue to process remaining potential mappings

            for query_item in remaining_query:
                for case_item in remaining_case:
                    sim_score = element_similarity(query_item, case_item)
                    new_mapping = current_mapping | {(query_item, case_item)}
                    new_score = current_score + sim_score  # Accumulate score correctly
                    new_remaining_query = remaining_query - {query_item}
                    new_remaining_case = remaining_case - {case_item}

                    heapq.heappush(
                        pq,
                        (
                            new_score,
                            new_mapping,
                            new_remaining_query,
                            new_remaining_case,
                        ),
                    )

                    if len(pq) > max_queue_size:
                        heapq.heappop(pq)

        return best_score

    return wrapped_func
