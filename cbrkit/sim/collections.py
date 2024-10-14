from collections.abc import Collection, Sequence, Set
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, cast, override

from cbrkit.helpers import dist2sim, get_metadata, unpack_sim
from cbrkit.typing import AnnotatedFloat, Float, JsonDict, SimPairFunc, SupportsMetadata

Number = float | int

__all__ = [
    "isolated_mapping",
    "sequence_mapping",
    "sequence_correctness",
    "SequenceSim",
    "Weight",
]

try:
    from nltk.metrics import jaccard_distance

    @dataclass(slots=True, frozen=True)
    class jaccard[V](SimPairFunc[Collection[Any], float], SupportsMetadata):
        """Jaccard similarity function.

        Examples:
            >>> sim = jaccard()
            >>> sim(["a", "b", "c", "d"], ["a", "b", "c"])
            0.8
        """

        @override
        def __call__(self, x: Collection[V], y: Collection[V]) -> float:
            if not isinstance(x, Set):
                x = set(x)
            if not isinstance(y, Set):
                y = set(y)

            return dist2sim(jaccard_distance(x, y))

    __all__ += ["jaccard"]

except ImportError:
    pass


try:
    from minineedle import core, smith

    @dataclass(slots=True, frozen=True)
    class smith_waterman[V](SimPairFunc[Sequence[Any], float], SupportsMetadata):
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

        match_score: int = 2
        mismatch_penalty: int = -1
        gap_penalty: int = -1

        @override
        def __call__(self, x: Sequence[V], y: Sequence[V]) -> float:
            try:
                alignment = smith.SmithWaterman(cast(Sequence, x), cast(Sequence, y))
                alignment.change_matrix(
                    core.ScoreMatrix(
                        match=self.match_score,
                        miss=self.mismatch_penalty,
                        gap=self.gap_penalty,
                    )
                )
                alignment.align()

                return alignment.get_score()
            except ZeroDivisionError:
                return 0.0

    __all__ += ["smith_waterman"]

except ImportError:
    pass


try:
    import numpy as np
    from dtaidistance.dtw import distance

    @dataclass(slots=True, frozen=True)
    class dtw(SimPairFunc[Collection[Number], float], SupportsMetadata):
        """Dynamic Time Warping similarity function.

        Examples:
            >>> sim = dtw()
            >>> sim([1, 2, 3], [1, 2, 3, 4])
            0.5
        """

        @override
        def __call__(
            self, x: Collection[Number] | np.ndarray, y: Collection[Number] | np.ndarray
        ) -> float:
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            return dist2sim(distance(x, y))

    __all__ += ["dtw"]

except ImportError:
    pass


@dataclass(slots=True, frozen=True)
class isolated_mapping[V](SimPairFunc[Sequence[V], float], SupportsMetadata):
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

    element_similarity: SimPairFunc[V, float]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {"element_similarity": get_metadata(self.element_similarity)}

    @override
    def __call__(self, x: Sequence[V], y: Sequence[V]) -> float:
        total_similarity = 0.0

        for xi in x:
            max_similarity = max(self.element_similarity(xi, yi) for yi in y)
            total_similarity += max_similarity

        average_similarity = total_similarity / len(x)

        return average_similarity


try:
    import heapq

    @dataclass(slots=True, frozen=True)
    class mapping[V](SimPairFunc[Sequence[V], float], SupportsMetadata):
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

        element_similarity: SimPairFunc[V, float]
        max_queue_size: int = 1000

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "element_similarity": get_metadata(self.element_similarity),
                "max_queue_size": self.max_queue_size,
            }

        @override
        def __call__(self, query: Sequence[V], case: Sequence[V]) -> float:
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
                        sim_score = self.element_similarity(query_item, case_item)
                        new_mapping = current_mapping | {(query_item, case_item)}
                        new_score = (
                            current_score + sim_score
                        )  # Accumulate score correctly
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

                        if len(pq) > self.max_queue_size:
                            heapq.heappop(pq)

            return best_score

    __all__ += ["mapping"]

except ImportError:
    pass


@dataclass(slots=True, frozen=True)
class SequenceSim[S: Float](AnnotatedFloat):
    value: float
    local_similarities: list[S] = field(default_factory=list)


@dataclass
class Weight:
    weight: float
    lower_bound: float
    upper_bound: float
    inclusive_lower: bool
    inclusive_upper: bool
    normalized_weight: float | None = None


@dataclass(slots=True, frozen=True)
class sequence_mapping[V, S: Float](
    SimPairFunc[Sequence[V], SequenceSim[S]], SupportsMetadata
):
    """List Mapping similarity function.

    Parameters:
    element_similarity: The similarity function to use for comparing elements.
    exact: Whether to use exact or inexact comparison. Default is False (inexact).
    weights: Optional list of weights for weighted similarity calculation.

    Examples:
        >>> sim = sequence_mapping(lambda x, y: 1.0 if x == y else 0.0, True)
        >>> result = sim(["a", "b", "c"], ["a", "b", "c"])
        >>> result.value
        1.0
        >>> result.local_similarities
        [1.0, 1.0, 1.0]
    """

    element_similarity: SimPairFunc[V, S]
    exact: bool = False
    weights: list[Weight] | None = None

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "element_similarity": get_metadata(self.element_similarity),
            "exact": self.exact,
            "weights": [asdict(weight) for weight in self.weights]
            if self.weights
            else None,
        }

    def compute_contains_exact(
        self, list1: Sequence[V], list2: Sequence[V]
    ) -> SequenceSim[S]:
        if len(list1) != len(list2):
            return SequenceSim(value=0.0)

        sim_sum = 0.0
        local_similarities: list[S] = []

        for elem1, elem2 in zip(list1, list2, strict=False):
            sim = self.element_similarity(elem1, elem2)
            sim_sum += unpack_sim(sim)
            local_similarities.append(sim)

        return SequenceSim(
            value=sim_sum / len(list1), local_similarities=local_similarities
        )

    def compute_contains_inexact(
        self, larger_list: Sequence[V], smaller_list: Sequence[V]
    ) -> SequenceSim[S]:
        max_similarity = -1.0
        best_local_similarities = []

        for i in range(len(larger_list) - len(smaller_list) + 1):
            sublist = larger_list[i : i + len(smaller_list)]
            sim_result = self.compute_contains_exact(sublist, smaller_list)

            if sim_result.value > max_similarity:
                max_similarity = sim_result.value
                best_local_similarities = sim_result.local_similarities

        return SequenceSim(
            value=max_similarity, local_similarities=best_local_similarities
        )

    @override
    def __call__(self, x: Sequence[V], y: Sequence[V]) -> SequenceSim[S]:
        if self.exact:
            result = self.compute_contains_exact(x, y)
        else:
            if len(x) >= len(y):
                result = self.compute_contains_inexact(x, y)
            else:
                result = self.compute_contains_inexact(y, x)

        if self.weights:
            total_weighted_sim = 0.0
            total_weight = 0.0

            # Arrange and normalize weights
            for weight in self.weights:
                weight_range = weight.upper_bound - weight.lower_bound
                weight.normalized_weight = weight.weight / weight_range

            for sim in result.local_similarities:
                sim = unpack_sim(sim)

                for weight in self.weights:
                    lower_bound = weight.lower_bound
                    upper_bound = weight.upper_bound
                    inclusive_lower = weight.inclusive_lower
                    inclusive_upper = weight.inclusive_upper

                    if (
                        (inclusive_lower and lower_bound <= sim <= upper_bound)
                        or (not inclusive_lower and lower_bound < sim <= upper_bound)
                    ) and (inclusive_upper or sim < upper_bound):
                        assert weight.normalized_weight is not None
                        weighted_sim = weight.normalized_weight * sim
                        total_weighted_sim += weighted_sim
                        total_weight += weight.normalized_weight

            if total_weight > 0:
                final_similarity = total_weighted_sim / total_weight
            else:
                final_similarity = result.value

            return SequenceSim(
                value=final_similarity, local_similarities=result.local_similarities
            )
        else:
            return result


@dataclass(slots=True, frozen=True)
class sequence_correctness[V](SimPairFunc[Sequence[Any], float], SupportsMetadata):
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

        count_concordant = 0
        count_discordant = 0

        for i, j in product(range(len(x)), repeat=2):
            if i >= j:
                continue

            index_x1 = x.index(x[i])
            index_x2 = x.index(x[j])
            index_y1 = y.index(x[i])
            index_y2 = y.index(x[j])

            if index_y1 == -1 or index_y2 == -1:
                continue
            elif (index_x1 < index_x2 and index_y1 < index_y2) or (
                index_x1 > index_x2 and index_y1 > index_y2
            ):
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
        else:
            return abs(correctness) * self.worst_case_sim
