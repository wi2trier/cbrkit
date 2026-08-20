from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ...helpers import get_metadata, unpack_float
from ...typing import Float, HasMetadata, JsonDict, SimFunc
from .common import SequenceSim

__all__ = [
    "sequence_mapping",
    "Weight",
]


@dataclass(slots=True, frozen=True)
class Weight:
    """A weighted interval with bounds for sequence mapping similarity."""

    weight: float
    lower_bound: float
    upper_bound: float
    inclusive_lower: bool
    inclusive_upper: bool

    def contains(self, value: float) -> bool:
        """Whether the given similarity lies within the interval."""
        lower = (
            self.lower_bound <= value
            if self.inclusive_lower
            else self.lower_bound < value
        )
        upper = (
            value <= self.upper_bound
            if self.inclusive_upper
            else value < self.upper_bound
        )

        return lower and upper

    @property
    def normalized_weight(self) -> float:
        """The weight scaled by the width of the interval."""
        width = self.upper_bound - self.lower_bound

        return self.weight / width if width > 0.0 else self.weight


@dataclass(slots=True, frozen=True)
class sequence_mapping[V, S: Float](
    SimFunc[Sequence[V], SequenceSim[V, S]], HasMetadata
):
    """
    List Mapping similarity function.

    Parameters:
        element_similarity: The similarity function to use for comparing elements.
        exact: Whether to use exact or inexact comparison. Default is False (inexact).
        weights: Optional list of weights for weighted similarity calculation.

    Examples:
        >>> sim = sequence_mapping(lambda x, y: 1.0 if x == y else 0.0, True)
        >>> result = sim(["a", "b", "c"], ["a", "b", "c"])
        >>> result.value
        1.0
        >>> result.similarities
        [1.0, 1.0, 1.0]
    """

    element_similarity: SimFunc[V, S]
    exact: bool = False
    weights: list[Weight] | None = None

    @property
    def metadata(self) -> JsonDict:
        """Return metadata describing the sequence mapping configuration."""
        return {
            "element_similarity": get_metadata(self.element_similarity),
            "exact": self.exact,
            "weights": [asdict(weight) for weight in self.weights]
            if self.weights
            else None,
        }

    def compute_contains_exact(
        self, query: Sequence[V], case: Sequence[V]
    ) -> SequenceSim[V, S]:
        """Compute element-wise similarity for sequences of equal length."""
        if len(query) != len(case):
            return SequenceSim(value=0.0, similarities=None, mapping=None)

        if not query:
            return SequenceSim(value=1.0, similarities=[], mapping=None)

        sim_sum = 0.0
        local_similarities: list[S] = []

        for elem_q, elem_c in zip(query, case, strict=True):
            sim = self.element_similarity(elem_q, elem_c)
            sim_sum += unpack_float(sim)
            local_similarities.append(sim)

        return SequenceSim(
            value=sim_sum / len(query),
            similarities=local_similarities,
            mapping=None,
        )

    def compute_contains_inexact(
        self, case_list: Sequence[V], query_list: Sequence[V]
    ) -> SequenceSim[V, S]:
        """
        Slides the *shorter* sequence across the *longer* one and always
        evaluates   query → case   (i.e. query elements are compared against
        the current window cut from the case list).
        """
        case_is_longer = len(case_list) >= len(query_list)
        larger, smaller = (
            (case_list, query_list) if case_is_longer else (query_list, case_list)
        )

        best_value = 0.0
        best_sims: Sequence[S] = []

        for start in range(len(larger) - len(smaller) + 1):
            window = larger[start : start + len(smaller)]

            if case_is_longer:
                sim_res = self.compute_contains_exact(smaller, window)
            else:
                sim_res = self.compute_contains_exact(window, smaller)

            if sim_res.value > best_value:
                best_value = sim_res.value
                best_sims = sim_res.similarities or []

        return SequenceSim(value=best_value, similarities=best_sims, mapping=None)

    def __call__(self, x: Sequence[V], y: Sequence[V]) -> SequenceSim[V, S]:
        # x is the "case", y is the "query"
        if self.exact:
            result = self.compute_contains_exact(y, x)
        else:
            result = self.compute_contains_inexact(x, y)

        if not self.weights or not result.similarities:
            return result

        total_weighted_sim = 0.0
        total_weight = 0.0

        for sim in result.similarities:
            sim_val = unpack_float(sim)

            for weight in self.weights:
                if weight.contains(sim_val):
                    total_weighted_sim += weight.normalized_weight * sim_val
                    total_weight += weight.normalized_weight

        return SequenceSim(
            value=total_weighted_sim / total_weight
            if total_weight > 0
            else result.value,
            similarities=result.similarities,
            mapping=result.mapping,
        )
