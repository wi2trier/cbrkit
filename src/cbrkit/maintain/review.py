"""Review of a case base: pairwise similarities, duplicates, and the competence of the cases.

The competence model and the footprint deletion policy follow
B. Smyth and M. T. Keane, Remembering to Forget: A Competence-Preserving Case Deletion Policy
for Case-Based Reasoning Systems, IJCAI 1995, pp. 377-382,
with the coverage and reachability sets of
B. Smyth and E. McKenna, Competence Models and the Maintenance Problem,
Computational Intelligence 17(2), 2001.
"""

from collections.abc import Collection, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass

from .._phase import score_pairs
from .._util import sorted_keys
from ..errors import Refused
from ..model import Category, Review
from ..typing import AnySimFunc, Casebase, Float, QueryCaseMatrix

__all__ = ["CATEGORIES", "Competence", "Matrix", "duplicates", "pairwise"]

type Matrix[K] = QueryCaseMatrix[K, K, float]
"""Similarity of every case to every other case as query, `matrix[query][case]`."""

CATEGORIES: tuple[Category, ...] = ("auxiliary", "support", "spanning", "pivotal")
"""The categories in the order of footprint deletion, from the least to the most important."""


def pairwise[K, V](
    casebase: Casebase[K, V],
    measure: AnySimFunc[V, Float],
    max_cases: int,
    executor: Executor | None = None,
    chunksize: int | None = None,
) -> Matrix[K]:
    """The similarities of all ordered pairs of distinct cases, scored in one batched call.

    The cost is n * (n - 1) similarity computations for n cases, so `max_cases` bounds it.

    Args:
        casebase: The cases.
        measure: Similarity of a case to another case as query.
        max_cases: Largest number of cases.
        executor: Runs chunks of pairs in parallel, in the current thread if None.
        chunksize: Number of pairs per call of the measure,
            all pairs in one call without an executor and one chunk per available CPU with one if None.

    Raises:
        cbrkit.errors.Refused: If there are more cases than `max_cases`.

    Examples:
        >>> from cbrkit.sim.numbers import Linear
        >>> pairwise({0: 1.0, 1: 2.0}, Linear(max_distance=4), 10)
        {0: {1: 0.75}, 1: {0: 0.75}}
    """
    if len(casebase) > max_cases:
        raise Refused(
            f"The case base has {len(casebase)} cases, more than the {max_cases} that are compared pairwise"
        )

    keys = sorted_keys(casebase)
    scored = score_pairs(
        [
            ({case: casebase[case] for case in keys if case != query}, casebase[query])
            for query in keys
        ],
        measure,
        executor,
        chunksize,
    )

    return {
        query: {case: float(value) for case, value in similarities.items()}
        for query, similarities in zip(keys, scored, strict=True)
    }


@dataclass(slots=True, frozen=True)
class Competence[K]:
    """Coverage and reachability of every case, estimated from the case base itself.

    A case solves another one if it is at least `competence` similar to it as query
    and among its `neighbors` most similar cases,
    which stands for `Adaptable(c, c')` of Smyth and Keane (1995).
    Every case solves itself.

    Args:
        coverage: The cases that a case solves, `Coverage(c)` (Definition 1).
        reachability: The cases that solve a case, `Reachable(c)` (Definition 2).
    """

    coverage: Mapping[K, frozenset[K]]
    reachability: Mapping[K, frozenset[K]]

    @classmethod
    def build(
        cls, matrix: Matrix[K], competence: float, neighbors: int | None = None
    ) -> Competence[K]:
        """Estimates the competence from pairwise similarities.

        Args:
            matrix: Similarities of the cases, see `pairwise`.
            competence: Similarity from which a case solves another one.
            neighbors: Only the most similar cases of a query solve it, all if None.
        """
        reachability: dict[K, frozenset[K]] = {}
        coverage: dict[K, set[K]] = {key: {key} for key in matrix}

        for query, similarities in matrix.items():
            ranked = sorted(
                sorted_keys(similarities), key=lambda case: -similarities[case]
            )
            solvers = {
                case for case in ranked[:neighbors] if similarities[case] >= competence
            }
            reachability[query] = frozenset({query, *solvers})

            for case in solvers:
                coverage[case].add(query)

        return cls(
            {key: frozenset(cases) for key, cases in coverage.items()}, reachability
        )

    def category(self, key: K) -> Category:
        """The competence category of a case.

        A case belongs to the first of these categories whose definition it meets:
        pivotal if no other case solves it (Definition 3),
        auxiliary if another case that solves it covers a proper superset (Definition 4),
        support if another case that solves it covers a subset (Definition 6),
        and spanning otherwise, since every other case meets Definition 5.

        Args:
            key: Key of the case.
        """
        coverage = self.coverage[key]
        others = self.reachability[key] - {key}

        if not others:
            return "pivotal"

        if any(coverage < self.coverage[other] for other in others):
            return "auxiliary"

        if any(self.coverage[other] <= coverage for other in others):
            return "support"

        return "spanning"

    def footprint(
        self,
        remove: Collection[Category],
        uses: Mapping[K, int] | None = None,
        keep: Collection[K] = (),
    ) -> list[K]:
        """The cases to delete by footprint-utility deletion without losing competence.

        Candidates of the given categories are deleted in the order of Algorithm 3,
        auxiliary before support before spanning cases,
        and within a category the least used first (footprint-utility deletion, Section 3.4),
        then those with the largest reachability and the smallest coverage.
        A candidate is only deleted while every case it solved,
        itself and those deleted before included, is still solved by a remaining case,
        so the deletions keep the competence of the whole case base.

        Args:
            remove: Categories whose cases may be deleted, pivotal cases never are.
            uses: How often every case was used, unknown cases count as unused.
            keep: Cases that must not be deleted.
        """
        uses = uses or {}
        categories = {key: self.category(key) for key in self.coverage}
        candidates = sorted(
            (
                key
                for key in sorted_keys(self.coverage)
                if categories[key] in remove
                and categories[key] != "pivotal"
                and key not in keep
            ),
            key=lambda key: (
                CATEGORIES.index(categories[key]),
                uses.get(key, 0),
                -len(self.reachability[key]),
                len(self.coverage[key]),
            ),
        )
        deleted: set[K] = set()

        for key in candidates:
            gone = deleted | {key}

            if all(
                self.reachability[solved] - gone for solved in self.coverage[key] & gone
            ):
                deleted.add(key)

        return [key for key in candidates if key in deleted]

    def reviews(
        self,
        uses: Mapping[K, int] | None = None,
        consistency: Mapping[K, float] | None = None,
    ) -> list[Review[K]]:
        """The assessment of every case in key order.

        Args:
            uses: How often every case was used, if the host counts it.
            consistency: The consistency of the cases, see `consistency`.
        """
        return [
            Review(
                key=key,
                category=self.category(key),
                coverage=len(self.coverage[key]),
                reachability=len(self.reachability[key]),
                uses=None if uses is None else uses.get(key, 0),
                consistency=(consistency or {}).get(key),
            )
            for key in sorted_keys(self.coverage)
        ]

    def consistency[V](
        self,
        casebase: Casebase[K, V],
        solution_measure: AnySimFunc[V, Float],
        executor: Executor | None = None,
        chunksize: int | None = None,
    ) -> dict[K, float]:
        """The mean similarity of the solution of every case to those of the other cases that solve it.

        Cases that solve a case have similar problems, so dissimilar solutions point to noise,
        as in the consistency measure of Reinartz, Iglezakis and Roth-Berghofer (2001).
        Cases that no other case solves are left out.
        All pairs are scored in one batched call.

        Args:
            casebase: The cases.
            solution_measure: Similarity of the solution of a case to that of another case as query.
            executor: Runs chunks of pairs in parallel, in the current thread if None.
            chunksize: Number of pairs per call of the measure,
                all pairs in one call without an executor and one chunk per available CPU with one if None.

        Examples:
            >>> from cbrkit.sim.numbers import Linear
            >>> competence = Competence.build({0: {1: 0.9}, 1: {0: 0.9}}, 0.5)
            >>> competence.consistency({0: 1.0, 1: 3.0}, Linear(max_distance=4))
            {0: 0.5, 1: 0.5}
        """
        solved = [
            key
            for key in sorted_keys(self.reachability)
            if len(self.reachability[key]) > 1
        ]
        scored = score_pairs(
            [
                (
                    {
                        other: casebase[other]
                        for other in self.reachability[key] - {key}
                    },
                    casebase[key],
                )
                for key in solved
            ],
            solution_measure,
            executor,
            chunksize,
        )

        return {
            key: sum(float(value) for value in scores.values()) / len(scores)
            for key, scores in zip(solved, scored, strict=True)
        }


def duplicates[K](matrix: Matrix[K], duplicate: float) -> list[list[K]]:
    """Groups of duplicate cases, which are at least `duplicate` similar in both directions.

    Each group starts with its first case in key order and holds the later cases that duplicate it,
    so every case is in at most one group.

    Examples:
        >>> duplicates({0: {1: 0.9, 2: 0.1}, 1: {0: 0.95, 2: 0.2}, 2: {0: 0.1, 1: 0.2}}, 0.9)
        [[0, 1]]
    """
    keys = sorted_keys(matrix)
    grouped: set[K] = set()
    groups: list[list[K]] = []

    for index, first in enumerate(keys):
        if first in grouped:
            continue

        group = [
            first,
            *(
                other
                for other in keys[index + 1 :]
                if other not in grouped
                and matrix[first].get(other, 0.0) >= duplicate
                and matrix[other].get(first, 0.0) >= duplicate
            ),
        ]

        if len(group) > 1:
            grouped.update(group)
            groups.append(group)

    return groups
