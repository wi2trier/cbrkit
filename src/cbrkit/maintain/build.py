from collections.abc import Collection, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from .._phase import required
from .._util import sorted_keys
from ..adapt import Adaptation
from ..helpers import unbatchify
from ..model import Category, Changes, Remove, Update, problem, solution
from ..schema import PythonOnly, serializable
from ..sim import Measure
from ..typing import Casebase, Float
from .review import Competence, Matrix, duplicates, pairwise

__all__ = ["Maintainer"]


def _without[K](matrix: Matrix[K], removed: Collection[K]) -> Matrix[K]:
    return {
        query: {case: value for case, value in row.items() if case not in removed}
        for query, row in matrix.items()
        if query not in removed
    }


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Maintainer[K, V]:
    """Plans the maintenance of a case base, outside the CBR cycle and without a query.

    Maintenance is its own activity beside the solving cycle
    (Leake and Wilson 1998, Wilson and Leake 2001, Göker and Roth-Berghofer 1999):
    a run reviews the stored cases and restores the case base
    (Reinartz, Iglezakis and Roth-Berghofer 2001).
    The review assesses the competence of every case after Smyth and Keane (1995),
    its coverage, reachability and category, and the consistency of its solution with those of the cases that solve it.
    The restore plans changes with their reasons:
    merging duplicates into the first of them (an update and removals) or removing them as redundant,
    removing cases inconsistent with the cases that solve them,
    and removing cases that the competence of the case base does not need by footprint-utility deletion.
    When to maintain, periodically, on a condition such as the size of the case base, or ad hoc, is up to the caller.
    It is a `cbrkit.model.Planner`, whose `plan` is pure and reviews the stored cases it is given,
    `cbrkit.model.Changes.apply` writes the changes as one mutation of a storage,
    and `cbrkit.model.replan` reads the storage, plans, writes, and plans again on a conflict.
    It is the `maintain` section of a `cbrkit.spec.Config`, which fills in a missing measure.

    Args:
        measure: Similarity of the problem of a case to that of another case as query, needed to compare cases,
            the measure of the configuration in a `cbrkit.spec.Config` if None.
        duplicate: Similarity in both directions from which a case duplicates another one, duplicates are kept if None.
        merge: Combines the first of a group of duplicates (the case) with a later one (the query),
            any adaptation of `cbrkit.adapt`, the later duplicates are removed as redundant if None.
        competence: Similarity from which a case solves another one,
            which enables the review of the competence of the cases.
        neighbors: Only the most similar cases of a case can solve it, as in a retrieval with this limit.
        remove: Competence categories whose cases are removed as long as every stored case stays solved.
        solution_measure: Similarity of the solution of a case to that of another case,
            which enables the review of the consistency of the cases.
        consistency: Minimal consistency of a case, inconsistent cases are kept if None.
        max_cases: Largest case base whose cases are compared pairwise.
        executor: Runs chunks of pairs in parallel, such as a `ProcessPoolExecutor`,
            `ThreadPoolExecutor` or `InterpreterPoolExecutor` that the caller owns,
            in the current thread if None.
        chunksize: Number of pairs per call of the measures, all pairs in one call
            without an executor and one chunk per available CPU with one if None.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.model import replan
        >>> from cbrkit.sim.numbers import Linear
        >>> storage = InMemory({0: 1.0, 1: 1.05, 2: 9.0})
        >>> maintainer = Maintainer(measure=Linear(max_distance=10), duplicate=0.99, competence=0.5)
        >>> changes = asyncio.run(replan(maintainer, storage))
        >>> [(change.op, change.key, change.rule) for change in changes.items]
        [('remove', 1, 'redundancy')]
        >>> dict(asyncio.run(storage.get_index()))
        {0: 1.0, 2: 9.0}
    """

    measure: Measure[Any, Float] | None = None
    duplicate: float | None = None
    merge: Adaptation[V] | None = None
    competence: float | None = None
    neighbors: int | None = None
    remove: frozenset[Category] = frozenset()
    solution_measure: Measure[Any, Float] | None = None
    consistency: float | None = None
    max_cases: int = 1000
    executor: PythonOnly[Executor | None] = None
    chunksize: PythonOnly[int | None] = None

    def __post_init__(self) -> None:
        if (
            self.remove or self.solution_measure is not None
        ) and self.competence is None:
            raise ValueError(
                "Removing cases by their category and reviewing their consistency need a competence threshold"
            )

        if self.consistency is not None and self.solution_measure is None:
            raise ValueError("A minimal consistency needs a solution measure")

        if "pivotal" in self.remove:
            raise ValueError("Pivotal cases are never removed")

    @property
    def destructive(self) -> bool:
        """Whether a run may replace or remove stored cases."""
        return (
            self.duplicate is not None
            or bool(self.remove)
            or self.consistency is not None
        )

    def plan(
        self, stored: Casebase[K, V], uses: Mapping[K, int] | None = None
    ) -> Changes[K, V]:
        """Reviews the stored cases and plans the changes without applying them.

        Args:
            stored: The stored cases, which the changes expect the storage to hold.
            uses: How often the host used every case, which orders the removals by competence.
        """
        matrix = (
            pairwise(
                {key: problem(case) for key, case in stored.items()},
                required(self.measure, "measure"),
                self.max_cases,
                self.executor,
                self.chunksize,
            )
            if stored and (self.duplicate is not None or self.competence is not None)
            else {}
        )
        updates, removals = self._duplicates(stored, matrix)

        if self.competence is None or not matrix:
            return Changes(
                items=[*updates, *removals],
                expected=stored if self.duplicate is not None else None,
            )

        reviewed = Competence.build(matrix, self.competence, self.neighbors)
        consistency = (
            reviewed.consistency(
                {key: solution(case) for key, case in stored.items()},
                self.solution_measure,
                self.executor,
                self.chunksize,
            )
            if self.solution_measure is not None
            else {}
        )
        kept = {change.key for change in updates}
        removals += self._inconsistent(
            reviewed, consistency, kept | {change.key for change in removals}
        )
        removals += self._surplus(
            matrix, self.competence, {change.key for change in removals}, uses, kept
        )

        return Changes(
            items=[*updates, *removals],
            reviews=reviewed.reviews(uses, consistency),
            expected=stored,
        )

    def _duplicates(
        self, stored: Casebase[K, V], matrix: Matrix[K]
    ) -> tuple[list[Update[K, V]], list[Remove[K]]]:
        """Merges the duplicates of every case into it if a merge is set, and removes them."""
        merge = unbatchify(self.merge) if self.merge is not None else None
        updates: list[Update[K, V]] = []
        removals: list[Remove[K]] = []

        for first, *others in (
            duplicates(matrix, self.duplicate) if self.duplicate is not None else []
        ):
            similarities = {
                other: min(matrix[first][other], matrix[other][first])
                for other in others
            }

            if merge is not None:
                case = stored[first]

                for other in others:
                    case = merge(case, stored[other])

                updates.append(
                    Update(
                        key=first,
                        value=case,
                        rule="duplicate",
                        reason=f"Cases {others!r} duplicate case {first!r}"
                        f" with a similarity of at least {self.duplicate} in both directions,"
                        " so they are merged into it.",
                        score=min(similarities.values()),
                        related=others,
                    )
                )

            removals.extend(
                Remove(
                    key=other,
                    rule="duplicate" if merge is not None else "redundancy",
                    reason=f"Case {other!r} duplicates case {first!r} with a similarity of {value:.3f},"
                    f" at least {self.duplicate} in both directions, so it is "
                    + ("merged into it." if merge is not None else "redundant."),
                    score=value,
                    related=[first],
                )
                for other, value in similarities.items()
            )

        return updates, removals

    def _inconsistent(
        self,
        reviewed: Competence[K],
        consistency: Mapping[K, float],
        excluded: Collection[K],
    ) -> list[Remove[K]]:
        """Removes the cases whose solvers disagree with them, except the excluded ones."""
        if self.consistency is None:
            return []

        removals: list[Remove[K]] = []

        for key in sorted_keys(consistency):
            if key not in excluded and consistency[key] < self.consistency:
                solvers = sorted_keys(reviewed.reachability[key] - {key})
                removals.append(
                    Remove(
                        key=key,
                        rule="consistency",
                        reason=f"The case is solved by cases {solvers!r}, whose solutions are"
                        f" {consistency[key]:.3f} similar to its own on average, below {self.consistency}.",
                        score=consistency[key],
                        related=solvers,
                    )
                )

        return removals

    def _surplus(
        self,
        matrix: Matrix[K],
        competence: float,
        removed: set[K],
        uses: Mapping[K, int] | None,
        kept: Collection[K],
    ) -> list[Remove[K]]:
        """Removes the cases of the categories to remove, which stay solved without them."""
        survivors = Competence.build(
            _without(matrix, removed), competence, self.neighbors
        )
        removals: list[Remove[K]] = []

        for key in survivors.footprint(self.remove, uses, kept):
            category = survivors.category(key)
            solvers = sorted_keys(survivors.reachability[key] - {key})
            removals.append(
                Remove(
                    key=key,
                    rule="competence",
                    reason=f"The {category} case is solved by cases {solvers!r},"
                    " and every case it solves stays solved without it.",
                    related=solvers,
                )
            )

        return removals
