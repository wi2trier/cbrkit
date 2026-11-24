from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from .._phase import required, score_pairs
from .._util import retry_conflicts, sorted_keys
from ..adapt import Adaptation
from ..helpers import to_thread, unbatchify
from ..model import Add, Changes, ChangeSim, Update, problem
from ..schema import PythonOnly, serializable
from ..sim import Measure
from ..typing import Batches, Casebase, Float, Results, Storage

__all__ = ["Retainer", "changed"]


def changed[K, V](solutions: Casebase[K, V], cases: Casebase[Any, V]) -> dict[K, V]:
    """The solutions that differ from the case under their key, leaving out cases that reuse passed on unchanged.

    Examples:
        >>> changed({0: "a", 1: "B", 2: "c"}, {0: "a", 1: "b"})
        {1: 'B', 2: 'c'}
    """
    return {
        key: value
        for key, value in solutions.items()
        if key not in cases or cases[key] != value
    }


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Retainer[K, V]:
    """Stores the solutions that revise accepted, the retain phase of the CBR cycle.

    Revise assesses and retain only stores, so it follows a `cbrkit.revise.Reviser` in `cbrkit.cycle`.
    Every case that reaches it is stored, and retrieved cases solve other problems,
    so it follows a `cbrkit.reuse.Reuser` that keeps the solutions to store, such as the best one,
    as `cbrkit.system.System.cycle` does.
    A run adds every solution of every query under a new key,
    unless it duplicates a stored case by the similarity of their problems, which a `merge` combines it with,
    the narrow retention of Aamodt and Plaza (1994) that integrates one solved problem.
    The changes of a run are applied as one mutation of the storage,
    planned again if the storage changed in between, see `cbrkit.model.replan`.
    A solution equal to the stored case under its key is that case passed on unchanged,
    such as by a null adaptation of reuse, so it is not stored again, see `changed`.
    The result holds the cases as stored under their keys with the `cbrkit.model.ChangeSim` that stored them,
    a solution that changed nothing is left out, and `cbrkit.model.QueryResult.changes` collects the changes of a result.
    Maintaining the whole case base is a separate activity, see `cbrkit.maintain`.
    It is the `retain` section of a `cbrkit.spec.Config`, which fills in a missing measure.

    Args:
        storage: The storage of the cases, which a `cbrkit.system.System` fills in.
        measure: Similarity of the problem of a stored case to that of a solution, needed for duplicates,
            the measure of the configuration in a `cbrkit.spec.Config` if None.
        duplicate: Similarity from which a solution duplicates a stored case, duplicates are stored if None.
        merge: Combines a stored case (the case) with a solution that duplicates it (the query),
            any adaptation of `cbrkit.adapt`, such as `cbrkit.adapt.generic.Null("query")` to replace it.
            A duplicate solution changes nothing if None.
        executor: Runs chunks of pairs in parallel, such as a `ProcessPoolExecutor`,
            `ThreadPoolExecutor` or `InterpreterPoolExecutor` that the caller owns,
            in the current thread if None.
        chunksize: Number of pairs per call of the measure, all pairs in one call
            without an executor and one chunk per available CPU with one if None.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.sim.generic import Equality
        >>> storage = InMemory({0: "a"})
        >>> retainer = Retainer(storage=storage, measure=Equality(), duplicate=1.0)
        >>> asyncio.run(retainer([({0: "a", 1: "b"}, "b")]))
        [({1: 'b'}, {1: ChangeSim(value=1.0, op='add', rule='solution', reason='The solution duplicates no stored case.', score=None, related=())})]
        >>> dict(asyncio.run(storage.get_index()))
        {0: 'a', 1: 'b'}
    """

    storage: PythonOnly[Storage[K, V] | None] = None
    measure: Measure[Any, Float] | None = None
    duplicate: float | None = None
    merge: Adaptation[V] | None = None
    executor: PythonOnly[Executor | None] = None
    chunksize: PythonOnly[int | None] = None

    @property
    def destructive(self) -> bool:
        """Whether a run may replace stored cases."""
        return self.duplicate is not None and self.merge is not None

    def plan(self, stored: Casebase[K, V], batches: Batches[Any, V]) -> Changes[K, V]:
        """Plans the changes of a run on the stored cases without applying them, such as for a dry run.

        Args:
            stored: The stored cases, which the changes expect the storage to hold.
            batches: The accepted solutions with their queries, such as the result of a `cbrkit.revise.Reviser`.
        """
        return self._plan(
            stored, [case for casebase, _ in batches for case in casebase.values()]
        )[0]

    async def __call__(
        self, batches: Batches[Any, V], /
    ) -> Results[K, V, ChangeSim[K]]:
        storage = required(self.storage, "storage")

        # like `cbrkit.model.replan`, but keeping which change stores every solution
        async def attempt() -> tuple[
            Changes[K, V], list[int | None], list[dict[Any, V]]
        ]:
            stored = await storage.get_index()
            kept = [changed(casebase, stored) for casebase, _ in batches]
            solutions = [case for casebase in kept for case in casebase.values()]
            changes, owners = await to_thread(self._plan, stored, solutions)

            return await changes.apply(storage), owners, kept

        changes, owners, kept = await retry_conflicts(attempt)
        owned = iter(owners)
        results: list[tuple[dict[K, V], dict[K, ChangeSim[K]]]] = []

        for casebase in kept:
            cases: dict[K, V] = {}
            similarities: dict[K, ChangeSim[K]] = {}

            for index in (next(owned) for _ in casebase):
                if index is None:
                    continue

                change = changes.items[index]

                if isinstance(change, Add | Update) and change.key is not None:
                    cases[change.key] = change.value
                    similarities[change.key] = ChangeSim.of(change)

            results.append((cases, similarities))

        return results

    def _plan(
        self, stored: Casebase[K, V], solutions: Sequence[V]
    ) -> tuple[Changes[K, V], list[int | None]]:
        """The changes of a run and, for every solution, the index of the change that stores it."""
        if self.duplicate is None:
            return Changes(
                items=[
                    Add(value=case, rule="solution", reason="The solution is accepted.")
                    for case in solutions
                ]
            ), list(range(len(solutions)))

        merge = unbatchify(self.merge) if self.merge is not None else None
        changes: list[Add[K, V] | Update[K, V]] = []
        owners: list[int | None] = []
        updates: dict[K, int] = {}
        problems = {key: problem(case) for key, case in stored.items()}
        scored = score_pairs(
            [(problems, problem(case)) for case in solutions],
            required(self.measure, "measure"),
            self.executor,
            self.chunksize,
        )

        for case, scores in zip(solutions, scored, strict=True):
            key = max(
                sorted_keys(scores), key=lambda key: float(scores[key]), default=None
            )

            if key is None or float(scores[key]) < self.duplicate:
                owners.append(len(changes))
                changes.append(
                    Add(
                        value=case,
                        rule="solution",
                        reason="The solution duplicates no stored case.",
                    )
                )
                continue

            if merge is None:
                owners.append(None)
                continue

            value = float(scores[key])
            duplication = (
                f"The solution duplicates case {key!r} with a similarity of {value:.3f},"
                f" at least {self.duplicate}"
            )

            if key in updates:
                current = changes[updates[key]]
                changes[updates[key]] = current.model_copy(
                    update={
                        "value": merge(current.value, case),
                        "reason": f"{current.reason} {duplication}, so it is merged as well.",
                    }
                )
            else:
                updates[key] = len(changes)
                changes.append(
                    Update(
                        key=key,
                        value=merge(stored[key], case),
                        rule="duplicate",
                        reason=f"{duplication}, so it is merged into it.",
                        score=value,
                        related=[key],
                    )
                )

            owners.append(updates[key])

        return Changes(items=changes, expected=stored), owners
