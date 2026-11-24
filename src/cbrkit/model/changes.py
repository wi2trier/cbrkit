"""Changes that retain, maintenance, and applications make to a storage, see `Changes`.

Planning is pure: a `Planner` such as `cbrkit.maintain.Maintainer` takes the stored cases and the inputs of a run and returns `Changes`.
The effects are `Changes.apply`, which writes them as one mutation of a storage,
and `replan`, which reads the storage, plans, and applies, planning again on a conflict.
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, SkipValidation
from pydantic.json_schema import SkipJsonSchema

from .._util import retry_conflicts
from ..helpers import frozendict, to_thread
from ..schema import Model
from ..typing import Casebase, Storage

__all__ = [
    "Add",
    "Category",
    "Change",
    "Changes",
    "Planner",
    "Remove",
    "Review",
    "Rule",
    "Update",
    "replan",
]

type Rule = Literal["solution", "duplicate", "redundancy", "competence", "consistency"]
"""The policy behind a change: storing a solution, merging duplicates, removing redundant duplicates,
removing cases the competence does not need, or removing cases inconsistent with the cases that solve them."""

type Category = Literal["pivotal", "spanning", "support", "auxiliary"]
"""Competence category of a case after Smyth and Keane (1995), see `cbrkit.maintain`."""


class Change[K](Model):
    """A change of a storage with its reason.

    Args:
        rule: The policy of the CBR cycle behind the change, None for a write of an application, whose reason says why.
        reason: Why the change is made, for people.
        score: The similarity that triggered the change, such as that of a duplicate.
        related: Keys of the cases that the change refers to, such as the duplicate of a solution.
    """

    rule: Rule | None = None
    reason: str
    score: float | None = None
    related: Sequence[K] = ()


class Add[K, V](Change[K]):
    """Adds a case, under its key if it has one and otherwise under a key that the storage allocates.

    A key of a stored case refuses the changes with `cbrkit.errors.Conflict`.

    Args:
        key: Key of the new case, which applying the changes fills in if None.
        value: The new case.
    """

    op: Literal["add"] = "add"
    key: K | None = None
    value: V


class Update[K, V](Change[K]):
    """Replaces a stored case.

    Args:
        key: Key of the replaced case.
        value: The new value of the case.
    """

    op: Literal["update"] = "update"
    key: K
    value: V


class Remove[K](Change[K]):
    """Removes a stored case.

    Args:
        key: Key of the removed case.
    """

    op: Literal["remove"] = "remove"
    key: K


class Review[K](Model):
    """The review of a stored case: its value for the competence of the case base and its consistency.

    Args:
        key: Key of the case.
        category: Competence category after Smyth and Keane (1995).
        coverage: Number of cases that the case solves, itself included.
        reachability: Number of cases that solve the case, itself included.
        uses: How often the host used the case, None if unknown.
        consistency: Mean similarity of the solution of the case to those of the cases that solve it,
            None if not assessed or no other case solves it.
    """

    key: K
    category: Category
    coverage: int
    reachability: int
    uses: int | None = None
    consistency: float | None = None


class Changes[K, V](Model):
    """The changes of one run, such as of retain, maintenance, or a write of an application, applied as one mutation of the storage.

    Args:
        items: Additions, updates, and removals, each with its reason.
            A merge of duplicates is an update of the case that keeps the merged value
            and a removal of every other duplicate.
        reviews: Review of the stored cases that the changes are based on,
            empty unless a maintenance run assesses their competence.
        applied: Whether the changes were made, which a dry run leaves out.
        expected: The stored cases that the changes were planned on, which the storage must still hold
            when the changes are applied, None if the changes depend on no stored case.
            It is never serialized.
    """

    items: Sequence[
        Annotated[
            Add[K, V] | Update[K, V] | Remove[K],
            Field(discriminator="op"),
        ]
    ] = ()
    reviews: Sequence[Review[K]] = ()
    applied: bool = False
    expected: SkipJsonSchema[Annotated[Casebase[K, V] | None, SkipValidation]] = Field(
        default=None, exclude=True, repr=False
    )

    def patch(self) -> tuple[dict[K, V], list[K], list[V], dict[K, V]]:
        """The mutation of the changes, as `cbrkit.typing.Storage.patch_index` takes it.

        Returns:
            The cases to upsert by key, the keys to delete, the cases to add under new keys,
            and the cases to insert under their keys.

        Raises:
            ValueError: If the changes change a case twice.
        """
        keys = [change.key for change in self.items if change.key is not None]

        if len(set(keys)) != len(keys):
            raise ValueError("The changes change a case twice")

        upsert: dict[K, V] = {}
        delete: list[K] = []
        add: list[V] = []
        insert: dict[K, V] = {}

        for change in self.items:
            match change:
                case Add(key=None):
                    add.append(change.value)
                case Add(key=key) if key is not None:
                    insert[key] = change.value
                case Update():
                    upsert[change.key] = change.value
                case Remove():
                    delete.append(change.key)

        return upsert, delete, add, insert

    async def apply(self, storage: Storage[K, V]) -> Changes[K, V]:
        """Applies the changes to a storage as one mutation, if it still holds the expected cases.

        Args:
            storage: The storage of the cases.

        Returns:
            The applied changes, with the keys of the added cases.

        Raises:
            ValueError: If the changes are applied already or change a case twice.
            cbrkit.errors.Conflict: If the storage holds other cases than the expected ones
                or a case under the key of an addition, in which case nothing is written.

        Examples:
            >>> import asyncio
            >>> from cbrkit.indexable import InMemory
            >>> storage = InMemory({0: "a", 1: "b"})
            >>> changes = Changes[int, str](items=[
            ...     Add(value="c", rule="solution", reason="new"),
            ...     Remove(key=0, rule="competence", reason="auxiliary"),
            ... ], expected={0: "a", 1: "b"})
            >>> asyncio.run(changes.apply(storage)).items[0].key
            2
            >>> dict(asyncio.run(storage.get_index()))
            {1: 'b', 2: 'c'}
            >>> asyncio.run(changes.apply(storage))
            Traceback (most recent call last):
            ...
            cbrkit.errors.Conflict: The cases changed since the mutation was planned
        """
        if self.applied:
            raise ValueError("The changes are applied already")

        upsert, delete, add, insert = self.patch()
        keys = iter(
            await storage.patch_index(
                frozendict(upsert),
                delete,
                add,
                insert=frozendict(insert),
                expected=self.expected,
            )
        )
        items: tuple[Any, ...] = tuple(
            change.model_copy(update={"key": next(keys)})
            if isinstance(change, Add) and change.key is None
            else change
            for change in self.items
        )

        return self.model_copy(update={"items": items, "applied": True})


class Planner[K, V, **P = ...](Protocol):
    """Plans the changes of a run on the stored cases purely, such as `cbrkit.retain.Retainer` and `cbrkit.maintain.Maintainer`.

    `replan` applies a plan with the inputs of the run, such as the solutions of a retain run,
    and hosts tell by `destructive` whether a run may replace or remove stored cases.
    """

    @property
    def destructive(self) -> bool:
        """Whether a run may replace or remove stored cases."""
        ...

    def plan(
        self, stored: Casebase[K, V], /, *args: P.args, **kwargs: P.kwargs
    ) -> Changes[K, V]:
        """Plans the changes of a run on the stored cases without applying them."""
        ...


async def replan[K, V, **P](
    planner: Planner[K, V, P],
    storage: Storage[K, V],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Changes[K, V]:
    """Reads the stored cases, plans a run on them, and applies it, planning again while the storage refuses its changes.

    A storage refuses changes with `cbrkit.errors.Conflict`
    if its cases changed since the changes were planned, such as by a concurrent run.
    The planner is pure and runs in a worker thread with `cbrkit.helpers.to_thread`,
    so its measures may wait for embedders and language models.

    Args:
        planner: Plans the run from the stored cases, such as a `cbrkit.maintain.Maintainer`.
        storage: The storage of the cases.
        *args: Further positional inputs of `Planner.plan`, such as the solutions of a `cbrkit.retain.Retainer`.
        **kwargs: Further keyword inputs of `Planner.plan`, such as the uses of a `cbrkit.maintain.Maintainer`.

    Returns:
        The applied changes, with the keys of the added cases.

    Raises:
        cbrkit.errors.Conflict: If the stored cases changed before every attempt to apply the changes.
    """

    async def attempt() -> Changes[K, V]:
        stored = await storage.get_index()
        changes = await to_thread(planner.plan, stored, *args, **kwargs)

        return await changes.apply(storage)

    return await retry_conflicts(attempt)
