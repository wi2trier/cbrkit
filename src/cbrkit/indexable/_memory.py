"""Cases kept in memory."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import InitVar, dataclass, field
from threading import Lock
from typing import override

from .._util import next_keys, patched, plan_patch
from ..helpers import frozendict
from ..typing import Casebase, Storage

__all__ = ["InMemory"]


@dataclass(slots=True)
class _State[K, V]:
    """The current snapshot of an `InMemory` storage and the lock of its writers."""

    cases: frozendict[K, V]
    lock: Lock = field(default_factory=Lock)


@dataclass(slots=True, frozen=True)
class InMemory[K, V](Storage[K, V]):
    """Cases kept in memory, the simplest storage.

    Every mutation replaces the immutable snapshot that `get_index` returns,
    so readers never see a partial mutation and need no lock.
    A mutation never awaits, so it is atomic on the event loop,
    and its lock guards it against writers on other threads.

    Args:
        casebase: The initial cases.
        new_keys: The distinct keys of a number of new cases outside the keys taken so far, `cbrkit.helpers.next_keys` by default.

    Examples:
        >>> import asyncio
        >>> storage = InMemory({0: "a"})
        >>> snapshot = asyncio.run(storage.get_index())
        >>> asyncio.run(storage.patch_index(upsert={1: "b"}, delete=[0]))
        []
        >>> dict(snapshot), dict(asyncio.run(storage.get_index()))
        ({0: 'a'}, {1: 'b'})
        >>> asyncio.run(storage.patch_index(add=["c", "d"], insert={5: "e"}))
        [6, 7]
        >>> asyncio.run(storage.patch_index(insert={5: "f"}))
        Traceback (most recent call last):
        ...
        cbrkit.errors.Conflict: 5 exists already
    """

    casebase: InitVar[Casebase[K, V] | None] = None
    new_keys: Callable[[Collection[K], int], list[K]] = field(
        default=next_keys, kw_only=True
    )
    _state: _State[K, V] = field(init=False, repr=False, compare=False)

    def __post_init__(self, casebase: Casebase[K, V] | None) -> None:
        object.__setattr__(self, "_state", _State(frozendict(casebase or {})))

    @override
    async def get_index(self) -> Casebase[K, V]:
        """The current cases."""
        return self._state.cases

    @override
    async def has_index(self) -> bool:
        """Whether there are cases."""
        return bool(self._state.cases)

    @override
    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Replace all cases."""
        with self._state.lock:
            self._state.cases = frozendict(data)

    @override
    async def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
        add: Sequence[V] = (),
        *,
        insert: Casebase[K, V] | None = None,
        expected: Casebase[K, V] | None = None,
    ) -> list[K]:
        """Insert, replace, delete, and add cases as one mutation."""
        if not upsert and not delete and not add and not insert:
            return []

        with self._state.lock:
            current = self._state.cases
            written, deleted, new = plan_patch(
                current, upsert, delete, add, insert, expected, self.new_keys
            )
            self._state.cases = frozendict(patched(current, written, deleted))

        return new
