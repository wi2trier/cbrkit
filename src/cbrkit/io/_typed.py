"""A storage of JSON cases as a storage of typed cases."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, override

from pydantic import TypeAdapter

from ..typing import Casebase, Storage

__all__ = ["TypedStorage"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TypedStorage[K, V](Storage[K, V]):
    """A storage of JSON cases as a storage of typed cases, such as a `FileStorage` of pydantic models.

    Args:
        storage: The storage of the JSON cases.
        adapter: Validates and dumps the cases, such as `TypeAdapter(Case[str, str])`.

    Examples:
        >>> import asyncio
        >>> from datetime import date
        >>> from cbrkit.indexable import InMemory
        >>> storage = TypedStorage(storage=InMemory(), adapter=TypeAdapter(date))
        >>> asyncio.run(storage.patch_index(add=[date(2024, 1, 2)]))
        [0]
        >>> dict(asyncio.run(storage.storage.get_index())), dict(asyncio.run(storage.get_index()))
        ({0: '2024-01-02'}, {0: datetime.date(2024, 1, 2)})
    """

    storage: Storage[K, Any]
    adapter: TypeAdapter[V]

    def _dump(self, cases: Casebase[K, V]) -> dict[K, Any]:
        return {
            key: self.adapter.dump_python(case, mode="json")
            for key, case in cases.items()
        }

    @override
    async def get_index(self) -> dict[K, V]:
        """The current cases."""
        return {
            key: self.adapter.validate_python(case)
            for key, case in (await self.storage.get_index()).items()
        }

    @override
    async def has_index(self) -> bool:
        """Whether the cases have been created."""
        return await self.storage.has_index()

    @override
    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Replaces all cases."""
        await self.storage.put_index(self._dump(data))

    @override
    async def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
        add: Sequence[V] = (),
        *,
        insert: Casebase[K, V] | None = None,
        expected: Casebase[K, V] | None = None,
    ) -> Sequence[K]:
        """Inserts, replaces, deletes, and adds cases as one mutation, see `cbrkit.typing.Storage`."""
        return await self.storage.patch_index(
            None if upsert is None else self._dump(upsert),
            delete,
            [self.adapter.dump_python(case, mode="json") for case in add],
            insert=None if insert is None else self._dump(insert),
            expected=None if expected is None else self._dump(expected),
        )
