from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..typing import Casebase, Float, SimMap

__all__ = [
    "IndexableRetrieverFunc",
]


class IndexableRetrieverFunc[K, V, S: Float](ABC):
    """Retrieves similar cases from casebases for given queries and supports indexing."""

    casebase: Casebase[K, V] | None

    def resolve_batches(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[tuple[Casebase[K, V], V]]:
        """Resolves empty casebases using the indexed casebase."""
        if self.casebase is None:
            return list(batches)

        return [
            (self.casebase if len(casebase) == 0 else casebase, query)
            for casebase, query in batches
        ]

    @abstractmethod
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[SimMap[K, S]]: ...

    @abstractmethod
    def index(
        self,
        data: Casebase[K, V],
        /,
        prune: bool = True,
    ) -> None: ...
