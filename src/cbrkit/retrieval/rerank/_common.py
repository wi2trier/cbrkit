"""Shared scaffolding for rerank-model retrievers."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import override

from ...helpers import run_coroutine
from ...typing import Casebase, RetrieverFunc


@dataclass(slots=True, frozen=True)
class RerankFunc[K](RetrieverFunc[K, str, float], ABC):
    """Base for rerank-model retrievers that score documents per query.

    Subclasses implement the single async API call :meth:`_rerank`; the
    batching and index-to-key mapping scaffolding is shared here.
    """

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], dict[K, float]]]:
        return run_coroutine(self._retrieve(batches))

    async def _retrieve(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], dict[K, float]]]:
        return await asyncio.gather(
            *(self._retrieve_single(query, casebase) for casebase, query in batches)
        )

    async def _retrieve_single(
        self,
        query: str,
        casebase: Casebase[K, str],
    ) -> tuple[Casebase[K, str], dict[K, float]]:
        scores = await self._rerank(query, list(casebase.values()))
        key_index = dict(enumerate(casebase))
        return casebase, {key_index[index]: score for index, score in scores}

    @abstractmethod
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        """Score *documents* against *query*, returning ``(index, score)`` pairs."""
        ...


__all__ = ["RerankFunc"]
