"""Shared scaffolding for rerank-model retrievers."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import override

from ...typing import AsyncRetrieverFunc, Casebase


@dataclass(slots=True)
class RerankFunc[K](AsyncRetrieverFunc[K, str, float], ABC):
    """Async base for rerank-model retrievers that score documents per query.

    Subclasses implement the single async call :meth:`_rerank`; the per-query
    fan-out and index-to-key mapping are shared here.  Because the base is an
    :class:`~cbrkit.typing.AsyncRetrieverFunc`, instances drop straight into an
    async retrieval pipeline — chain one after an indexed retriever in
    :func:`cbrkit.retrieval.apply_query_indexed_async`, where the upstream
    retriever supplies the candidate casebase this reranker rescores.
    """

    @override
    async def __call__(
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
        documents = list(casebase.values())
        if not documents:
            return casebase, {}
        scores = await self._rerank(query, documents)
        key_index = dict(enumerate(casebase))
        return casebase, {key_index[index]: score for index, score in scores}

    @abstractmethod
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        """Score *documents* against *query*, returning ``(index, score)`` pairs."""
        ...


__all__ = ["RerankFunc"]
