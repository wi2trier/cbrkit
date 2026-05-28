import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import override

from voyageai.client_async import AsyncClient

from ...helpers import get_logger, run_coroutine
from ...typing import Casebase, RetrieverFunc

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class voyageai[K](RetrieverFunc[K, str, float]):
    """Semantic similarity using Voyage AI's rerank models

    Args:
        model: Name of the [rerank model](https://docs.voyageai.com/docs/reranker).
    """

    model: str
    truncation: bool = True
    client: AsyncClient = field(default_factory=AsyncClient, repr=False)

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
        response = await self.client.rerank(
            model=self.model,
            query=query,
            documents=list(casebase.values()),
            truncation=self.truncation,
        )
        key_index = {idx: key for idx, key in enumerate(casebase)}

        return (
            casebase,
            {
                key_index[result.index]: result.relevance_score
                for result in response.results
            },
        )


__all__ = ["voyageai"]
