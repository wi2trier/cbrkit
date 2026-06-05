from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import override

from voyageai.client_async import AsyncClient

from ._common import RerankFunc


@dataclass(slots=True)
class voyageai[K](RerankFunc[K]):
    """Semantic similarity using Voyage AI's rerank models

    Args:
        model: Name of the [rerank model](https://docs.voyageai.com/docs/reranker).
    """

    model: str
    truncation: bool = True
    client: AsyncClient = field(default_factory=AsyncClient, repr=False)

    @override
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        response = await self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            truncation=self.truncation,
        )
        return ((result.index, result.relevance_score) for result in response.results)


__all__ = ["voyageai"]
