from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import override

from cohere import AsyncClient
from cohere.core import RequestOptions

from ...helpers import get_logger
from ._common import RerankFunc

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class cohere[K](RerankFunc[K]):
    """Semantic similarity using Cohere's rerank models

    Args:
        model: Name of the [rerank model](https://docs.cohere.com/reference/rerank).
    """

    model: str
    max_tokens_per_doc: int | None = None
    client: AsyncClient = field(default_factory=AsyncClient, repr=False)
    request_options: RequestOptions | None = field(default=None, repr=False)

    @override
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        response = await self.client.v2.rerank(
            model=self.model,
            query=query,
            documents=documents,
            max_tokens_per_doc=self.max_tokens_per_doc,
            request_options=self.request_options,
        )
        return ((result.index, result.relevance_score) for result in response.results)


__all__ = ["cohere"]
