import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import override

from cohere import AsyncClient
from cohere.core import RequestOptions

from ...helpers import get_logger, run_coroutine
from ...typing import Casebase, RetrieverFunc

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class cohere[K](RetrieverFunc[K, str, float]):
    """Semantic similarity using Cohere's rerank models

    Args:
        model: Name of the [rerank model](https://docs.cohere.com/reference/rerank).
    """

    model: str
    max_tokens_per_doc: int | None = None
    client: AsyncClient = field(default_factory=AsyncClient, repr=False)
    request_options: RequestOptions | None = field(default=None, repr=False)

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
        response = await self.client.v2.rerank(
            model=self.model,
            query=query,
            documents=list(casebase.values()),
            max_tokens_per_doc=self.max_tokens_per_doc,
            request_options=self.request_options,
        )
        key_index = {idx: key for idx, key in enumerate(casebase)}

        return (
            casebase,
            {
                key_index[result.index]: result.relevance_score
                for result in response.results
            },
        )


__all__ = ["cohere"]
