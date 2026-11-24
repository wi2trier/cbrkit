"""Reranking of retrieved cases by a scoring function of a rerank model."""

import asyncio
from dataclasses import dataclass

from ..._phase import Ranking
from ...typing import Batches, Casebase, RerankFunc, Results, SimMap

__all__ = ["Rerank"]


@dataclass(slots=True, frozen=True, kw_only=True)
class Rerank(Ranking):
    """Rescores the cases that a preceding function retrieved with a rerank model.

    Chain it after a retriever in `cbrkit.cycle.run_query_async`,
    where the upstream retriever supplies the candidate cases that the model rescores,
    one call of the model per query, which receives the `limit` so an endpoint returns only the best documents.

    Args:
        model: Scores the documents for a query, such as `CrossEncoder` or `HTTP`.

    Examples:
        >>> import asyncio
        >>> async def length(query, documents, /, limit=None):
        ...     return {i: -abs(len(d) - len(query)) for i, d in enumerate(documents)}
        >>> rerank = Rerank(model=length, limit=1)
        >>> asyncio.run(rerank([({"a": "xxxx", "b": "xx"}, "xx")]))
        [({'a': 'xxxx', 'b': 'xx'}, {'b': 0})]
    """

    model: RerankFunc

    async def __call__[K](self, batches: Batches[K, str], /) -> Results[K, str]:
        results = await asyncio.gather(
            *(self._rerank(casebase, query) for casebase, query in batches)
        )

        return self.select(results)

    async def _rerank[K](
        self, casebase: Casebase[K, str], query: str
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        if not casebase:
            return casebase, {}

        keys = list(casebase)
        scores = await self.model(query, list(casebase.values()), limit=self.limit)

        return casebase, {keys[index]: score for index, score in scores.items()}
