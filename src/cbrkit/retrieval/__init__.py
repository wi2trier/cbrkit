import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast, override

from ..model import QueryResultStep, Result, ResultStep
from ..typing import (
    Casebase,
    HasMetadata,
    JsonDict,
    RetrieverFunc,
)
from . import genai
from ._apply import apply_batches, apply_queries, apply_query
from ._build import build, dropout, transpose

__all__ = [
    "build",
    "transpose",
    "dropout",
    "genai",
    "apply_batches",
    "apply_queries",
    "apply_query",
    "Result",
    "ResultStep",
    "QueryResultStep",
]


try:
    from cohere import AsyncClient
    from cohere.core import RequestOptions

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
        ) -> Sequence[Casebase[K, float]]:
            return asyncio.run(self._retrieve(batches))

        async def _retrieve(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(query, casebase) for casebase, query in batches)
            )

        async def _retrieve_single(
            self,
            query: str,
            casebase: Casebase[K, str],
        ) -> dict[K, float]:
            response = await self.client.v2.rerank(
                model=self.model,
                query=query,
                documents=list(casebase.values()),
                return_documents=False,
                max_tokens_per_doc=self.max_tokens_per_doc,
                request_options=self.request_options,
            )
            key_index = {idx: key for idx, key in enumerate(casebase)}

            return {
                key_index[result.index]: result.relevance_score
                for result in response.results
            }

    __all__ += ["cohere"]

except ImportError:
    pass


try:
    from voyageai.client_async import AsyncClient

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
        ) -> Sequence[Casebase[K, float]]:
            return asyncio.run(self._retrieve(batches))

        async def _retrieve(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(query, casebase) for casebase, query in batches)
            )

        async def _retrieve_single(
            self,
            query: str,
            casebase: Casebase[K, str],
        ) -> dict[K, float]:
            response = await self.client.rerank(
                model=self.model,
                query=query,
                documents=list(casebase.values()),
                truncation=self.truncation,
            )
            key_index = {idx: key for idx, key in enumerate(casebase)}

            return {
                key_index[result.index]: result.relevance_score
                for result in response.results
            }

    __all__ += ["voyageai"]

except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer, util

    @dataclass(slots=True, frozen=True)
    class sentence_transformers[K](
        RetrieverFunc[K, str, float],
        HasMetadata,
    ):
        """Semantic similarity using sentence transformers

        Args:
            model: Name of the [sentence transformer model](https://www.sbert.net/docs/pretrained_models.html).
        """

        model: SentenceTransformer | str
        query_chunk_size: int = 100
        corpus_chunk_size: int = 500000
        device: str = "cpu"

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "model": self.model if isinstance(self.model, str) else "custom",
                "query_chunk_size": self.query_chunk_size,
                "corpus_chunk_size": self.corpus_chunk_size,
                "device": self.device,
            }

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            model = (
                SentenceTransformer(self.model, device=self.device)
                if isinstance(self.model, str)
                else self.model
            )

            return [
                self._retrieve_single(query, casebase, model)
                for casebase, query in batches
            ]

        def _retrieve_single(
            self,
            query: str,
            casebase: Casebase[K, str],
            model: SentenceTransformer,
        ) -> dict[K, float]:
            case_texts = list(casebase.values())
            query_text = query
            embeddings = model.encode([query_text] + case_texts, convert_to_tensor=True)
            embeddings = embeddings.to(self.device)
            embeddings = util.normalize_embeddings(embeddings)
            query_embeddings = embeddings[0:1]
            case_embeddings = embeddings[1:]

            response = util.semantic_search(
                query_embeddings,
                case_embeddings,
                top_k=len(casebase),
                query_chunk_size=self.query_chunk_size,
                corpus_chunk_size=self.corpus_chunk_size,
                score_function=util.dot_score,
            )[0]
            key_index = {idx: key for idx, key in enumerate(casebase)}

            return {
                key_index[cast(int, res["corpus_id"])]: cast(float, res["score"])
                for res in response
            }

    __all__ += ["sentence_transformers"]

except ImportError:
    pass
