import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast, override

from ..helpers import (
    dispatch_batches,
    get_logger,
    optional_dependencies,
    run_coroutine,
)
from ..typing import (
    Casebase,
    HasMetadata,
    JsonDict,
    RetrieverFunc,
)

logger = get_logger(__name__)


with optional_dependencies():
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


with optional_dependencies():
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


with optional_dependencies():
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
        device: str | None = None

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
        ) -> Sequence[tuple[Casebase[K, str], dict[K, float]]]:
            if not batches:
                return []

            if isinstance(self.model, str):
                model = SentenceTransformer(self.model, device=self.device)  # pyright: ignore
            else:
                model = self.model

            model.to(self.device)

            sim_maps = dispatch_batches(
                batches,
                lambda queries, casebase: self.__call_queries__(
                    queries, casebase, model
                ),
            )

            return [
                (casebase, sim_map)
                for (casebase, _), sim_map in zip(batches, sim_maps, strict=True)
            ]

        def __call_queries__(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
            model: SentenceTransformer,
        ) -> Sequence[dict[K, float]]:
            case_texts = list(casebase.values())
            query_texts = cast(list[str], queries)

            case_embeddings = util.normalize_embeddings(
                model.encode(case_texts, convert_to_tensor=True).to(self.device)
            )
            query_embeddings = util.normalize_embeddings(
                model.encode(query_texts, convert_to_tensor=True).to(self.device)
            )

            response = util.semantic_search(
                query_embeddings,
                case_embeddings,
                top_k=len(casebase),
                query_chunk_size=self.query_chunk_size,
                corpus_chunk_size=self.corpus_chunk_size,
                score_function=util.dot_score,
            )

            key_index = {idx: key for idx, key in enumerate(casebase)}

            return [
                {
                    key_index[cast(int, res["corpus_id"])]: cast(float, res["score"])
                    for res in query_response
                }
                for query_response in response
            ]
