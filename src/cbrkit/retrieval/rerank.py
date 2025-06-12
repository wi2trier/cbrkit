import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, override

from ..helpers import event_loop, get_logger, optional_dependencies
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
        ) -> Sequence[Casebase[K, float]]:
            return event_loop.get().run_until_complete(self._retrieve(batches))

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
        ) -> Sequence[Casebase[K, float]]:
            return event_loop.get().run_until_complete(self._retrieve(batches))

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
        ) -> Sequence[dict[K, float]]:
            if isinstance(self.model, str):
                model = SentenceTransformer(self.model, device=self.device)  # pyright: ignore
            else:
                model = self.model

            model.to(self.device)

            # if all casebases are the same, we can optimize the retrieval
            first_casebase = batches[0][0]

            if all(casebase == first_casebase for casebase, _ in batches):
                logger.debug(
                    "All casebases are the same, performing for all queries in one go"
                )
                return self.__call_queries__(
                    [query for _, query in batches], first_casebase, model
                )

            logger.debug("Casebases are different, performing retrieval for each query")
            return [
                self.__call_query__(query, casebase, model)
                for casebase, query in batches
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

        def __call_query__(
            self,
            query: str,
            casebase: Casebase[K, str],
            model: SentenceTransformer,
        ) -> dict[K, float]:
            case_texts = list(casebase.values())
            query_text = query
            embeddings = util.normalize_embeddings(
                model.encode([query_text] + case_texts, convert_to_tensor=True).to(
                    self.device
                )
            )
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


with optional_dependencies():
    import bm25s
    import numpy as np
    import Stemmer

    @dataclass(slots=True)
    class bm25[K](RetrieverFunc[K, str, float]):
        """BM25 retriever based on bm25s"""

        language: str
        stopwords: list[str] | None = None
        _indexed_retriever: bm25s.BM25 | None = field(
            default=None, init=False, repr=False
        )
        _indexed_casebase: Casebase[K, str] | None = field(
            default=None, init=False, repr=False
        )

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[dict[K, float]]:
            stemmer = Stemmer.Stemmer(self.language)
            stopwords: str | list[str] = self.stopwords or self.language

            # if all casebases are the same, we can optimize the retrieval
            first_casebase = batches[0][0]

            if all(casebase == first_casebase for casebase, _ in batches):
                logger.debug(
                    "All casebases are the same, performing for all queries in one go"
                )
                return self.__call_queries__(
                    [query for _, query in batches], first_casebase, stemmer, stopwords
                )

            logger.debug("Casebases are different, performing retrieval for each query")
            return [
                self.__call_queries__([query], casebase, stemmer, stopwords)[0]
                for casebase, query in batches
            ]

        def __call_queries__(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
            stemmer: Callable[..., Any],
            stopwords: str | list[str],
        ) -> Sequence[dict[K, float]]:
            if self._indexed_retriever and self._indexed_casebase == casebase:
                retriever = self._indexed_retriever
            else:
                cases_tokens = bm25s.tokenize(
                    list(casebase.values()), stemmer=stemmer, stopwords=stopwords
                )
                retriever = bm25s.BM25()
                retriever.index(cases_tokens)
                self._indexed_retriever = retriever
                self._indexed_casebase = dict(casebase)

            queries_tokens = bm25s.tokenize(
                cast(list[str], queries), stemmer=stemmer, stopwords=stopwords
            )

            results, scores = retriever.retrieve(
                queries_tokens,
                sorted=False,
                k=len(casebase),
            )
            max_score = np.max(scores)
            min_score = np.min(scores)

            key_index = {idx: key for idx, key in enumerate(casebase)}

            return [
                {
                    key_index[case_id]: float(
                        (score - min_score) / (max_score - min_score)
                    )
                    for case_id, score in zip(
                        results[query_id], scores[query_id], strict=True
                    )
                }
                for query_id in range(len(queries))
            ]
