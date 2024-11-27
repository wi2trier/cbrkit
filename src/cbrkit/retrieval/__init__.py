import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import cast, override

import orjson
from pydantic import BaseModel

from ..model import QueryResultStep, Result, ResultStep
from ..typing import Casebase, HasMetadata, JsonDict, RetrieverFunc
from ._apply import apply_pairs, apply_queries, apply_query
from ._build import build, dropout, transpose

__all__ = [
    "build",
    "transpose",
    "dropout",
    "apply_pairs",
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
        max_chunks_per_doc: int | None = None
        client: AsyncClient = field(default_factory=AsyncClient, repr=False)
        request_options: RequestOptions | None = field(default=None, repr=False)

        @override
        def __call__(
            self,
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return asyncio.run(self._retrieve(pairs))

        async def _retrieve(
            self,
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(query, casebase) for casebase, query in pairs)
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
                max_chunks_per_doc=self.max_chunks_per_doc,
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
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return asyncio.run(self._retrieve(pairs))

        async def _retrieve(
            self,
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(query, casebase) for casebase, query in pairs)
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
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            model = (
                SentenceTransformer(self.model, device=self.device)
                if isinstance(self.model, str)
                else self.model
            )

            return [
                self._retrieve_single(query, casebase, model)
                for casebase, query in pairs
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


try:
    from openai import AsyncOpenAI, pydantic_function_tool
    from openai.types.chat import ChatCompletionMessageParam

    class RetrievalModel(BaseModel):
        similarities: dict[str, float]

    @dataclass(slots=True, frozen=True)
    class openai[V](RetrieverFunc[str, V, float]):
        model: str
        prompt: str | Callable[[Casebase[str, V], V], str]
        messages: Sequence[ChatCompletionMessageParam] = field(default_factory=list)
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)

        def __call__(
            self,
            pairs: Sequence[tuple[Casebase[str, V], V]],
        ) -> Sequence[Casebase[str, float]]:
            if self.messages and self.messages[-1]["role"] == "user":
                raise ValueError("The last message cannot be from the user")

            return asyncio.run(self._retrieve(pairs))

        async def _retrieve(
            self,
            pairs: Sequence[tuple[Casebase[str, V], V]],
        ) -> Sequence[Casebase[str, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(*pair) for pair in pairs)
            )

        async def _retrieve_single(
            self,
            casebase: Casebase[str, V],
            query: V,
        ) -> dict[str, float]:
            if isinstance(self.prompt, Callable):
                prompt = self.prompt(casebase, query)

            else:
                prompt = dedent(f"""
                    {self.prompt}

                    ## Query

                    ```json
                    {str(orjson.dumps(query))}
                    ```

                    ## Cases
                """)

                for key, value in casebase.items():
                    prompt += dedent(f"""
                        ### {key}

                        ```json
                        {str(orjson.dumps(value))}
                        ```
                    """)

            messages: list[ChatCompletionMessageParam] = [
                *self.messages,
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            tool = pydantic_function_tool(RetrievalModel)

            res = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                tools=[tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": tool["function"]["name"]},
                },
            )

            tool_calls = res.choices[0].message.tool_calls

            if tool_calls is None:
                raise ValueError("The completion is empty")

            parsed = tool_calls[0].function.parsed_arguments

            if parsed is None:
                raise ValueError("The tool call is empty")

            return cast(RetrievalModel, parsed).similarities

    __all__ += ["openai"]

except ImportError:
    pass
