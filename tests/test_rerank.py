"""Tests for the async-composable rerank retrievers."""

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import cbrkit
from cbrkit.retrieval.rerank._common import RerankFunc


@dataclass(slots=True, frozen=True)
class _StubIndexed:
    """Indexed-retriever stand-in: emits a fixed candidate casebase per query."""

    casebase: Mapping[str, str]

    async def __call__(
        self, batches: Sequence[tuple[Mapping[str, str], str]]
    ) -> Sequence[tuple[Mapping[str, str], dict[str, float]]]:
        return [(dict(self.casebase), {k: 1.0 for k in self.casebase}) for _ in batches]


@dataclass(slots=True)
class _LengthRerank(RerankFunc[str]):
    """Toy reranker preferring documents whose length matches the query's."""

    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        return [(i, -abs(len(d) - len(query))) for i, d in enumerate(documents)]


def test_rerank_composes_after_indexed_retriever() -> None:
    """A RerankFunc chains after an indexed retriever in the async pipeline."""
    candidates = {"a": "xx", "b": "xxxxxx", "c": "x"}
    result = asyncio.run(
        cbrkit.retrieval.apply_query_indexed_async(
            "xx", [_StubIndexed(candidates), _LengthRerank()]
        )
    )
    step = result.final_step.queries["default"]
    assert list(step.ranking) == ["a", "c", "b"]


def test_synced_runs_reranker_in_sync_pipeline() -> None:
    """`synced` drives an async reranker through the synchronous pipeline."""
    casebase = {"a": "xx", "b": "xxxxxx", "c": "x"}
    result = cbrkit.retrieval.apply_query(
        casebase, "xx", cbrkit.retrieval.synced(_LengthRerank())
    )
    assert list(result.ranking) == ["a", "c", "b"]


def test_http_reranker_parses_results() -> None:
    """The HTTP reranker maps endpoint results back onto casebase keys."""
    import httpx

    from cbrkit.retrieval.rerank.http import http

    casebase = {"a": "doc a", "b": "doc b", "c": "doc c"}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            },
            request=request,
        )

    async def run_reranker() -> dict[str, float]:
        client = httpx.AsyncClient(
            base_url="https://reranker.test/v1",
            transport=httpx.MockTransport(handler),
        )
        try:
            reranker = http[str](model="rerank-x", url="rerank", client=client, top_n=2)
            ((_, scores),) = await reranker([(casebase, "query")])
            return scores
        finally:
            await client.aclose()

    scores = asyncio.run(run_reranker())

    assert scores == {"a": 0.1, "b": 0.9, "c": 0.5}
    assert requests[0].url == "https://reranker.test/v1/rerank"
    body = cast(dict[str, Any], json.loads(requests[0].content))
    assert body == {
        "model": "rerank-x",
        "query": "query",
        "documents": ["doc a", "doc b", "doc c"],
        "top_n": 2,
    }
