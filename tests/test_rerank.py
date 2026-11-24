"""Tests for the reranking of retrieved cases by rerank models."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import cbrkit
from cbrkit.retrieve.rerank import Rerank
from cbrkit.typing import Casebase, SimMap


@dataclass(slots=True, frozen=True)
class _StubStorageRetriever:
    """Storage-retriever stand-in: emits a fixed candidate casebase per query."""

    casebase: Mapping[str, str]

    async def __call__(
        self, batches: Sequence[tuple[Mapping[str, str], str]]
    ) -> Sequence[tuple[Casebase[str, str], SimMap[str, float]]]:
        return [(dict(self.casebase), {k: 1.0 for k in self.casebase}) for _ in batches]


async def _length(
    query: str, documents: Sequence[str], /, limit: int | None = None
) -> Mapping[int, float]:
    """Toy rerank model preferring documents whose length matches the query's."""
    return {i: -abs(len(d) - len(query)) for i, d in enumerate(documents)}


def test_rerank_composes_after_storage_retriever() -> None:
    """A reranker chains after a storage retriever in the async pipeline."""
    candidates = {"a": "xx", "b": "xxxxxx", "c": "x"}
    result = asyncio.run(
        cbrkit.cycle.run_query_async(
            None,
            "xx",
            _StubStorageRetriever(candidates),
            Rerank(model=_length, limit=2),
        )
    )
    assert list(result.ranking) == ["a", "c"]


def test_sync_and_async_functions_mix() -> None:
    """Both engines run synchronous and asynchronous functions in any order."""
    casebase = {"a": "xx", "b": "xxxxxx", "c": "x"}
    retriever = cbrkit.retrieve.Retriever(measure=lambda case, query: 1.0)
    rerank = Rerank(model=_length)
    result = cbrkit.cycle.run_query(casebase, "xx", retriever, rerank)
    result_async = asyncio.run(
        cbrkit.cycle.run_query_async(casebase, "xx", rerank, retriever)
    )

    assert list(result.ranking) == ["a", "c", "b"]
    assert set(result_async.casebase) == set(casebase)


def _run_http_reranker(
    response: dict[str, Any],
    batches: Sequence[tuple[Mapping[str, str], str]],
    client_kwargs: Mapping[str, Any] | None = None,
    limit: int | None = None,
    **model_kwargs: Any,
) -> tuple[Sequence[tuple[Casebase[str, str], SimMap[str, float]]], list[Any]]:
    """Run the HTTP reranker against a request-capturing mock endpoint."""
    import httpx2

    from cbrkit.retrieve.rerank import HTTP

    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=response, request=request)

    async def run() -> Sequence[tuple[Casebase[str, str], SimMap[str, float]]]:
        client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler), **(client_kwargs or {})
        )
        try:
            model = HTTP(client=client, **model_kwargs)
            return await Rerank(model=model, limit=limit)(batches)
        finally:
            await client.aclose()

    return asyncio.run(run()), requests


def test_http_reranker_parses_results() -> None:
    """The HTTP reranker maps endpoint results back onto casebase keys."""
    casebase = {"a": "doc a", "b": "doc b", "c": "doc c"}
    results, requests = _run_http_reranker(
        {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        },
        [(casebase, "query")],
        client_kwargs={"base_url": "https://reranker.test/v1"},
        model="rerank-x",
        url="rerank",
        limit=2,
    )

    ((_, scores),) = results
    assert scores == {"b": 0.9, "c": 0.5}
    assert requests[0].url == "https://reranker.test/v1/rerank"
    body = cast(dict[str, Any], json.loads(requests[0].content))
    assert body == {
        "model": "rerank-x",
        "query": "query",
        "documents": ["doc a", "doc b", "doc c"],
        "top_n": 2,
    }


def test_http_reranker_sends_api_key_and_headers() -> None:
    """``api_key`` becomes a bearer header alongside any ``extra_headers``."""
    _, requests = _run_http_reranker(
        {"results": []},
        [({"a": "doc a"}, "query")],
        model="rerank-x",
        url="https://reranker.test/rerank",
        api_key="secret",
        extra_headers={"X-Tenant": "acme"},
    )

    assert requests[0].headers["Authorization"] == "Bearer secret"
    assert requests[0].headers["X-Tenant"] == "acme"
