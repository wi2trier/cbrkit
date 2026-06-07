"""Re-ranking via an HTTP ``/rerank`` endpoint."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import override

import httpx
from pydantic import BaseModel

from ._common import RerankFunc


class _RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


class _RerankResult(BaseModel):
    index: int
    relevance_score: float


class _RerankResponse(BaseModel):
    results: list[_RerankResult]


@dataclass(slots=True)
class http[K](RerankFunc[K]):
    """Re-ranking via an HTTP ``/rerank`` endpoint.

    The endpoint receives ``{"model", "query", "documents"}`` and returns
    ``{"results": [{"index", "relevance_score"}]}``.

    Args:
        model: Name of the reranker model served by the endpoint.
        url: Absolute URL, or a path resolved against the client's base URL.
        client: HTTP client used for transport, auth, and timeouts.
        api_key: Optional bearer token sent as an ``Authorization`` header.
            Kept separate from ``client`` so a shared transport can stay
            auth-neutral while each reranker carries its own token. Excluded
            from ``repr`` so the secret is not leaked when printing instances.
        extra_headers: Optional extra headers sent with every request.
            ``api_key`` takes precedence over an ``Authorization`` entry here.
        top_n: Optional cap on the number of results the server returns.
    """

    model: str
    url: str
    client: httpx.AsyncClient = field(
        default_factory=httpx.AsyncClient, repr=False, compare=False
    )
    api_key: str | None = field(default=None, repr=False)
    extra_headers: Mapping[str, str] | None = None
    top_n: int | None = None

    @property
    def _request_headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @override
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        body = _RerankRequest(
            model=self.model, query=query, documents=documents, top_n=self.top_n
        )
        response = await self.client.post(
            self.url,
            json=body.model_dump(exclude_none=True),
            headers=self._request_headers,
        )
        response.raise_for_status()
        data = _RerankResponse.model_validate_json(response.content)
        return ((result.index, result.relevance_score) for result in data.results)


__all__ = ["http"]
