"""Rerank model behind an HTTP `/rerank` endpoint."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import httpx2
from pydantic import BaseModel

from ...typing import RerankFunc


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


@dataclass(slots=True, frozen=True, kw_only=True)
class HTTP(RerankFunc):
    """Rerank model behind an HTTP `/rerank` endpoint, the scores of `Rerank`.

    The endpoint receives `{"model", "query", "documents", "top_n"}` and returns
    `{"results": [{"index", "relevance_score"}]}`, where `top_n` is the limit of the call, all documents if None.

    Args:
        model: Name of the reranker model served by the endpoint.
        url: Absolute URL, or a path resolved against the client's base URL.
        client: HTTP client used for transport, auth, and timeouts,
            which the host opens and closes, so it may share it with other rerankers.
        api_key: Optional bearer token sent as an `Authorization` header.
            Kept separate from `client` so a shared transport can stay
            auth-neutral while each reranker carries its own token. Excluded
            from `repr` so the secret is not leaked when printing instances.
        extra_headers: Optional extra headers sent with every request.
            `api_key` takes precedence over an `Authorization` entry here.
    """

    model: str
    url: str
    client: httpx2.AsyncClient = field(repr=False, compare=False)
    api_key: str | None = field(default=None, repr=False)
    extra_headers: Mapping[str, str] | None = None

    @property
    def _request_headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def __call__(
        self, query: str, documents: Sequence[str], /, limit: int | None = None
    ) -> Mapping[int, float]:
        body = _RerankRequest(
            model=self.model, query=query, documents=list(documents), top_n=limit
        )
        response = await self.client.post(
            self.url,
            json=body.model_dump(exclude_none=True),
            headers=self._request_headers,
        )
        response.raise_for_status()
        data = _RerankResponse.model_validate_json(response.content)
        return {result.index: result.relevance_score for result in data.results}


__all__ = ["HTTP"]
