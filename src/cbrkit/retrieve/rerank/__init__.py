"""Reranking of retrieved cases by rerank models.

`Rerank` rescores an already-retrieved candidate set with a rerank model (`cbrkit.typing.RerankFunc`),
so it is chained after a retriever, for instance `cbrkit.cycle.run_query_async(None, query, base, Rerank(model=...))`.
It is an asynchronous function of the cycle, which `cbrkit.cycle.run_async` awaits.
`HTTP` takes the HTTP client of the host, and `CrossEncoder` loads its weights on the first call.
Pick the model by deployment:
`CrossEncoder` (local sentence-transformers cross-encoder)
or `HTTP` (generic HTTP `/rerank` endpoint, e.g. vLLM, or a hosted rerank API).
"""

from typing import TYPE_CHECKING

from ..._lazy import lazy_exports
from ._common import Rerank

if TYPE_CHECKING:
    from ._cross_encoder import CrossEncoder
    from ._http import HTTP

__all__ = [
    "HTTP",
    "CrossEncoder",
    "Rerank",
]

lazy_exports(__name__)
