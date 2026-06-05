"""Re-ranking retrievers backed by rerank models.

A reranker rescores an already-retrieved candidate set, so it is chained
*after* an indexed retriever in an async pipeline
(``apply_query_indexed_async(query, [base, reranker])``).  Pick by deployment:
``cross_encoder`` (local sentence-transformers cross-encoder), ``http``
(generic HTTP ``/rerank`` endpoint, e.g. vLLM), ``cohere`` / ``voyageai``
(hosted APIs), or ``bi_encoder`` (cheap embedding-similarity rescoring).
"""

from ...helpers import optional_dependencies

with optional_dependencies():
    from .cohere import cohere

with optional_dependencies():
    from .voyageai import voyageai

with optional_dependencies():
    from .cross_encoder import cross_encoder

with optional_dependencies():
    from .bi_encoder import bi_encoder, sentence_transformers

with optional_dependencies():
    from .http import http

__all__ = [
    "cohere",
    "voyageai",
    "cross_encoder",
    "bi_encoder",
    "sentence_transformers",
    "http",
]
