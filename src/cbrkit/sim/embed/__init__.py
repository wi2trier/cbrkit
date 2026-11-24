"""Embedding-based similarity: metrics, builder, cache, and providers.

`Embedding` compares two texts by the vectors of an `Embedder` with a metric such as `Cosine`,
where the `Embedder` names an embedding function, the one it is bound to in Python or by `cbrkit.spec.bind`,
and `EmbeddingCache` keeps the vectors of an embedding function in a store the caller owns,
a dict in memory or a `SQLiteStore` on disk.
Local embedding functions are `SentenceTransformers`, `SparseEncoder` and `BM25Encoder`,
and `PydanticAI` reaches every remote provider that Pydantic AI supports, with the `pydantic-ai` extra.
A specification names its embedders, which `cbrkit.spec.bind` binds, so it holds no provider settings.
Embedding functions are asynchronous (`cbrkit.typing.EmbeddingFunc`), local models load their weights on the first call,
and the synchronous `Embedding` measure waits for them with `cbrkit.helpers.wait`.

Example:
    >>> from pathlib import Path
    >>> embedder = EmbeddingCache(embedder=SentenceTransformers("all-MiniLM-L6-v2"), store={})  # doctest: +SKIP
    >>> persistent = EmbeddingCache(  # doctest: +SKIP
    ...     embedder=SentenceTransformers("all-MiniLM-L6-v2"),
    ...     store=SQLiteStore(Path("embeddings.sqlite3"), "all-MiniLM-L6-v2"),
    ... )
    >>> measure = Embedding(embedder=Embedder("minilm", embedder), metric=Cosine())  # doctest: +SKIP
"""

from typing import TYPE_CHECKING

from ..._lazy import lazy_exports
from .core import (
    Concat,
    Embedder,
    Embedding,
    EmbeddingCache,
    Scratch,
    SQLiteStore,
    embed_pairs,
    for_queries,
)
from .metrics import (
    Angular,
    Cosine,
    Dot,
    Euclidean,
    Manhattan,
    SparseCosine,
    SparseDot,
)

if TYPE_CHECKING:
    from .providers.bm25 import BM25Encoder
    from .providers.pydantic_ai import PydanticAI
    from .providers.sentence_transformers import SentenceTransformers
    from .providers.sparse_encoder import SparseEncoder

__all__ = [
    "Angular",
    "BM25Encoder",
    "Concat",
    "Cosine",
    "Dot",
    "Embedder",
    "Embedding",
    "EmbeddingCache",
    "Euclidean",
    "Manhattan",
    "PydanticAI",
    "SQLiteStore",
    "Scratch",
    "SentenceTransformers",
    "SparseCosine",
    "SparseDot",
    "SparseEncoder",
    "embed_pairs",
    "for_queries",
]

lazy_exports(__name__)
