"""Case retrieval with similarity-based, sparse, and embedding-based methods.

This module provides functions for building and applying retrieval pipelines.
Retrievers compute similarity scores between a query and cases in a casebase,
then return ranked results.
Multiple retrievers can be composed sequentially (MAC/FAC pattern) or combined
in parallel with score aggregation.

Building Retrievers:
    ``build``: Creates a retriever from a similarity function.
    Flattens all batches into pairwise comparisons and optionally parallelizes
    the similarity computations within batches.
    ``dropout``: Wraps a retriever with filtering by ``min_similarity`` and/or ``limit``.

Applying Retrievers:
    ``apply_query``: Runs retrieval for a single query against a casebase.
    ``apply_queries``: Runs retrieval for multiple queries.
    ``apply_batches``: Runs retrieval for batches of (casebase, query) pairs.
    ``apply_query_indexed`` / ``apply_queries_indexed``: Convenience functions
    for indexed retrieval without passing a casebase.

Wrappers:
    ``combine``: Merges results from multiple retrievers using an aggregator.
    ``distribute``: Parallelizes retrieval across batches by calling the wrapped
    retriever separately for each (casebase, query) pair.
    ``persist``: Caches retrieval results to disk.
    ``transpose`` / ``transpose_value``: Transforms cases/queries before retrieval.
    ``chunk``: Splits cases into chunks for retrieval (requires ``chunking`` extra).

Indexable Retrieval:
    ``embed``: Embedding-based retrieval using vector similarity.
    ``bm25``: BM25 sparse text retrieval (requires ``bm25`` extra).
    ``chromadb``: ChromaDB vector store retrieval (requires ``chromadb`` extra).
    ``lancedb``: LanceDB vector store retrieval (requires ``lancedb`` extra).
    ``zvec``: Zvec vector store retrieval (requires ``zvec`` extra).

Re-ranking:
    ``cohere``: Cohere re-ranking model (requires ``cohere`` extra).
    ``voyageai``: Voyage AI re-ranking model (requires ``voyageai`` extra).
    ``sentence_transformers``: Cross-encoder re-ranking (requires ``transformers`` extra).

Example:
    >>> from cbrkit.sim.generic import equality
    >>> retriever = build(equality())
    >>> result = apply_query({"a": "hello", "b": "world"}, "hello", retriever)
    >>> result.ranking[0]
    'a'
"""

from ..helpers import optional_dependencies
from ..model import QueryResultStep, Result, ResultStep
from .apply import (
    apply_batches,
    apply_queries,
    apply_queries_indexed,
    apply_query,
    apply_query_indexed,
)
from .build import build
from .indexable import embed
from .wrappers import combine, distribute, dropout, persist, transpose, transpose_value

with optional_dependencies():
    from .wrappers import chunk

with optional_dependencies():
    from .rerank import cohere

with optional_dependencies():
    from .rerank import voyageai

with optional_dependencies():
    from .rerank import sentence_transformers

with optional_dependencies():
    from .indexable import bm25

with optional_dependencies():
    from .indexable import chromadb

with optional_dependencies():
    from .indexable import lancedb

with optional_dependencies():
    from .indexable import zvec

__all__ = [
    "build",
    "transpose",
    "transpose_value",
    "dropout",
    "distribute",
    "combine",
    "chunk",
    "apply_batches",
    "apply_queries",
    "apply_queries_indexed",
    "apply_query",
    "apply_query_indexed",
    "Result",
    "ResultStep",
    "QueryResultStep",
    "cohere",
    "voyageai",
    "sentence_transformers",
    "bm25",
    "chromadb",
    "embed",
    "lancedb",
    "persist",
    "zvec",
]
