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
from .wrappers import combine, distribute, dropout, stateful, transpose, transpose_value

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
    "stateful",
]
