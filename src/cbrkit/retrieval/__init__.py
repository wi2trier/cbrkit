from ..helpers import optional_dependencies
from ..model import QueryResultStep, Result, ResultStep
from .apply import apply_batches, apply_queries, apply_query
from .build import build, combine, distribute, dropout, transpose, transpose_value
from .rerank import embed

with optional_dependencies():
    from .build import chunk

with optional_dependencies():
    from .rerank import cohere

with optional_dependencies():
    from .rerank import voyageai

with optional_dependencies():
    from .rerank import sentence_transformers

with optional_dependencies():
    from .rerank import bm25

with optional_dependencies():
    from .rerank import lancedb

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
    "apply_query",
    "Result",
    "ResultStep",
    "QueryResultStep",
    "cohere",
    "voyageai",
    "sentence_transformers",
    "bm25",
    "embed",
    "lancedb",
]
