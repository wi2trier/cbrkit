"""Embedding-based similarity: metrics, builder, cache, and providers."""

from ...helpers import optional_dependencies
from .core import build, cache, concat
from .metrics import (
    angular,
    cosine,
    default_score_func,
    dot,
    euclidean,
    manhattan,
    sparse_cosine,
    sparse_dot,
)

with optional_dependencies():
    from .providers.spacy import load_spacy, spacy

with optional_dependencies():
    from .providers.pydantic_ai import pydantic_ai

with optional_dependencies():
    from .providers.sentence_transformers import sentence_transformers

with optional_dependencies():
    from .providers.sparse_encoder import sparse_encoder

with optional_dependencies():
    from .providers.bm25 import bm25

with optional_dependencies():
    from .providers.openai import openai

with optional_dependencies():
    from .providers.ollama import ollama

with optional_dependencies():
    from .providers.cohere import cohere

with optional_dependencies():
    from .providers.voyageai import voyageai

__all__ = [
    "cosine",
    "dot",
    "angular",
    "euclidean",
    "manhattan",
    "sparse_dot",
    "sparse_cosine",
    "default_score_func",
    "build",
    "cache",
    "concat",
    "spacy",
    "load_spacy",
    "sentence_transformers",
    "sparse_encoder",
    "bm25",
    "pydantic_ai",
    "openai",
    "ollama",
    "cohere",
    "voyageai",
]
