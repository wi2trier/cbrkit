"""Embedding providers (one optional module per provider)."""

from ....helpers import optional_dependencies

with optional_dependencies():
    from .spacy import load_spacy, spacy

with optional_dependencies():
    from .pydantic_ai import pydantic_ai

with optional_dependencies():
    from .sentence_transformers import sentence_transformers

with optional_dependencies():
    from .sparse_encoder import sparse_encoder

with optional_dependencies():
    from .bm25 import bm25

with optional_dependencies():
    from .openai import openai

with optional_dependencies():
    from .ollama import ollama

with optional_dependencies():
    from .cohere import cohere

with optional_dependencies():
    from .voyageai import voyageai

__all__ = [
    "spacy",
    "load_spacy",
    "pydantic_ai",
    "sentence_transformers",
    "sparse_encoder",
    "bm25",
    "openai",
    "ollama",
    "cohere",
    "voyageai",
]
