from ...helpers import optional_dependencies
from .embeddings import EmbedFunc, VectorDB, numpy_db
from .patterns import glob, regex, table
from .taxonomy import (
    SerializedTaxonomyNode,
    Taxonomy,
    TaxonomyNode,
    TaxonomySimFunc,
    TaxonomyStrategy,
    taxonomy_auto_weights,
    taxonomy_levels,
    taxonomy_paths,
    taxonomy_weights,
    wu_palmer,
)

with optional_dependencies():
    from .embeddings import torch_db

with optional_dependencies():
    from .edit import jaro, jaro_winkler, levenshtein

with optional_dependencies():
    from .edit import ngram

with optional_dependencies():
    from .embeddings import spacy

with optional_dependencies():
    from .embeddings import sentence_transformers

with optional_dependencies():
    from .embeddings import openai

with optional_dependencies():
    from .embeddings import ollama

with optional_dependencies():
    from .embeddings import voyageai

__all__ = [
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "ngram",
    "ngram",
    "regex",
    "glob",
    "table",
    "spacy",
    "sentence_transformers",
    "openai",
    "ollama",
    "voyageai",
    "numpy_db",
    "torch_db",
    "VectorDB",
    "EmbedFunc",
    "wu_palmer",
    "taxonomy_weights",
    "taxonomy_auto_weights",
    "taxonomy_levels",
    "taxonomy_paths",
    "Taxonomy",
    "TaxonomyNode",
    "SerializedTaxonomyNode",
    "TaxonomySimFunc",
    "TaxonomyStrategy",
]
