"""LLM-based synthesis for generating insights from retrieved cases.

This module integrates Large Language Models with CBR to create new insights
from retrieved cases, for example in a Retrieval-Augmented Generation (RAG) context.
A synthesizer maps a retrieval ``Result`` to an LLM output which can be a plain
string or a structured Pydantic model.

Building Synthesizers:
    ``build``: Creates a synthesizer from a provider and a prompt function.
    ``transpose``: Wraps a synthesizer to transform cases/queries before prompting.
    ``chunks``: Splits the synthesis into batches with a pooling step for aggregation.
    ``pooling``: Creates a pooling function that aggregates partial synthesis results.

Applying Synthesizers:
    ``apply_result``: Applies the synthesizer to a retrieval result.
    ``apply_query``: Applies the synthesizer to a single query.
    ``apply_queries``: Applies the synthesizer to multiple queries.
    ``apply_batches``: Applies the synthesizer to batches.
    ``apply_casebase``: Applies the synthesizer to a full casebase.

Submodules:
    ``cbrkit.synthesis.prompts``: Prompt functions (``default``, ``document_aware``,
    ``transpose``, ``pooling``) that format cases and queries for the LLM.
    ``cbrkit.synthesis.providers``: LLM provider integrations (OpenAI, Anthropic,
    Cohere, Google, Ollama, and more).

Example:
    Build and apply a synthesizer::

        import cbrkit

        provider = cbrkit.synthesis.providers.openai(model="gpt-4o", response_type=str)
        prompt = cbrkit.synthesis.prompts.default(instructions="Summarize the cases.")
        synthesizer = cbrkit.synthesis.build(provider, prompt)
        response = cbrkit.synthesis.apply_result(retrieval_result, synthesizer).response
"""

from . import prompts, providers
from .apply import (
    apply_batches,
    apply_casebase,
    apply_queries,
    apply_query,
    apply_result,
)
from .build import build, chunks, pooling, transpose
from .model import QueryResultStep, Result, ResultStep

__all__ = [
    "providers",
    "prompts",
    "build",
    "transpose",
    "chunks",
    "pooling",
    "apply_batches",
    "apply_result",
    "apply_queries",
    "apply_query",
    "apply_casebase",
    "QueryResultStep",
    "ResultStep",
    "Result",
]
