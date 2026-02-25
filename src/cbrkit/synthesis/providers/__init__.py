"""LLM provider integrations for synthesis.

Each provider wraps an LLM API and exposes a unified interface for use with
`cbrkit.synthesis.build`.
Providers are initialized with a model name and a response type (`str` for
plain text or a Pydantic model for structured output).
Additional options like `temperature`, `seed`, and `max_tokens` can be set.

Providers (each requires its respective extra and API key):
- `openai` / `openai_completions`: OpenAI Completions API (`OPENAI_API_KEY`).
- `openai_responses`: OpenAI Responses API (`OPENAI_API_KEY`).
- `openai_agents`: OpenAI Agents framework (`OPENAI_API_KEY`).
- `anthropic`: Anthropic Claude API (`ANTHROPIC_API_KEY`).
- `cohere`: Cohere API (`CO_API_KEY`).
- `google`: Google Generative AI (`GOOGLE_API_KEY`).
- `ollama`: Ollama local inference (no API key needed).
- `pydantic_ai`: Pydantic AI framework.
- `instructor`: Instructor for structured output.

Wrappers:
- `pipe`: Chains multiple providers sequentially.
- `conversation`: Manages multi-turn conversations with a provider.

Base Classes:
- `BaseProvider`: Base class for synchronous providers.
- `AsyncProvider`: Base class for asynchronous providers.
- `Response`: Response model returned by providers.
- `Usage`: Token usage tracking.

Example:
    >>> provider = openai(  # doctest: +SKIP
    ...     model="gpt-4o",
    ...     response_type=str,
    ...     temperature=0.7,
    ... )
"""

from ...helpers import optional_dependencies
from .model import AsyncProvider, BaseProvider, Response, Usage
from .wrappers import conversation, pipe

with optional_dependencies():
    from .openai_completions import openai_completions

    openai = openai_completions
with optional_dependencies():
    from .openai_responses import openai_responses
with optional_dependencies():
    from .anthropic import anthropic
with optional_dependencies():
    from .cohere import cohere
with optional_dependencies():
    from .ollama import ollama
with optional_dependencies():
    from .instructor import instructor
with optional_dependencies():
    from .google import google
with optional_dependencies():
    from .pydantic_ai import pydantic_ai
with optional_dependencies():
    from .openai_agents import openai_agents

__all__ = [
    "AsyncProvider",
    "BaseProvider",
    "Response",
    "Usage",
    "pipe",
    "conversation",
    "anthropic",
    "cohere",
    "google",
    "instructor",
    "ollama",
    "openai",
    "openai_agents",
    "openai_completions",
    "openai_responses",
    "pydantic_ai",
]
