from ...helpers import optional_dependencies
from .model import AsyncProvider, BaseProvider, Response, Usage
from .wrappers import conversation, pipe

with optional_dependencies():
    from .openai_completions import openai_completions

    openai = openai_completions
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
    "openai_completions",
    "openai",
    "pydantic_ai",
    "openai_agents",
    "ollama",
    "cohere",
    "instructor",
    "google",
    "anthropic",
    "pipe",
    "conversation",
]
