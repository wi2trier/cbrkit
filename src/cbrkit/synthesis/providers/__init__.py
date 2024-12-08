from ...helpers import optional_dependencies
from .model import BaseProvider, ChatMessage, ChatPrompt, ChatProvider, DocumentsPrompt
from .wrappers import conversation, pipe

with optional_dependencies():
    from .openai import openai
with optional_dependencies():
    from .ollama import ollama
with optional_dependencies():
    from .cohere import cohere

__all__ = [
    "openai",
    "ollama",
    "cohere",
    "conversation",
    "pipe",
    "BaseProvider",
    "ChatProvider",
    "ChatMessage",
    "ChatPrompt",
    "DocumentsPrompt",
]
