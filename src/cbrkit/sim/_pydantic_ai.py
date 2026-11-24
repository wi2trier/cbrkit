"""Language models of Pydantic AI as generation functions, requires the `pydantic-ai` extra."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from pydantic_ai import Agent

from ..typing import GenerationFunc

__all__ = ["PydanticAI"]


@cache
def _agent[T](model: str, response_type: type[T]) -> Agent[None, T]:
    return Agent(model, output_type=response_type)


@dataclass(slots=True, frozen=True)
class PydanticAI(GenerationFunc):
    """Generates a structured response for every prompt with a Pydantic AI model, running the prompts concurrently.

    The agent of every model and response type is built once per process and shared.

    Args:
        model: Pydantic AI model identifier, such as `anthropic:claude-sonnet-4-6`,
            with keys and endpoints from the environment variables of the provider.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     score: float
        >>> answers = asyncio.run(PydanticAI("test")(["a", "b"], Answer))
        >>> [type(answer).__name__ for answer in answers]
        ['Answer', 'Answer']
    """

    model: str

    async def __call__[T](
        self, prompts: Sequence[str], response_type: type[T], /
    ) -> Sequence[T]:
        agent = _agent(self.model, response_type)
        results = await asyncio.gather(*(agent.run(prompt) for prompt in prompts))

        return [result.output for result in results]
