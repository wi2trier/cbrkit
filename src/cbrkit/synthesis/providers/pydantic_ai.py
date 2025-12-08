from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, override

from cbrkit import helpers
from cbrkit.typing import MaybeSequence

from ...helpers import get_logger, optional_dependencies
from .model import AsyncProvider

logger = get_logger(__name__)

with optional_dependencies():
    from pydantic_ai.agent import Agent, AgentRunResult
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        UserContent,
    )

    type PydanticAiPrompt = str | Sequence[UserContent] | Sequence[ModelMessage]

    @dataclass(slots=True)
    class pydantic_ai[T, R](AsyncProvider[PydanticAiPrompt, AgentRunResult[R]]):
        agents: MaybeSequence[Agent[T, R]]
        deps: T

        @override
        async def __call_batch__(self, prompt: PydanticAiPrompt) -> AgentRunResult[R]:
            agents = helpers.produce_sequence(self.agents)

            user_prompt: str | Sequence[UserContent] | None = None
            message_history: Sequence[ModelMessage] | None = None

            if isinstance(prompt, str) or all(
                isinstance(msg, UserContent) for msg in prompt
            ):
                user_prompt = cast(str | Sequence[UserContent], prompt)
            if all(isinstance(msg, ModelRequest | ModelResponse) for msg in prompt):
                message_history = cast(Sequence[ModelMessage], prompt)
            else:
                raise ValueError(
                    "Prompt must be a string, a sequence of UserContent, or a sequence of ModelMessage."
                )

            response: AgentRunResult[R] | None = None

            for agent in agents:
                response = await agent.run(
                    user_prompt=user_prompt,
                    deps=self.deps,
                    message_history=message_history,
                )
                message_history = response.all_messages()

            if not response:
                raise ValueError("No agents given.")

            return response
