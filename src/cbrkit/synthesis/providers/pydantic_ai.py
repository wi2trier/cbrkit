from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, override

from ...helpers import get_logger, optional_dependencies, produce_sequence
from ...typing import MaybeSequence
from .model import AsyncProvider, Response, Usage

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
    class pydantic_ai[T, R](AsyncProvider[PydanticAiPrompt, Response[R]]):
        """Provider that runs pydantic-ai agents.

        Agents are run sequentially, threading the message history from one agent
        into the next, so a chain of agents can refine a shared conversation.
        Capabilities, tools, and model settings are configured on the agents
        themselves via the pydantic-ai v2 `capabilities` primitive.
        """

        agents: MaybeSequence[Agent[T, R]]
        deps: T

        @override
        async def __call_batch__(self, prompt: PydanticAiPrompt) -> Response[R]:
            agents = produce_sequence(self.agents)

            user_prompt: str | Sequence[UserContent] | None = None
            message_history: Sequence[ModelMessage] | None = None

            if isinstance(prompt, str):
                user_prompt = prompt
            elif all(isinstance(msg, (ModelRequest, ModelResponse)) for msg in prompt):
                message_history = cast(Sequence[ModelMessage], prompt)
            else:
                user_prompt = cast(Sequence[UserContent], prompt)

            result: AgentRunResult[R] | None = None
            usage = Usage()

            for agent in agents:
                result = await agent.run(
                    user_prompt=user_prompt,
                    deps=self.deps,
                    message_history=message_history,
                )
                usage = Usage(
                    usage.prompt_tokens + result.usage.input_tokens,
                    usage.completion_tokens + result.usage.output_tokens,
                )
                message_history = result.all_messages()

            if result is None:
                raise ValueError("No agents given.")

            return Response(result.output, usage)
