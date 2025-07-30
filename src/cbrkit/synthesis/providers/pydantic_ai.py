from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from cbrkit import helpers
from cbrkit.typing import MaybeSequence

from ...helpers import get_logger, optional_dependencies
from .model import AsyncProvider

logger = get_logger(__name__)

with optional_dependencies():
    from pydantic_ai.agent import Agent, AgentRunResult
    from pydantic_ai.messages import UserContent

    type PydanticAiPrompt = str | Sequence[UserContent]

    @dataclass(slots=True)
    class pydantic_ai[T, R](AsyncProvider[PydanticAiPrompt, AgentRunResult[R]]):
        agents: MaybeSequence[Agent[T, R]]
        deps: T

        @override
        async def __call_batch__(self, prompt: PydanticAiPrompt) -> AgentRunResult[R]:
            agents = helpers.produce_sequence(self.agents)

            if not agents:
                raise ValueError("No agents given.")

            head_agent, *tail_agents = agents

            response: AgentRunResult[R] = await head_agent.run(prompt, deps=self.deps)

            for agent in tail_agents:
                response = await agent.run(
                    # inject the system prompt because the default is not used if message history is provided
                    agent._system_prompts,
                    deps=self.deps,
                    message_history=response.all_messages() if response else None,
                )

            return response
