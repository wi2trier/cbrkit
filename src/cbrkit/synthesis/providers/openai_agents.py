from dataclasses import dataclass
from functools import partial
from typing import Any, cast, override
from uuid import uuid1

from ...helpers import get_logger, optional_dependencies, produce_sequence
from ...typing import MaybeSequence
from .model import AsyncProvider

logger = get_logger(__name__)

with optional_dependencies():
    from agents import (
        Agent,
        RunConfig,
        RunHooks,
        Runner,
        RunResult,
        SQLiteSession,
        TResponseInputItem,
    )
    from agents.run import DEFAULT_MAX_TURNS

    type OpenaiAgentsPrompt = str | list[TResponseInputItem]

    # the output is any in the base class, so we override it here
    class TypedRunResult[R](RunResult):
        """RunResult subclass with a typed final_output field."""

        final_output: R

    @dataclass(slots=True)
    class openai_agents[T, R](AsyncProvider[OpenaiAgentsPrompt, TypedRunResult[R]]):
        """Provider that runs OpenAI Agents SDK agents."""

        agents: MaybeSequence[Agent[T]]
        context: T | None = None
        max_turns: int = DEFAULT_MAX_TURNS
        hooks: RunHooks[T] | None = None
        run_config: RunConfig | None = None

        @override
        async def __call_batch__(self, prompt: OpenaiAgentsPrompt) -> TypedRunResult[R]:
            agents = produce_sequence(self.agents)

            if not agents:
                raise ValueError("No agents given.")

            head_agent, *tail_agents = agents

            session: Any = SQLiteSession(uuid1().hex) if len(agents) > 1 else None

            run = partial(
                Runner.run,
                context=self.context,
                max_turns=self.max_turns,
                hooks=self.hooks,
                run_config=self.run_config,
                session=session,
            )

            response: RunResult = await run(head_agent, prompt)

            for agent in tail_agents:
                response = await run(agent, [])

            return cast(TypedRunResult[R], response)
