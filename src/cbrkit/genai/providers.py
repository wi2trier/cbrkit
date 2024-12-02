import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from ..helpers import unpack_value
from ..typing import BatchGenerationFunc
from ._model import ChatMessage, ChatPrompt

__all__ = []


# class StructuredOpenaiPrompt(ChatPrompt[str]): ...


type OpenaiPrompt = str | ChatPrompt[str]

try:
    from openai import AsyncOpenAI, pydantic_function_tool
    from openai.types.chat import ChatCompletionMessageParam

    @dataclass(slots=True)
    class openai[R: BaseModel | str](BatchGenerationFunc[OpenaiPrompt, R]):
        model: str
        response_type: type[R]
        system_message: str | None = None
        messages: Sequence[ChatMessage] = field(default_factory=tuple)
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)
        frequency_penalty: float | None = None
        logit_bias: dict[str, int] | None = None
        logprobs: bool | None = None
        max_completion_tokens: int | None = None
        max_tokens: int | None = None
        metadata: dict[str, str] | None = None
        n: int | None = None
        presence_penalty: float | None = None
        seed: int | None = None
        stop: str | list[str] | None = None
        store: bool | None = None
        temperature: float | None = None
        timeout: float | None = None
        top_logprobs: int | None = None
        top_p: float | None = None
        extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

        def __call__(self, batches: Sequence[OpenaiPrompt]) -> Sequence[R]:
            return asyncio.run(self._generate(batches))

        async def _generate(self, batches: Sequence[OpenaiPrompt]) -> Sequence[R]:
            return await asyncio.gather(
                *(self._generate_single(batch) for batch in batches)
            )

        async def _generate_single(self, prompt: OpenaiPrompt) -> R:
            messages: list[ChatCompletionMessageParam] = []

            if self.system_message:
                messages.append(
                    {
                        "role": "system",
                        "content": self.system_message,
                    }
                )

            messages.extend(cast(Sequence[ChatCompletionMessageParam], self.messages))

            if isinstance(prompt, ChatPrompt):
                messages.extend(
                    cast(Sequence[ChatCompletionMessageParam], prompt.messages)
                )

            if self.messages and self.messages[-1]["role"] == "user":
                messages.append(
                    {
                        "role": "assistant",
                        "content": unpack_value(prompt),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": unpack_value(prompt),
                    }
                )

            result: R | None = None

            if self.response_type is BaseModel:
                tool = pydantic_function_tool(cast(type[BaseModel], self.response_type))

                res = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    tools=[tool],
                    tool_choice={
                        "type": "function",
                        "function": {"name": tool["function"]["name"]},
                    },
                    frequency_penalty=self.frequency_penalty,
                    logit_bias=self.logit_bias,
                    logprobs=self.logprobs,
                    max_completion_tokens=self.max_completion_tokens,
                    max_tokens=self.max_tokens,
                    metadata=self.metadata,
                    n=self.n,
                    presence_penalty=self.presence_penalty,
                    seed=self.seed,
                    stop=self.stop,
                    store=self.store,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    top_logprobs=self.top_logprobs,
                    top_p=self.top_p,
                    **self.extra_kwargs,
                )

                tool_calls = res.choices[0].message.tool_calls

                if tool_calls is None:
                    raise ValueError("The completion is empty")

                parsed = tool_calls[0].function.parsed_arguments

                if parsed is None:
                    raise ValueError("The tool call is empty")

                result = cast(R, parsed)

            else:
                res = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    frequency_penalty=self.frequency_penalty,
                    logit_bias=self.logit_bias,
                    logprobs=self.logprobs,
                    max_completion_tokens=self.max_completion_tokens,
                    max_tokens=self.max_tokens,
                    metadata=self.metadata,
                    n=self.n,
                    presence_penalty=self.presence_penalty,
                    seed=self.seed,
                    stop=self.stop,
                    store=self.store,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    top_logprobs=self.top_logprobs,
                    top_p=self.top_p,
                    **self.extra_kwargs,
                )

                content = res.choices[0].message.content

                if content is None:
                    raise ValueError("The completion is empty")

                result = cast(R, content)

            return result

    __all__ += ["openai"]

except ImportError:
    pass
