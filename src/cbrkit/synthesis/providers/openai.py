from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider

with optional_dependencies():
    from openai import AsyncOpenAI, pydantic_function_tool
    from openai._types import NOT_GIVEN
    from openai.types.chat import ChatCompletionMessageParam

    type OpenaiPrompt = str | ChatPrompt[str]

    @dataclass(slots=True)
    class openai[R: BaseModel | str](ChatProvider[OpenaiPrompt, R]):
        structured_outputs: bool = True
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

        @override
        async def __call_batch__(self, prompt: OpenaiPrompt) -> R:
            messages: list[ChatCompletionMessageParam] = []

            if self.system_message is not None:
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

            tool = (
                pydantic_function_tool(self.response_type)
                if issubclass(self.response_type, BaseModel)
                and not self.structured_outputs
                else None
            )

            res = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.response_type
                if issubclass(self.response_type, BaseModel) and self.structured_outputs
                else NOT_GIVEN,
                tools=[tool] if tool is not None else NOT_GIVEN,
                tool_choice={
                    "type": "function",
                    "function": {"name": tool["function"]["name"]},
                }
                if tool is not None
                else NOT_GIVEN,
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

            if (
                issubclass(self.response_type, BaseModel)
                and (parsed := res.choices[0].message.parsed) is not None
            ):
                return parsed

            elif (
                issubclass(self.response_type, BaseModel)
                and (tool_calls := res.choices[0].message.tool_calls) is not None
                and (parsed := tool_calls[0].function.parsed_arguments) is not None
            ):
                return cast(R, parsed)

            elif (
                issubclass(self.response_type, str)
                and (content := res.choices[0].message.content) is not None
            ):
                return cast(R, content)

            raise ValueError("The completion is empty")
