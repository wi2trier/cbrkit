from collections.abc import Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin, override

from pydantic import BaseModel, ValidationError

from ...helpers import get_logger, optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider, Response, Usage

logger = get_logger(__name__)

with optional_dependencies():
    from httpx import Timeout
    from openai import NOT_GIVEN, AsyncOpenAI, NotGiven, pydantic_function_tool
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionNamedToolChoiceParam,
        ChatCompletionToolParam,
    )

    type OpenaiPrompt = str | ChatPrompt[str]

    def if_given[T](value: T | None) -> T | NotGiven:
        return value if value is not None else NOT_GIVEN

    @dataclass(slots=True)
    class openai[R: BaseModel | str](ChatProvider[OpenaiPrompt, R]):
        tool_choice: type[BaseModel] | str | None = None
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)
        frequency_penalty: float | None = None
        logit_bias: dict[str, int] | None = None
        logprobs: bool | None = None
        max_completion_tokens: int | None = None
        metadata: dict[str, str] | None = None
        n: int | None = None
        presence_penalty: float | None = None
        seed: int | None = None
        stop: str | list[str] | None = None
        store: bool | None = None
        reasoning_effort: Literal["low", "medium", "high"] | None = None
        temperature: float | None = None
        top_logprobs: int | None = None
        top_p: float | None = None
        extra_headers: Any | None = None
        extra_query: Any | None = None
        extra_body: Any | None = None
        timeout: float | Timeout | None = None

        @override
        async def __call_batch__(self, prompt: OpenaiPrompt) -> Response[R]:
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

            if messages and messages[-1]["role"] == "user":
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

            tools: list[ChatCompletionToolParam] | None = None
            tool_choice: ChatCompletionNamedToolChoiceParam | None = None
            response_type_origin = get_origin(self.response_type)

            if response_type_origin is UnionType or response_type_origin is Union:
                tools = [
                    pydantic_function_tool(tool)
                    for tool in get_args(self.response_type)
                    if issubclass(tool, BaseModel)
                ]
            elif (
                issubclass(self.response_type, BaseModel)
                and self.tool_choice is not None
            ):
                tools = [pydantic_function_tool(self.response_type)]

            if self.tool_choice is not None:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": self.tool_choice
                        if isinstance(self.tool_choice, str)
                        else self.tool_choice.__name__,
                    },
                }

            try:
                res = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=self.response_type
                    if tools is None and issubclass(self.response_type, BaseModel)
                    else NOT_GIVEN,
                    tools=if_given(tools),
                    tool_choice=if_given(tool_choice),
                    frequency_penalty=if_given(self.frequency_penalty),
                    logit_bias=if_given(self.logit_bias),
                    logprobs=if_given(self.logprobs),
                    max_completion_tokens=if_given(self.max_completion_tokens),
                    metadata=if_given(self.metadata),
                    n=if_given(self.n),
                    presence_penalty=if_given(self.presence_penalty),
                    seed=if_given(self.seed),
                    stop=if_given(self.stop),
                    store=if_given(self.store),
                    reasoning_effort=if_given(self.reasoning_effort),
                    temperature=if_given(self.temperature),
                    top_logprobs=if_given(self.top_logprobs),
                    top_p=if_given(self.top_p),
                    extra_headers=self.extra_headers,
                    extra_query=self.extra_query,
                    extra_body=self.extra_body,
                    timeout=if_given(self.timeout),
                    **self.extra_kwargs,
                )
            except ValidationError as e:
                for error in e.errors():
                    logger.error(f"Invalid response ({error['msg']}): {error['input']}")
                raise

            choice = res.choices[0]
            message = choice.message

            assert res.usage is not None
            usage = Usage(res.usage.prompt_tokens, res.usage.completion_tokens)

            if choice.finish_reason == "length":
                raise ValueError("Length limit", res)

            if choice.finish_reason == "content_filter":
                raise ValueError("Content filter", res)

            if message.refusal:
                raise ValueError("Refusal", res)

            if (
                isinstance(self.response_type, type)
                and issubclass(self.response_type, BaseModel)
                and (parsed := message.parsed) is not None
            ):
                return Response(cast(R, parsed), usage)

            if (
                isinstance(self.response_type, type)
                and issubclass(self.response_type, str)
                and (content := message.content) is not None
            ):
                return Response(cast(R, content), usage)

            if (
                tools is not None
                and (tool_calls := message.tool_calls) is not None
                and (parsed := tool_calls[0].function.parsed_arguments) is not None
            ):
                return Response(cast(R, parsed), usage)

            raise ValueError("Invalid response", res)
