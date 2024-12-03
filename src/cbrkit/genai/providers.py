import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, override

from pydantic import BaseModel

from ..helpers import unpack_value
from ..typing import BatchGenerationFunc
from ._model import ChatMessage, ChatPrompt, DocumentsPrompt

__all__ = []


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseProvider[P, R](BatchGenerationFunc[P, R], ABC):
    model: str
    response_type: type[R]
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, batches: Sequence[P]) -> Sequence[R]:
        return asyncio.run(self.__call_batches__(batches))

    async def __call_batches__(self, batches: Sequence[P]) -> Sequence[R]:
        return await asyncio.gather(*(self.__call_batch__(batch) for batch in batches))

    @abstractmethod
    async def __call_batch__(self, prompt: P) -> R: ...


@dataclass(slots=True, frozen=True, kw_only=True)
class ChatProvider[P, R](BaseProvider[P, R], ABC):
    system_message: str | None = None
    messages: Sequence[ChatMessage] = field(default_factory=tuple)


type OpenaiPrompt = str | ChatPrompt[str]

try:
    from openai import AsyncOpenAI, pydantic_function_tool
    from openai._types import NOT_GIVEN
    from openai.types.chat import ChatCompletionMessageParam

    @dataclass(slots=True, frozen=True)
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

    __all__ += ["openai"]

except ImportError:
    pass


type OllamaPrompt = str | ChatPrompt[str]

try:
    from ollama import AsyncClient, Message, Options

    @dataclass(slots=True, frozen=True)
    class ollama[R: str | object](ChatProvider[OllamaPrompt, R]):
        client: AsyncClient = field(default_factory=AsyncClient, repr=False)
        options: Options | None = None
        keep_alive: float | str | None = None

        @override
        async def __call_batch__(self, prompt: OllamaPrompt) -> R:
            messages: list[Message] = []

            if self.system_message is not None:
                messages.append(Message(role="system", content=self.system_message))

            messages.extend(Message(**msg) for msg in self.messages)

            if isinstance(prompt, ChatPrompt):
                messages.extend(Message(**msg) for msg in prompt.messages)

            if self.messages and self.messages[-1]["role"] == "user":
                messages.append(Message(role="assistant", content=unpack_value(prompt)))
            else:
                messages.append(Message(role="user", content=unpack_value(prompt)))

            res = await self.client.chat(
                model=self.model,
                messages=messages,
                options=self.options,
                keep_alive=self.keep_alive,
                format="" if self.response_type is str else "json",
                **self.extra_kwargs,
            )

            content = res["message"]["content"]

            if self.response_type is str:
                return content

            return json.loads(content)

    __all__ += ["ollama"]

except ImportError:
    pass


type CoherePrompt = str | ChatPrompt[str] | DocumentsPrompt[str]

try:
    from cohere import (
        AssistantChatMessageV2,
        AsyncClient,
        ChatMessageV2,
        CitationOptions,
        Document,
        JsonObjectResponseFormatV2,
        SystemChatMessageV2,
        UserChatMessageV2,
        V2ChatRequestSafetyMode,
    )
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere[R: str | BaseModel](ChatProvider[CoherePrompt, R]):
        client: AsyncClient = field(default_factory=AsyncClient, repr=False)
        request_options: RequestOptions | None = None
        citation_options: CitationOptions | None = None
        safety_mode: V2ChatRequestSafetyMode | None = None
        max_tokens: int | None = None
        stop_sequences: Sequence[str] | None = None
        temperature: float | None = None
        seed: int | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        k: float | None = None
        p: float | None = None
        logprobs: bool | None = None

        @override
        async def __call_batch__(self, prompt: CoherePrompt) -> R:
            if isinstance(prompt, DocumentsPrompt) and issubclass(
                self.response_type, BaseModel
            ):
                raise ValueError(
                    "Structured output format is not supported when using documents"
                )

            messages: list[ChatMessageV2] = []

            if self.system_message is not None:
                messages.append(SystemChatMessageV2(content=self.system_message))

            if isinstance(prompt, ChatPrompt):
                messages.extend(
                    UserChatMessageV2(content=msg["content"])
                    if msg["role"] == "user"
                    else AssistantChatMessageV2(content=msg["content"])
                    for msg in prompt.messages
                )

            if self.messages and self.messages[-1]["role"] == "user":
                messages.append(AssistantChatMessageV2(content=unpack_value(prompt)))
            else:
                messages.append(UserChatMessageV2(content=unpack_value(prompt)))

            res = await self.client.v2.chat(
                model=self.model,
                messages=messages,
                request_options=self.request_options,
                documents=[
                    Document(id=id, data=cast(dict[str, str], data))
                    for id, data in prompt.documents.items()
                ]
                if isinstance(prompt, DocumentsPrompt)
                else None,
                response_format=JsonObjectResponseFormatV2(
                    json_schema=self.response_type.model_json_schema()
                )
                if issubclass(self.response_type, BaseModel)
                else None,
                citation_options=self.citation_options,
                safety_mode=self.safety_mode,
                max_tokens=self.max_tokens,
                stop_sequences=self.stop_sequences,
                temperature=self.temperature,
                seed=self.seed,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                k=self.k,
                p=self.p,
                logprobs=self.logprobs,
                **self.extra_kwargs,
            )

            content = res.message.content

            if content is None:
                raise ValueError("The completion is empty")

            if issubclass(self.response_type, BaseModel):
                if len(content) != 1:
                    raise ValueError("The completion is empty or has multiple outputs")

                return self.response_type.model_validate_json(content[0].text)

            return cast(R, "\n".join(x.text for x in content))

    __all__ += ["cohere"]

except ImportError:
    pass
