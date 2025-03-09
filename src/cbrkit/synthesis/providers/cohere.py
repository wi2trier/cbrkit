from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider, DocumentsPrompt, Response

with optional_dependencies():
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

    type CoherePrompt = str | ChatPrompt[str] | DocumentsPrompt[str]

    @dataclass(slots=True)
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
        async def __call_batch__(self, prompt: CoherePrompt) -> Response[R]:
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
                    UserChatMessageV2(content=msg.content)
                    if msg.role == "user"
                    else AssistantChatMessageV2(content=msg.content)
                    for msg in prompt.messages
                )

            if self.messages and self.messages[-1].role == "user":
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

                return Response(self.response_type.model_validate_json(content[0].text))

            return Response(cast(R, "\n".join(x.text for x in content)))
