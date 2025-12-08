from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies
from .model import BaseProvider, Response

with optional_dependencies():
    from cohere import (
        AsyncClient,
        ChatMessageV2,
        CitationOptions,
        JsonObjectResponseFormatV2,
        SystemChatMessageV2,
        UserChatMessageV2,
        V2ChatRequestDocumentsItem,
        V2ChatRequestSafetyMode,
    )
    from cohere.core import RequestOptions

    @dataclass(slots=True)
    class CohereDocumentsPrompt:
        messages: Sequence[ChatMessageV2]
        documents: Sequence[V2ChatRequestDocumentsItem]

    type CoherePrompt = str | Sequence[ChatMessageV2] | CohereDocumentsPrompt

    @dataclass(slots=True)
    class cohere[R: str | BaseModel](BaseProvider[CoherePrompt, R]):
        messages: Sequence[ChatMessageV2] = field(default_factory=tuple)
        documents: Sequence[V2ChatRequestDocumentsItem] = field(default_factory=tuple)
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
        k: int | None = None
        p: float | None = None
        logprobs: bool | None = None

        @override
        async def __call_batch__(self, prompt: CoherePrompt) -> Response[R]:
            documents: list[V2ChatRequestDocumentsItem] = list(self.documents)

            if isinstance(prompt, CohereDocumentsPrompt):
                documents.extend(prompt.documents)

            if issubclass(self.response_type, BaseModel) and documents:
                raise ValueError(
                    "Structured output format is not supported when using documents"
                )

            messages: list[ChatMessageV2] = []

            if self.system_message is not None:
                messages.append(SystemChatMessageV2(content=self.system_message))

            if isinstance(prompt, str):
                messages.append(UserChatMessageV2(content=prompt))
            elif isinstance(prompt, CohereDocumentsPrompt):
                messages.extend(prompt.messages)
            else:
                messages.extend(prompt)

            res = await self.client.v2.chat(
                model=self.model,
                messages=messages,
                request_options=self.request_options,
                documents=documents if documents else None,
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
                if len(content) != 1 or content[0].type != "text":
                    raise ValueError(
                        "The completion is empty, has multiple outputs, or is not text"
                    )

                return Response(self.response_type.model_validate_json(content[0].text))

            aggregated_content = "".join(
                block.text for block in content if block.type == "text"
            )

            return Response(cast(R, aggregated_content))
