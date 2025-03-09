import json
from dataclasses import dataclass, field
from typing import override

from pydantic import BaseModel

from ...helpers import optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider, Response

with optional_dependencies():
    from ollama import AsyncClient, Message, Options

    type OllamaPrompt = str | ChatPrompt[str]

    @dataclass(slots=True)
    class ollama[R: str | BaseModel](ChatProvider[OllamaPrompt, R]):
        client: AsyncClient = field(default_factory=AsyncClient, repr=False)
        options: Options | None = None
        keep_alive: float | str | None = None

        @override
        async def __call_batch__(self, prompt: OllamaPrompt) -> Response[R]:
            messages: list[Message] = []

            if self.system_message is not None:
                messages.append(Message(role="system", content=self.system_message))

            messages.extend(
                Message(role=msg.role, content=msg.content) for msg in self.messages
            )

            if isinstance(prompt, ChatPrompt):
                messages.extend(
                    Message(role=msg.role, content=msg.content)
                    for msg in prompt.messages
                )

            if self.messages and self.messages[-1].role == "user":
                messages.append(Message(role="assistant", content=unpack_value(prompt)))
            else:
                messages.append(Message(role="user", content=unpack_value(prompt)))

            res = await self.client.chat(
                model=self.model,
                messages=messages,
                options=self.options,
                keep_alive=self.keep_alive,
                format=self.response_type.model_json_schema()
                if issubclass(self.response_type, BaseModel)
                else None,
                **self.extra_kwargs,
            )

            content = res["message"]["content"]

            if self.response_type is str:
                return Response(content)

            return Response(json.loads(content))
