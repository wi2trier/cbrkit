import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import override

from pydantic import BaseModel

from ...helpers import optional_dependencies
from .model import BaseProvider, Response

with optional_dependencies():
    from ollama import AsyncClient, Message, Options

    type OllamaPrompt = str | Sequence[Message]

    @dataclass(slots=True)
    class ollama[R: str | BaseModel](BaseProvider[OllamaPrompt, R]):
        """Provider that calls Ollama's chat API."""

        client: AsyncClient = field(default_factory=AsyncClient, repr=False)
        messages: Sequence[Message] = field(default_factory=tuple)
        options: Options | None = None
        keep_alive: float | str | None = None

        @override
        async def __call_batch__(self, prompt: OllamaPrompt) -> Response[R]:
            messages: list[Message] = []

            if self.system_message is not None:
                messages.append(Message(role="system", content=self.system_message))

            messages.extend(self.messages)

            if isinstance(prompt, str):
                messages.append(Message(role="user", content=prompt))
            else:
                messages.extend(prompt)

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
