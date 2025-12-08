from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, override

from pydantic import BaseModel

from ...helpers import get_logger, optional_dependencies
from .model import BaseProvider, Response

logger = get_logger(__name__)

with optional_dependencies():
    from instructor import AsyncInstructor
    from openai.types.chat import ChatCompletionMessageParam

    type InstructorPrompt = str | Sequence[ChatCompletionMessageParam]

    @dataclass(slots=True)
    class instructor[R: BaseModel](BaseProvider[InstructorPrompt, R]):
        client: AsyncInstructor = field(repr=False)
        messages: Sequence[ChatCompletionMessageParam] = field(default_factory=tuple)
        strict: bool = True
        context: dict[str, Any] | None = None

        @override
        async def __call_batch__(self, prompt: InstructorPrompt) -> Response[R]:
            messages: list[ChatCompletionMessageParam] = []

            if self.system_message is not None:
                messages.append({"role": "system", "content": self.system_message})

            messages.extend(self.messages)

            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            else:
                messages.extend(prompt)

            # retries are already handled by the base provider
            return Response(
                await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=self.response_type,
                    context=self.context,
                    **self.extra_kwargs,
                )
            )
