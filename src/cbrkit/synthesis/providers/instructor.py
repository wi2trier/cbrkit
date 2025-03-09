from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast, override

from pydantic import BaseModel

from ...helpers import get_logger, optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider, Response

logger = get_logger(__name__)

with optional_dependencies():
    from instructor import AsyncInstructor
    from openai.types.chat import ChatCompletionMessageParam

    type InstructorPrompt = str | ChatPrompt[str]

    @dataclass(slots=True)
    class instructor[R: BaseModel](ChatProvider[InstructorPrompt, R]):
        client: AsyncInstructor = field(repr=False)
        strict: bool = True
        context: dict[str, Any] | None = None

        @override
        async def __call_batch__(self, prompt: InstructorPrompt) -> Response[R]:
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
