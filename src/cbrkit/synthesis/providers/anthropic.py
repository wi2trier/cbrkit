from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies
from .model import BaseProvider, Response, Usage

with optional_dependencies():
    from anthropic import AsyncAnthropic, Omit, omit
    from anthropic.types import (
        MetadataParam,
        ModelParam,
        TextBlockParam,
        ToolChoiceParam,
        ToolParam,
    )
    from anthropic.types.beta import BetaMessageParam
    from httpx import Timeout

    type AnthropicPrompt = str | Sequence[BetaMessageParam]

    def if_given[T](value: T | None | Omit) -> T | Omit:
        return value if value is not None else omit

    @dataclass(slots=True)
    class anthropic[R: str | BaseModel](BaseProvider[AnthropicPrompt, R]):
        model: ModelParam
        max_tokens: int
        messages: Sequence[BetaMessageParam] = field(default_factory=tuple)
        client: AsyncAnthropic = field(default_factory=AsyncAnthropic, repr=False)
        metadata: MetadataParam | None = None
        stop_sequences: list[str] | None = None
        system: str | Iterable[TextBlockParam] | None = None
        temperature: float | None = None
        tool_choice: ToolChoiceParam | None = None
        tools: Iterable[ToolParam] | None = None
        top_k: int | None = None
        top_p: float | None = None
        extra_headers: Any | None = None
        extra_query: Any | None = None
        extra_body: Any | None = None
        timeout: float | Timeout | None = None

        @override
        async def __call_batch__(self, prompt: AnthropicPrompt) -> Response[R]:
            messages: list[BetaMessageParam] = []

            if self.system_message is not None:
                # anthropic does not have a system/developer role
                messages.append({"role": "user", "content": self.system_message})

            messages.extend(self.messages)

            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            else:
                messages.extend(prompt)

            res = await self.client.beta.messages.parse(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                output_format=self.response_type  # type: ignore[arg-type]
                if issubclass(self.response_type, BaseModel)
                else omit,
            )

            usage = Usage(
                res.usage.input_tokens,
                res.usage.output_tokens,
            )

            if (
                isinstance(self.response_type, type)
                and issubclass(self.response_type, BaseModel)
                and (parsed := res.parsed_output) is not None
            ):
                return Response(parsed, usage)

            if (
                isinstance(self.response_type, type)
                and issubclass(self.response_type, str)
                and len(res.content) > 0
            ):
                aggregated_content = "".join(  # type: ignore[arg-type]
                    block.text for block in res.content if hasattr(block, "text")
                )
                return Response(cast(R, aggregated_content), usage)

            raise ValueError("Invalid response", res)
