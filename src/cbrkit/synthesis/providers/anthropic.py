from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies
from .model import BaseProvider, Response, Usage

with optional_dependencies():
    from anthropic import NOT_GIVEN, AsyncAnthropic, NotGiven, Omit, omit
    from anthropic.types import (
        MessageParam,
        MetadataParam,
        ModelParam,
        ToolChoiceParam,
        ToolUnionParam,
    )
    from httpx2 import Timeout

    type AnthropicPrompt = str | Sequence[MessageParam]

    def if_given[T](value: T | None | Omit) -> T | Omit:
        """Return the value if not None, otherwise return the Anthropic omit sentinel."""
        return value if value is not None else omit

    @dataclass(slots=True)
    class anthropic[R: str | BaseModel](BaseProvider[AnthropicPrompt, R]):
        """Provider that calls Anthropic's messages API."""

        model: ModelParam
        max_tokens: int
        messages: Sequence[MessageParam] = field(default_factory=tuple)
        client: AsyncAnthropic = field(default_factory=AsyncAnthropic, repr=False)
        metadata: MetadataParam | None = None
        stop_sequences: list[str] | None = None
        tool_choice: ToolChoiceParam | None = None
        tools: Iterable[ToolUnionParam] | None = None
        extra_headers: Any | None = None
        extra_query: Any | None = None
        extra_body: Any | None = None
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN

        @override
        async def __call_batch__(self, prompt: AnthropicPrompt) -> Response[R]:
            messages: list[MessageParam] = []

            messages.extend(self.messages)

            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            else:
                messages.extend(prompt)

            output_format = (
                cast(Any, self.response_type)
                if issubclass(self.response_type, BaseModel)
                else omit
            )
            res = await self.client.messages.parse(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                output_format=output_format,
                metadata=if_given(self.metadata),
                stop_sequences=if_given(self.stop_sequences),
                system=if_given(self.system_message),
                tool_choice=if_given(self.tool_choice),
                tools=if_given(self.tools),
                extra_headers=self.extra_headers,
                extra_query=self.extra_query,
                extra_body=self.extra_body,
                timeout=self.timeout,
                **self.extra_kwargs,
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
                aggregated_content = "".join(
                    getattr(block, "text", "") for block in res.content
                )
                return Response(cast(R, aggregated_content), usage)

            raise ValueError("Invalid response", res)
