from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider, Response


def pydantic_to_anthropic_schema(model: type[BaseModel], description: str = "") -> dict:
    """
    Convert a Pydantic model to an Anthropic-compatible JSON schema format.

    Args:
        model: The Pydantic model class to convert
        description: Optional description of the function

    Returns:
        Tuple containing name of the tool and the JSON schema
    """
    # Get the JSON schema from Pydantic model
    schema = model.model_json_schema()

    # Extract the model name
    name = model.__name__

    # Create the Anthropic-compatible schema structure
    anthropic_schema = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        },
    }

    return anthropic_schema


with optional_dependencies():
    from anthropic import NOT_GIVEN, AsyncAnthropic, NotGiven
    from anthropic.types import (
        MessageParam,
        MetadataParam,
        ModelParam,
        TextBlockParam,
        ToolChoiceParam,
        ToolParam,
    )
    from httpx import Timeout

    type AnthropicPrompt = str | ChatPrompt[str]

    @dataclass(slots=True)
    class anthropic[R: str | BaseModel](ChatProvider[AnthropicPrompt, R]):
        max_tokens: int
        model: ModelParam
        client: AsyncAnthropic = field(default_factory=AsyncAnthropic, repr=False)
        metadata: MetadataParam | NotGiven = NOT_GIVEN
        stop_sequences: list[str] | NotGiven = NOT_GIVEN
        stream: NotGiven | Literal[False] = NOT_GIVEN
        system: str | Iterable[TextBlockParam] | NotGiven = NOT_GIVEN
        temperature: float | NotGiven = NOT_GIVEN
        tool_choice: ToolChoiceParam | NotGiven = NOT_GIVEN
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN
        top_k: int | NotGiven = NOT_GIVEN
        top_p: float | NotGiven = NOT_GIVEN
        extra_headers: Any | None = None
        extra_query: Any | None = None
        extra_body: Any | None = None
        timeout: float | Timeout | NotGiven | None = NOT_GIVEN

        @override
        async def __call_batch__(self, prompt: AnthropicPrompt) -> Response[R]:
            messages: list[MessageParam] = []

            if self.system_message is not None:
                messages.append(
                    {
                        "role": "user",  # anthropic doesn't have a "system" role
                        "content": self.system_message,
                    }
                )
            messages.extend(cast(Sequence[MessageParam], self.messages))

            if isinstance(prompt, ChatPrompt):
                messages.extend(cast(Sequence[MessageParam], prompt.messages))

            if self.messages and self.messages[-1].role == "user":
                messages.append({"role": "assistant", "content": unpack_value(prompt)})
            else:
                messages.append({"role": "user", "content": unpack_value(prompt)})

            tool = (
                pydantic_to_anthropic_schema(self.response_type)
                if issubclass(self.response_type, BaseModel)
                else None
            )

            toolchoice = (
                cast(ToolChoiceParam, {"type": "tool", "name": tool["name"]})
                if tool is not None
                else None
            )
            tool = cast(ToolParam, tool) if tool is not None else None
            res = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                tools=[tool] if tool is not None else NOT_GIVEN,
                tool_choice=toolchoice if toolchoice is not None else NOT_GIVEN,
            )
            if issubclass(self.response_type, BaseModel):
                # res.content should contain one ToolUseBlock
                if len(res.content) != 1:
                    raise ValueError("Expected one ToolUseBlock, got", len(res.content))
                block = res.content[0]
                if block.type != "tool_use":
                    raise ValueError("Expected one ToolUseBlock, got", block.type)
                return Response(self.response_type.model_validate(block.input))

            elif self.response_type is str:
                str_res = ""
                for block in res.content:
                    if block.type == "text":
                        str_res += block.text
                return Response(cast(R, str_res))

            raise ValueError("Invalid response", res)
