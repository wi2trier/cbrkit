import json
from dataclasses import dataclass, field
from typing import Any, Type, cast, override, Iterable, List, Literal, Sequence

from pydantic import BaseModel

from ...helpers import optional_dependencies, unpack_value
from .model import ChatPrompt, ChatProvider

def pydantic_to_anthropic_schema(model: Type[BaseModel], description: str = "") -> tuple[str, str]:
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
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "title": schema.get("title", name)
        }
    }

    # Add required fields if they exist
    if "required" in schema:
        anthropic_schema["parameters"]["required"] = schema["required"]

    return name, json.dumps(anthropic_schema, indent=2)

with optional_dependencies():
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, ModelParam, MetadataParam, TextBlockParam, ToolChoiceParam, ToolParam, ToolUseBlock
    from anthropic._types import NOT_GIVEN, Headers, Query, Body, NotGiven
    from httpx import Timeout

    type AnthropicPrompt = str | ChatPrompt[str]

    @dataclass(slots=True, frozen=True)
    class anthropic[R: str | BaseModel](ChatProvider[AnthropicPrompt, R]):
        max_tokens: int
        model: ModelParam
        client: AsyncAnthropic = field(default_factory=AsyncAnthropic, repr=False)
        metadata: MetadataParam | NotGiven = NOT_GIVEN
        stop_sequences: List[str] | NotGiven = NOT_GIVEN
        stream: NotGiven | Literal[False] = NOT_GIVEN
        system: str | Iterable[TextBlockParam] | NotGiven = NOT_GIVEN
        temperature: float | NotGiven = NOT_GIVEN
        tool_choice: ToolChoiceParam | NotGiven = NOT_GIVEN
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN
        top_k: int | NotGiven = NOT_GIVEN
        top_p: float | NotGiven = NOT_GIVEN
        extra_headers: Headers | None = None
        extra_query: Query | None = None
        extra_body: Body | None = None
        timeout: float | Timeout | NotGiven | None = NOT_GIVEN

        @override
        async def __call_batch__(self, prompt: AnthropicPrompt) -> R:
            messages: list[MessageParam] = []

            if self.system_message is not None:
                messages.append({
                    "role": "user", # anthropic doesn't have a "system" role
                    "content": self.system_message,
                })
            messages.extend(cast(Sequence[MessageParam], self.messages))

            if isinstance(prompt, ChatPrompt):
                messages.extend(
                    cast(Sequence[MessageParam], prompt.messages)
                )

            if self.messages and self.messages[-1]["role"] == "user":
                messages.append({"role": "assistant", "content": unpack_value(prompt)})
            else:
                messages.append({"role": "user", "content": unpack_value(prompt)})

            toolname, tool = cast(ToolParam, pydantic_to_anthropic_schema(self.response_type)) if issubclass(self.response_type, BaseModel) else None, None

            toolchoice = cast(ToolChoiceParam, {"type": "tool", "name": toolname})

            res = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                tools=[tool] if tool is not None else NOT_GIVEN,
                tool_choice= toolchoice if toolname is not None else NOT_GIVEN,
            )
            if issubclass(self.response_type, BaseModel):
                # res.content should contain one ToolUseBlock
                if len(res.content) != 1:
                    raise ValueError("Expected one ToolUseBlock, got", len(res.content))
                block = res.content[0]
                if block.type != "tool_use":
                    raise ValueError("Expected one ToolUseBlock, got", block.type)
                return self.response_type.model_validate(block.input)
            
            elif self.response_type is str:
                str_res = ""
                for block in res.content:
                    if block.type == "text":
                        str_res += block.text
                return cast(R, str_res)

            raise ValueError("Problem with response type", res.content)

