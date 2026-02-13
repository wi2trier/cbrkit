from collections.abc import Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, override

from pydantic import BaseModel, ValidationError

from ...helpers import get_logger, optional_dependencies
from .model import BaseProvider, Response, Usage

logger = get_logger(__name__)

with optional_dependencies():
    from httpx import Timeout
    from openai import AsyncOpenAI, Omit, omit, pydantic_function_tool
    from openai.types.responses import (
        ResponseIncludable,
        ResponseTextConfigParam,
        ToolChoiceFunctionParam,
    )
    from openai.types.responses.response_create_params import ToolChoice
    from openai.types.responses.response_input_item_param import ResponseInputItemParam
    from openai.types.responses.response_input_param import ResponseInputParam
    from openai.types.responses.tool_param import ParseableToolParam

    type OpenAiResponsesPrompt = str | ResponseInputParam

    def if_given[T](value: T | None | Omit) -> T | Omit:
        return value if value is not None else omit

    @dataclass(slots=True)
    class openai_responses[R: BaseModel | str](BaseProvider[OpenAiResponsesPrompt, R]):
        """Provider that calls the OpenAI Responses API and parses structured outputs."""

        input_items: Sequence[ResponseInputItemParam] = field(default_factory=tuple)
        tool_choice: type[BaseModel] | str | None = None
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)
        include: Sequence[ResponseIncludable] | None = None
        max_output_tokens: int | None = None
        max_tool_calls: int | None = None
        metadata: dict[str, str] | None = None
        parallel_tool_calls: bool | None = None
        store: bool | None = None
        temperature: float | None = None
        top_logprobs: int | None = None
        top_p: float | None = None
        text: ResponseTextConfigParam | None = None
        extra_headers: Any | None = None
        extra_query: Any | None = None
        extra_body: Any | None = None
        timeout: float | Timeout | None = None

        @override
        async def __call_batch__(self, prompt: OpenAiResponsesPrompt) -> Response[R]:
            inputs: list[ResponseInputItemParam] = []

            if self.system_message is not None:
                inputs.append({"role": "system", "content": self.system_message})

            inputs.extend(self.input_items)

            if isinstance(prompt, str):
                inputs.append({"role": "user", "content": prompt})
            else:
                inputs.extend(prompt)

            tools: list[ParseableToolParam] | None = None
            tool_choice: ToolChoice | None = None
            text_format: type[BaseModel] | Omit = omit

            response_type_origin = get_origin(self.response_type)

            if response_type_origin is UnionType or response_type_origin is Union:
                tools = [
                    cast(ParseableToolParam, pydantic_function_tool(tool))
                    for tool in get_args(self.response_type)
                    if issubclass(tool, BaseModel)
                ]
            elif issubclass(self.response_type, BaseModel):
                if self.tool_choice is not None:
                    tools = [
                        cast(
                            ParseableToolParam,
                            pydantic_function_tool(self.response_type),
                        )
                    ]
                else:
                    text_format = self.response_type

            if self.tool_choice is not None:
                tool_choice = ToolChoiceFunctionParam(
                    name=self.tool_choice
                    if isinstance(self.tool_choice, str)
                    else self.response_type.__name__,
                    type="function",
                )

            text_param: ResponseTextConfigParam | Omit

            if self.text is None:
                text_param = omit
            elif text_format is not omit and "format" in self.text:
                raise ValueError(
                    "`text.format` cannot be set when using structured outputs."
                )
            else:
                text_param = self.text

            try:
                res = await self.client.responses.parse(
                    model=self.model,
                    input=inputs,
                    instructions=if_given(self.system_message),
                    include=if_given(
                        list(self.include) if self.include is not None else None
                    ),
                    tools=if_given(tools),
                    tool_choice=if_given(tool_choice),
                    max_output_tokens=if_given(self.max_output_tokens),
                    max_tool_calls=if_given(self.max_tool_calls),
                    metadata=if_given(self.metadata),
                    parallel_tool_calls=if_given(self.parallel_tool_calls),
                    store=if_given(self.store),
                    temperature=if_given(self.temperature),
                    top_logprobs=if_given(self.top_logprobs),
                    top_p=if_given(self.top_p),
                    text=text_param,
                    text_format=text_format,  # type: ignore[arg-type]
                    extra_headers=self.extra_headers,
                    extra_query=self.extra_query,
                    extra_body=self.extra_body,
                    timeout=self.timeout,
                    **self.extra_kwargs,
                )
            except ValidationError as e:
                for error in e.errors():
                    logger.error(f"Invalid response ({error['msg']}): {error['input']}")
                raise

            if res.incomplete_details is not None:
                raise ValueError(
                    res.incomplete_details.reason or "Response incomplete", res
                )

            for output in res.output:
                if hasattr(output, "content") and output.content is not None:
                    for content in output.content:  # type: ignore[union-attr]
                        if content.type == "refusal":
                            raise ValueError("Refusal", res)

            assert res.usage is not None
            usage = Usage(res.usage.input_tokens, res.usage.output_tokens)

            if tools is not None:
                for output in res.output:
                    if output.type == "function_call":
                        parsed_arguments = getattr(output, "parsed_arguments", None)

                        if parsed_arguments is not None:
                            return Response(cast(R, parsed_arguments), usage)

                raise ValueError("Invalid response", res)

            if text_format is not omit and (parsed := res.output_parsed) is not None:
                return Response(cast(R, parsed), usage)

            if issubclass(self.response_type, str):
                content = res.output_text

                if content:
                    return Response(cast(R, content), usage)

            raise ValueError("Invalid response", res)
