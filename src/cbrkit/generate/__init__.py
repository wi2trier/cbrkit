import asyncio
from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel

from cbrkit.helpers import GenerationSeqWrapper

from ..typing import AnyGenerationFunc, GenerationSeqFunc

__all__ = [
    "transpose",
    "pipe",
]


@dataclass(slots=True, frozen=True)
class transpose[T, U](GenerationSeqFunc[T]):
    generation_func: AnyGenerationFunc[U]
    conversion_func: Callable[[U], T]

    def __call__(self, prompts: Sequence[str]) -> Sequence[T]:
        func = GenerationSeqWrapper(self.generation_func)
        return [self.conversion_func(output) for output in func(prompts)]


@dataclass(slots=True, frozen=True)
class pipe[T](GenerationSeqFunc[T]):
    generation_funcs: Sequence[AnyGenerationFunc[T]] | AnyGenerationFunc[T]
    prompt_template: Callable[[T], str]

    def __call__(self, prompts: Sequence[str]) -> Sequence[T]:
        funcs = (
            self.generation_funcs
            if isinstance(self.generation_funcs, Sequence)
            else [self.generation_funcs]
        )

        current_input = prompts
        current_output: Sequence[T] = []

        for func in funcs:
            wrapped_func = GenerationSeqWrapper(func)
            current_output = wrapped_func(current_input)
            current_input = [self.prompt_template(output) for output in current_output]

        if not len(current_output) == len(prompts):
            raise ValueError(
                "The number of outputs does not match the number of inputs, "
                "did you provie a generation function?"
            )

        return current_output


try:
    from openai import AsyncOpenAI, pydantic_function_tool
    from openai.types.chat import ChatCompletionMessageParam

    @dataclass(slots=True)
    class openai[T: BaseModel | str](GenerationSeqFunc[T]):
        model: str
        schema: type[T]
        messages: MutableSequence[ChatCompletionMessageParam] = field(
            default_factory=list
        )
        memorize: bool = False
        memorize_func: Callable[[T], str] = str
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)

        def __call__(self, prompts: Sequence[str]) -> Sequence[T]:
            return asyncio.run(self._generate(prompts))

        async def _generate(self, prompts: Sequence[str]) -> Sequence[T]:
            return await asyncio.gather(
                *(self._generate_single(prompt) for prompt in prompts)
            )

        async def _generate_single(self, prompt: str) -> T:
            if self.messages and self.messages[-1]["role"] == "user":
                raise ValueError("The last message cannot be from the user")

            messages: list[ChatCompletionMessageParam] = [
                *self.messages,
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            result: T | None = None

            if self.schema is BaseModel:
                tool = pydantic_function_tool(cast(type[BaseModel], self.schema))

                res = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    tools=[tool],
                    tool_choice={
                        "type": "function",
                        "function": {"name": tool["function"]["name"]},
                    },
                )

                tool_calls = res.choices[0].message.tool_calls

                if tool_calls is None:
                    raise ValueError("The completion is empty")

                parsed = tool_calls[0].function.parsed_arguments

                if parsed is None:
                    raise ValueError("The tool call is empty")

                result = cast(T, parsed)

            else:
                res = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                )

                content = res.choices[0].message.content

                if content is None:
                    raise ValueError("The completion is empty")

                result = cast(T, content)

            if self.memorize:
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append(
                    {"role": "system", "content": self.memorize_func(result)}
                )

            return result

    __all__ += ["openai"]

except ImportError:
    pass
