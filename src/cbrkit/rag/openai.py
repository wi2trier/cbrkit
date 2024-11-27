import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import cast

import orjson
from pydantic import BaseModel

from cbrkit.helpers import similarities2ranking, unpack_sim

from ..typing import Casebase, Float, RagFunc, SimMap

__all__ = []

try:
    from openai import AsyncOpenAI, pydantic_function_tool
    from openai.types.chat import ChatCompletionMessageParam

    @dataclass(slots=True, frozen=True)
    class build[K, V, S: Float, T: BaseModel | str](RagFunc[K, V, S, T]):
        model: str
        prompt: str | Callable[[Casebase[K, V], V, SimMap[K, S]], str]
        schema: type[T]
        messages: Sequence[ChatCompletionMessageParam] = field(default_factory=list)
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)

        def __call__(
            self, pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S]]]
        ) -> Sequence[T]:
            return asyncio.run(self._generate(pairs))

        async def _generate(
            self, pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S]]]
        ) -> list[T]:
            return await asyncio.gather(
                *(self._generate_single(*pair) for pair in pairs)
            )

        async def _generate_single(
            self, casebase: Casebase[K, V], query: V, similarities: SimMap[K, S]
        ) -> T:
            if self.messages and self.messages[-1]["role"] == "user":
                raise ValueError("The last message cannot be from the user")

            ranking: list[K] = similarities2ranking(similarities)

            if isinstance(self.prompt, Callable):
                prompt = self.prompt(casebase, query, similarities)
            else:
                prompt = dedent(f"""
                    {self.prompt}

                    ## Query

                    ```json
                    {str(orjson.dumps(query))}
                    ```

                    ## Retrieved Cases
                """)

                for rank, key in enumerate(ranking, start=1):
                    prompt += dedent(f"""
                        ### {rank}. {key} (Similarity: {unpack_sim(similarities[key])})

                        ```json
                        {str(orjson.dumps(casebase[key]))}
                        ```
                    """)

            messages: list[ChatCompletionMessageParam] = [
                *self.messages,
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

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

                return cast(T, parsed)

            elif self.schema is str:
                res = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                )

                content = res.choices[0].message.content

                if content is None:
                    raise ValueError("The completion is empty")

                return cast(T, content)

            raise ValueError(f"Unsupported schema type: {self.schema}")

    __all__ += ["build"]

except ImportError:
    pass
