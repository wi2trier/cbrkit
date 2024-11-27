from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Literal, cast, override

import orjson
from pydantic import BaseModel

from cbrkit.helpers import SimPairWrapper, unpack_sim

from ..typing import AdaptMapFunc, AdaptPairFunc, AnySimFunc, Casebase, Float

__all__ = [
    "pipe",
    "null",
]


@dataclass(slots=True, frozen=True)
class pipe[V](AdaptPairFunc[V]):
    """Chain multiple adaptation functions together.

    Args:
        functions: List of adaptation functions to apply in order.

    Returns:
        The adapted value.

    Examples:
        >>> func = pipe([
        ...     lambda x, y: x + y,
        ...     lambda x, y: x * y,
        ... ])
        >>> func(2, 3)
        15
    """

    functions: AdaptPairFunc[V] | list[AdaptPairFunc[V]]
    similarity_func: AnySimFunc[V, Float] | None = None
    similarity_delta: float = 0.0
    strategy: Literal["continue", "break"] = "continue"

    @override
    def __call__(self, case: V, query: V) -> V:
        functions = (
            self.functions if isinstance(self.functions, list) else [self.functions]
        )
        current_case = case
        similarity_func = None
        current_similarity = None

        if self.similarity_func is not None:
            similarity_func = SimPairWrapper(self.similarity_func)
            current_similarity = similarity_func(current_case, query)

        for func in functions:
            adapted_case = func(current_case, query)

            if similarity_func is not None and current_similarity is not None:
                adapted_similarity = similarity_func(current_case, adapted_case)

                if (
                    unpack_sim(adapted_similarity)
                    >= unpack_sim(current_similarity) + self.similarity_delta
                ):
                    current_case = adapted_case
                    current_similarity = adapted_similarity
                elif self.strategy == "break":
                    break

            else:
                current_case = adapted_case

        return current_case


@dataclass(slots=True, frozen=True)
class null[V](AdaptPairFunc[V]):
    """Perform a null adaptation and return the original case or query value.

    Args:
        select: Either "case" or "query".
        copy: Whether to copy the value before returning it.

    Returns:
        The original case value.

    Examples:
        >>> func = null()
        >>> func(2, 3)
        2
    """

    target: Literal["case", "query"] = "case"
    copy: bool = False

    @override
    def __call__(self, case: V, query: V) -> V:
        value: V

        if self.target == "case":
            value = case
        elif self.target == "query":
            value = query
        else:
            raise ValueError(f"Invalid target value: {self.target}")

        if self.copy:
            value = deepcopy(value)

        return value


try:
    from openai import OpenAI, pydantic_function_tool
    from openai.types.chat import ChatCompletionMessageParam

    @dataclass(slots=True, frozen=True)
    class openai[V: BaseModel](AdaptMapFunc[str, V]):
        model: str
        prompt: str | Callable[[Casebase[str, V], V], str]
        schema: type[V]
        single_case: bool = False
        messages: Sequence[ChatCompletionMessageParam] = field(default_factory=list)
        client: OpenAI = field(default_factory=OpenAI, repr=False)

        def __call__(
            self,
            casebase: Casebase[str, V],
            query: V,
        ) -> Casebase[str, V]:
            schema = self.schema

            class CasebaseModel(BaseModel):
                casebase: dict[str, schema]

            if self.messages and self.messages[-1]["role"] == "user":
                raise ValueError("The last message cannot be from the user")

            if isinstance(self.prompt, Callable):
                prompt = self.prompt(casebase, query)

            else:
                prompt = dedent(f"""
                    {self.prompt}

                    ## Query

                    ```json
                    {str(orjson.dumps(query))}
                    ```

                    ## Retrieved Cases
                """)

                for key, value in casebase.items():
                    prompt += dedent(f"""
                        ### {key}

                        ```json
                        {str(orjson.dumps(value))}
                        ```
                    """)

            messages: list[ChatCompletionMessageParam] = [
                *self.messages,
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            tool = pydantic_function_tool(schema if self.single_case else CasebaseModel)

            res = self.client.beta.chat.completions.parse(
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

            if self.single_case:
                return {"default": cast(schema, parsed)}

            return cast(CasebaseModel, parsed).casebase

    __all__ += ["openai"]

except ImportError:
    pass
