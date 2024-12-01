from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from textwrap import dedent
from typing import Literal, Protocol, override

import orjson
from pydantic import BaseModel

from cbrkit.helpers import GenerationSingleWrapper, SimPairWrapper, unpack_sim

from ..typing import (
    AdaptMapFunc,
    AdaptPairFunc,
    AnyGenerationFunc,
    AnySimFunc,
    Casebase,
    Float,
)

__all__ = [
    "pipe",
    "null",
    "genai",
    "GenaiModel",
    "GenaiPydanticModel",
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


def default_prompt_template[K, V](
    prompt: str,
    casebase: Casebase[K, V],
    query: V,
) -> str:
    result = dedent(f"""
        {prompt}

        ## Query

        ```json
        {str(orjson.dumps(query))}
        ```

        ## Retrieved Cases
    """)

    for key, value in casebase.items():
        result += dedent(f"""
            ### {str(key)}

            ```json
            {str(orjson.dumps(value))}
            ```
        """)

    return result


class GenaiModel[K, V](Protocol):
    casebase: Mapping[K, V]


class GenaiPydanticModel[K, V](BaseModel):
    casebase: Mapping[K, V]


@dataclass(slots=True, frozen=True)
class genai[K, V: BaseModel](AdaptMapFunc[K, V]):
    generation_func: AnyGenerationFunc[GenaiModel[K, V]]
    prompt: str
    prompt_template: Callable[[str, Casebase[K, V], V], str] = default_prompt_template

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]:
        generation_func = GenerationSingleWrapper(self.generation_func)
        prompt = self.prompt_template(self.prompt, casebase, query)
        generation_result = generation_func(prompt)

        return generation_result.casebase
