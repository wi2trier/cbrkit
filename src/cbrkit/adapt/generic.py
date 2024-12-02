from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, override

from ..helpers import unbatchify_sim, unpack_float
from ..typing import (
    AdaptationFunc,
    AnySimFunc,
    Float,
)

__all__ = [
    "pipe",
    "null",
]


@dataclass(slots=True, frozen=True)
class pipe[V](AdaptationFunc[V]):
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

    functions: AdaptationFunc[V] | list[AdaptationFunc[V]]
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
            similarity_func = unbatchify_sim(self.similarity_func)
            current_similarity = similarity_func(current_case, query)

        for func in functions:
            adapted_case = func(current_case, query)

            if similarity_func is not None and current_similarity is not None:
                adapted_similarity = similarity_func(current_case, adapted_case)

                if (
                    unpack_float(adapted_similarity)
                    >= unpack_float(current_similarity) + self.similarity_delta
                ):
                    current_case = adapted_case
                    current_similarity = adapted_similarity
                elif self.strategy == "break":
                    break

            else:
                current_case = adapted_case

        return current_case


@dataclass(slots=True, frozen=True)
class null[V](AdaptationFunc[V]):
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
