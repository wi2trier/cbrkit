from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, override

from cbrkit.helpers import SimPairWrapper, unpack_sim

from ..typing import AdaptPairFunc, AnySimFunc, Float

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
    similarity_delta: float = -1.0

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
