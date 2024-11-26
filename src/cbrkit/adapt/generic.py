from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, override

from ..typing import AdaptPairFunc

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

    functions: list[AdaptPairFunc[V]]

    @override
    def __call__(self, case: V, query: V) -> V:
        current_case = case

        for func in self.functions:
            current_case = func(current_case, query)

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
