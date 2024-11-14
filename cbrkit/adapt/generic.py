from dataclasses import dataclass
from typing import override

from cbrkit.helpers import get_metadata
from cbrkit.typing import AdaptPairFunc, JsonDict, SupportsMetadata

__all__ = [
    "pipe",
]


@dataclass(slots=True, frozen=True)
class pipe[V](AdaptPairFunc[V], SupportsMetadata):
    functions: list[AdaptPairFunc[V]]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "functions": [get_metadata(func) for func in self.functions],
        }

    @override
    def __call__(self, case: V, query: V) -> V:
        current_case = case

        for func in self.functions:
            current_case = func(current_case, query)

        return current_case
