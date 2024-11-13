from dataclasses import dataclass
from typing import override

from cbrkit.helpers import get_metadata
from cbrkit.typing import AdaptPairFunc, JsonDict, SupportsMetadata


@dataclass(slots=True, frozen=True)
class rules[V](AdaptPairFunc[V], SupportsMetadata):
    rules: list[AdaptPairFunc[V]]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "rules": [get_metadata(rule) for rule in self.rules],
        }

    @override
    def __call__(self, case: V, query: V) -> V:
        current_case = case

        for rule in self.rules:
            current_case = rule(current_case, query)

        return current_case
