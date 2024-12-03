from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from ..rag._build import transpose
from ..typing import (
    AdaptationMapFunc,
    Casebase,
    Float,
    RagFunc,
)

__all__ = [
    "structured_response",
    "StructuredModel",
]


class StructuredModel[K, V](BaseModel):
    casebase: Mapping[K, V]


def _from_pydantic[K, V](obj: StructuredModel[K, V]) -> Mapping[K, V]:
    return obj.casebase


@dataclass(slots=True, frozen=True)
class structured_response[K, V](AdaptationMapFunc[K, V]):
    rag_func: RagFunc[StructuredModel[K, V], K, V, Float]

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]:
        func = transpose(
            self.rag_func,
            conversion_func=_from_pydantic,
        )
        return func([(casebase, query, None)])[0]
