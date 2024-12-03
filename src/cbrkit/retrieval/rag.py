from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..rag._build import transpose
from ..typing import (
    Casebase,
    RagFunc,
    RetrieverFunc,
)

__all__ = [
    "structured_response",
    "StructuredResponse",
]


class StructuredResponse[K](BaseModel):
    similarities: Mapping[K, float]


def _from_pydantic[K](obj: StructuredResponse[K]) -> Mapping[K, float]:
    return obj.similarities


@dataclass(slots=True, frozen=True)
class structured_response[K, V](RetrieverFunc[K, V, float]):
    rag_func: RagFunc[StructuredResponse[K], K, V, Any]

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[Casebase[K, float]]:
        func = transpose(
            self.rag_func,
            conversion_func=_from_pydantic,
        )
        return func([(casebase, query, None) for casebase, query in batches])
