from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..rag._build import transpose
from ..typing import (
    Casebase,
    Float,
    RagFunc,
    RetrieverFunc,
)

__all__ = [
    "build",
    "PydanticModel",
    "unpack_pydantic_model",
]


class PydanticModel[K](BaseModel):
    similarities: Mapping[K, float]


def _unpack_pydantic_model[K](obj: PydanticModel[K]) -> Mapping[K, float]:
    return obj.similarities


def unpack_pydantic_model[K, V, S: Float](
    rag_func: RagFunc[PydanticModel[K], K, V, S],
) -> RagFunc[Mapping[K, float], K, V, S]:
    return transpose(
        rag_func,
        conversion_func=_unpack_pydantic_model,
    )


@dataclass(slots=True, frozen=True)
class build[K, V](RetrieverFunc[K, V, float]):
    rag_func: RagFunc[Mapping[K, float], K, V, Any]

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[Casebase[K, float]]:
        return self.rag_func([(casebase, query, None) for casebase, query in batches])
