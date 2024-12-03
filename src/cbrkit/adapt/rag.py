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
    "build",
    "PydanticModel",
    "unpack_pydantic_model",
]


class PydanticModel[K, V](BaseModel):
    casebase: Mapping[K, V]


def _unpack_pydantic_model[K, V](obj: PydanticModel[K, V]) -> Mapping[K, V]:
    return obj.casebase


def unpack_pydantic_model[K, V, S: Float](
    rag_func: RagFunc[PydanticModel[K, V], K, V, S],
) -> RagFunc[Mapping[K, V], K, V, S]:
    return transpose(
        rag_func,
        conversion_func=_unpack_pydantic_model,
    )


@dataclass(slots=True, frozen=True)
class build[K, V](AdaptationMapFunc[K, V]):
    rag_func: RagFunc[Mapping[K, V], K, V, Float]

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]:
        return self.rag_func([(casebase, query, None)])[0]
