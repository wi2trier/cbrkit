from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..synthesis.build import transpose
from ..typing import (
    Casebase,
    Float,
    MapAdaptationFunc,
    SynthesizerFunc,
)


class SynthesisResponse[K, V](BaseModel):
    model_config = ConfigDict(frozen=True)
    casebase: Mapping[K, V]


def _from_pydantic[K, V](obj: SynthesisResponse[K, V]) -> Mapping[K, V]:
    return obj.casebase


@dataclass(slots=True, frozen=True)
class synthesis[K, V](MapAdaptationFunc[K, V]):
    func: SynthesizerFunc[SynthesisResponse[K, V], K, V, Float]

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]:
        func = transpose(
            self.func,
            conversion_func=_from_pydantic,
        )
        return func([(casebase, query, None)])[0]
