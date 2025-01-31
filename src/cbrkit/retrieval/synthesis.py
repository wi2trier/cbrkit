from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..synthesis.build import transpose
from ..typing import (
    Casebase,
    RetrieverFunc,
    SynthesizerFunc,
)

# TODO: return ranking that is transformed to a similarity map
# todo: compute mean average error between similarities from llm and benchmark


class SynthesisResponse[K](BaseModel):
    similarities: Mapping[K, float]


def _from_pydantic[K](obj: SynthesisResponse[K]) -> Mapping[K, float]:
    return obj.similarities


@dataclass(slots=True, frozen=True)
class synthesis[K, V](RetrieverFunc[K, V, float]):
    func: SynthesizerFunc[SynthesisResponse[K], K, V, Any]

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[Casebase[K, float]]:
        func = transpose(
            self.func,
            conversion_func=_from_pydantic,
        )
        return func([(casebase, query, None) for casebase, query in batches])
