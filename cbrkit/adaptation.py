from dataclasses import dataclass

from cbrkit.typing import AdaptationFunc, Casebase, Float


@dataclass(slots=True, frozen=True)
class Result[V, S: Float]:
    case: V
    similarity: S


def apply[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    adaptation_funcs: AdaptationFunc[K, V, S],
) -> Result[V, S]:
    raise NotImplementedError()
