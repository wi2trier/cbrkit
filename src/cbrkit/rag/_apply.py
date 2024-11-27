from collections.abc import Mapping

from ..model import Result, ResultStep
from ..typing import Float, RagFunc


def apply_result[Q, C, V, S: Float, T](
    result: Result[Q, C, V, S] | ResultStep[Q, C, V, S],
    rag_func: RagFunc[C, V, S, T],
) -> dict[Q, T]:
    return apply_pairs(
        {
            key: (entry.casebase, entry.query, entry.similarities)
            for key, entry in result.queries.items()
        },
        rag_func,
    )


def apply_pairs[Q, C, V, S: Float, T](
    pairs: Mapping[Q, tuple[Mapping[C, V], V, Mapping[C, S]]],
    rag_func: RagFunc[C, V, S, T],
) -> dict[Q, T]:
    rag_results = rag_func([pair for pair in pairs.values()])
    return {key: result for key, result in zip(pairs.keys(), rag_results, strict=True)}
