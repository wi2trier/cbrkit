from collections.abc import Mapping

from cbrkit.helpers import get_metadata

from .. import model
from ..typing import Float, RagFunc
from ._model import QueryResultStep, Result, ResultStep


def apply_result[Q, C, V, S: Float, T](
    result: model.Result[Q, C, V, S] | model.ResultStep[Q, C, V, S],
    rag_func: RagFunc[C, V, S, T],
) -> Result[Q, T]:
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
) -> Result[Q, T]:
    rag_results = rag_func([pair for pair in pairs.values()])

    return Result(
        [
            ResultStep(
                {
                    key: QueryResultStep(value)
                    for key, value in zip(pairs.keys(), rag_results, strict=True)
                },
                get_metadata(rag_func),
            )
        ]
    )
