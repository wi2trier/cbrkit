from collections.abc import Mapping

from .. import model
from ..helpers import get_metadata
from ..typing import Float, RagFunc
from ._model import QueryResultStep, Result, ResultStep


def apply_result[R, Q, C, V, S: Float](
    result: model.Result[Q, C, V, S] | model.ResultStep[Q, C, V, S],
    rag_func: RagFunc[R, C, V, S],
) -> Result[Q, R]:
    return apply_batches(
        {
            key: (entry.casebase, entry.query, entry.similarities)
            for key, entry in result.queries.items()
        },
        rag_func,
    )


def apply_batches[R, Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V, Mapping[C, S]]],
    rag_func: RagFunc[R, C, V, S],
) -> Result[Q, R]:
    rag_results = rag_func(list(batches.values()))

    return Result(
        [
            ResultStep(
                {
                    key: QueryResultStep(value)
                    for key, value in zip(batches.keys(), rag_results, strict=True)
                },
                get_metadata(rag_func),
            )
        ]
    )
