from collections.abc import Mapping

from .. import model
from ..helpers import get_metadata
from ..typing import Float, SynthesizerFunc
from .model import QueryResultStep, Result, ResultStep


def apply_result[R, Q, C, V, S: Float](
    result: model.Result[Q, C, V, S] | model.ResultStep[Q, C, V, S],
    synthesis_func: SynthesizerFunc[R, C, V, S],
) -> Result[Q, R]:
    return apply_batches(
        {
            key: (entry.casebase, entry.query, entry.similarities)
            for key, entry in result.queries.items()
        },
        synthesis_func,
    )


def apply_batches[R, Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V, Mapping[C, S]]],
    synthesis_func: SynthesizerFunc[R, C, V, S],
) -> Result[Q, R]:
    synthesis_results = synthesis_func(list(batches.values()))

    return Result(
        [
            ResultStep(
                {
                    key: QueryResultStep(value)
                    for key, value in zip(
                        batches.keys(), synthesis_results, strict=True
                    )
                },
                get_metadata(synthesis_func),
            )
        ]
    )
