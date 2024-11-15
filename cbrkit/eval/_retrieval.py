from typing import Any

from ..retrieval import Result, ResultStep
from ..typing import Float
from ._common import compute


def retrieval_step[QK, CK, S: Float](
    qrels: dict[QK, dict[CK, int]],
    queries_step: dict[QK, ResultStep[CK, Any, S]],
) -> dict[str, float]:
    return compute(
        qrels,
        {
            query: {case: sim for case, sim in step.similarities.items()}
            for query, step in queries_step.items()
        },
    )


def retrieval[QK, CK, S: Float](
    qrels: dict[QK, dict[CK, int]],
    queries_result: dict[QK, Result[CK, Any, S]],
) -> list[dict[str, float]]:
    all_steps = {len(result.steps) for result in queries_result.values()}

    if len(all_steps) != 1:
        raise ValueError("All queries must have the same number of retrieval steps")

    num_steps = all_steps.pop()

    return [
        retrieval_step(
            qrels,
            {query: result.steps[idx] for query, result in queries_result.items()},
        )
        for idx in range(num_steps)
    ]
