from typing import override

import ranx

from cbrkit.helpers import unpack_sim
from cbrkit.retrieval import ResultStep

from ._base import Base


class Retrieval(Base):
    @override
    def __init__(
        self,
        qrels: dict[str, dict[str, int]],
        queries_results: dict[str, ResultStep],
    ) -> None:
        super().__init__(
            ranx.Qrels(qrels),
            ranx.Run(
                {
                    query: {
                        case: unpack_sim(sim)
                        for case, sim in result.similarities.items()
                    }
                    for query, result in queries_results.items()
                }
            ),
        )
