from pathlib import Path

from cbrkit import model
from cbrkit.data_sim._taxonomy import Taxonomy, TaxonomyMeasure
from cbrkit.data_sim.helpers import apply


def taxonomy(
    path: Path, measure: TaxonomyMeasure | None = None
) -> model.DataSimilarityBatchFunc[str]:
    taxonomy = Taxonomy(path)

    def wrapped_func(x: str, y: str) -> model.SimilarityValue:
        return taxonomy.similarity(x, y, measure)

    return apply(wrapped_func)
