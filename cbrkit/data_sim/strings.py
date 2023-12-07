import itertools
from pathlib import Path

import Levenshtein

from cbrkit import model
from cbrkit.data_sim._taxonomy import Taxonomy, TaxonomyMeasure
from cbrkit.data_sim.helpers import apply


def spacy(model_name: str = "en_core_web_lg") -> model.DataSimilarityBatchFunc[str]:
    import spacy

    nlp = spacy.load(model_name)

    def wrapped_func(*args: tuple[str, str]) -> model.SimilaritySequence:
        texts = set(itertools.chain.from_iterable((x[0], x[1]) for x in args))

        with nlp.select_pipes(enable=[]):
            docs = dict(zip(texts, nlp.pipe(texts), strict=True))

        return [docs[x].similarity(docs[y]) for x, y in args]

    return wrapped_func


def taxonomy(
    path: Path, measure: TaxonomyMeasure | None = None
) -> model.DataSimilarityBatchFunc[str]:
    taxonomy = Taxonomy(path)

    @apply
    def wrapped_func(x: str, y: str) -> model.SimilarityValue:
        return taxonomy.similarity(x, y, measure)

    return wrapped_func


def levenshtein(
    score_cutoff: float | None = None
) -> model.DataSimilarityBatchFunc[str]:
    @apply
    def wrapped_func(x: str, y: str) -> model.SimilarityValue:
        return Levenshtein.ratio(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro(score_cutoff: float | None = None) -> model.DataSimilarityBatchFunc[str]:
    @apply
    def wrapped_func(x: str, y: str) -> model.SimilarityValue:
        return Levenshtein.jaro(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro_winkler(
    score_cutoff: float | None = None, prefix_weight: float | None = None
) -> model.DataSimilarityBatchFunc[str]:
    @apply
    def wrapped_func(x: str, y: str) -> model.SimilarityValue:
        return Levenshtein.jaro_winkler(
            x, y, score_cutoff=score_cutoff, prefix_weight=prefix_weight
        )

    return wrapped_func
