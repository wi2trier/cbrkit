import itertools

from cbrkit.data_sim._taxonomy import Taxonomy, TaxonomyMeasure
from cbrkit.data_sim.helpers import batchify
from cbrkit.typing import (
    DataSimBatchFunc,
    FilePath,
    SimilaritySequence,
    SimilarityValue,
)


def spacy(model_name: str = "en_core_web_lg") -> DataSimBatchFunc[str]:
    import spacy

    nlp = spacy.load(model_name)

    def wrapped_func(*args: tuple[str, str]) -> SimilaritySequence:
        texts = set(itertools.chain.from_iterable((x[0], x[1]) for x in args))

        with nlp.select_pipes(enable=[]):
            docs = dict(zip(texts, nlp.pipe(texts), strict=True))

        return [docs[x].similarity(docs[y]) for x, y in args]

    return wrapped_func


def taxonomy(
    path: FilePath, measure: TaxonomyMeasure = "wu_palmer"
) -> DataSimBatchFunc[str]:
    taxonomy = Taxonomy(path)

    @batchify
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return taxonomy.similarity(x, y, measure)

    return wrapped_func


def levenshtein(score_cutoff: float | None = None) -> DataSimBatchFunc[str]:
    import Levenshtein

    @batchify
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return Levenshtein.ratio(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro(score_cutoff: float | None = None) -> DataSimBatchFunc[str]:
    import Levenshtein

    @batchify
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return Levenshtein.jaro(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro_winkler(
    score_cutoff: float | None = None, prefix_weight: float | None = None
) -> DataSimBatchFunc[str]:
    import Levenshtein

    @batchify
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return Levenshtein.jaro_winkler(
            x, y, score_cutoff=score_cutoff, prefix_weight=prefix_weight
        )

    return wrapped_func
