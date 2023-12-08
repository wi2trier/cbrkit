import itertools
from collections.abc import Sequence

from cbrkit.sim._taxonomy import Taxonomy, TaxonomyMeasure
from cbrkit.sim.helpers import sim2seq
from cbrkit.typing import (
    FilePath,
    SimilaritySequence,
    SimilarityValue,
    SimSeqFunc,
)


def spacy(model_name: str = "en_core_web_lg") -> SimSeqFunc[str]:
    import spacy

    nlp = spacy.load(model_name)

    def wrapped_func(pairs: Sequence[tuple[str, str]]) -> SimilaritySequence:
        texts = set(itertools.chain.from_iterable(pairs))

        with nlp.select_pipes(enable=[]):
            docs = dict(zip(texts, nlp.pipe(texts), strict=True))

        return [docs[x].similarity(docs[y]) for x, y in pairs]

    return wrapped_func


def taxonomy(path: FilePath, measure: TaxonomyMeasure = "wu_palmer") -> SimSeqFunc[str]:
    taxonomy = Taxonomy(path)

    @sim2seq
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return taxonomy.similarity(x, y, measure)

    return wrapped_func


def levenshtein(score_cutoff: float | None = None) -> SimSeqFunc[str]:
    import Levenshtein

    @sim2seq
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return Levenshtein.ratio(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro(score_cutoff: float | None = None) -> SimSeqFunc[str]:
    import Levenshtein

    @sim2seq
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return Levenshtein.jaro(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro_winkler(
    score_cutoff: float | None = None, prefix_weight: float | None = None
) -> SimSeqFunc[str]:
    import Levenshtein

    @sim2seq
    def wrapped_func(x: str, y: str) -> SimilarityValue:
        return Levenshtein.jaro_winkler(
            x, y, score_cutoff=score_cutoff, prefix_weight=prefix_weight
        )

    return wrapped_func
