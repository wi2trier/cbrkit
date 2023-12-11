import csv
import itertools
from collections.abc import Sequence

from cbrkit.sim._taxonomy import Taxonomy, TaxonomyMeasure
from cbrkit.sim.generic import table as generic_table
from cbrkit.typing import (
    FilePath,
    SimFunc,
    SimSeq,
    SimSeqFunc,
    SimVal,
)


def spacy(model_name: str = "en_core_web_lg") -> SimSeqFunc[str]:
    import spacy

    nlp = spacy.load(model_name)

    def wrapped_func(pairs: Sequence[tuple[str, str]]) -> SimSeq:
        texts = set(itertools.chain.from_iterable(pairs))

        with nlp.select_pipes(enable=[]):
            docs = dict(zip(texts, nlp.pipe(texts), strict=True))

        return [docs[x].similarity(docs[y]) for x, y in pairs]

    return wrapped_func


def taxonomy(path: FilePath, measure: TaxonomyMeasure = "wu_palmer") -> SimFunc[str]:
    taxonomy = Taxonomy(path)

    def wrapped_func(x: str, y: str) -> SimVal:
        return taxonomy.similarity(x, y, measure)

    return wrapped_func


def levenshtein(score_cutoff: float | None = None) -> SimFunc[str]:
    import Levenshtein

    def wrapped_func(x: str, y: str) -> SimVal:
        return Levenshtein.ratio(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro(score_cutoff: float | None = None) -> SimFunc[str]:
    import Levenshtein

    def wrapped_func(x: str, y: str) -> SimVal:
        return Levenshtein.jaro(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro_winkler(
    score_cutoff: float | None = None, prefix_weight: float | None = None
) -> SimFunc[str]:
    import Levenshtein

    def wrapped_func(x: str, y: str) -> SimVal:
        return Levenshtein.jaro_winkler(
            x, y, score_cutoff=score_cutoff, prefix_weight=prefix_weight
        )

    return wrapped_func


def table(
    entries: Sequence[tuple[str, str, SimVal]] | FilePath,
    symmetric: bool = True,
    default: SimVal = 0.0,
) -> SimFunc[str]:
    if isinstance(entries, FilePath):
        with open(entries) as f:
            reader = csv.reader(f)
            parsed_entries = [(x, y, float(z)) for x, y, z in reader]
    else:
        parsed_entries = entries

    return generic_table(parsed_entries, symmetric=symmetric, default=default)
