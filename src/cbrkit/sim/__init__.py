"""Similarity measures for different data types with aggregation and utility functions.

A similarity measure compares a case and a query, `sim = f(case, query)`,
and a batch measure scores several pairs at once, `sims = f([(case1, query1), ...])`,
see `cbrkit.helpers.batched`.
Built-in measures are frozen classes whose instances are configured similarity functions,
and they serialize by their `kind` as members of the family `MEASURES`.
A measure returns a float or a detailed value such as `cbrkit.model.AttributeValueSim`,
`float(sim)` gives the score of both, comparisons use it, and results keep the details through JSON.
The numeric, date, string and generic measures also score whole columns of pairs with numpy or rapidfuzz
(`cbrkit.typing.ColumnSimFunc`), which `AttributeValue` and the retrievers use automatically.

Submodules:
- `cbrkit.sim.numbers`: Numeric similarity (linear, linear interval, exponential, sigmoid, threshold).
- `cbrkit.sim.dates`: Similarity of dates, datetimes and durations.
- `cbrkit.sim.strings`: String similarity (Levenshtein, Indel, Jaro, Jaro-Winkler, n-grams, regex, glob, word coverage).
- `cbrkit.sim.generic`: Generic similarity (equality, type equality, static, lookup tables).
- `cbrkit.sim.collections`: Collection and sequence similarity (Jaccard, DTW, TWED, Smith-Waterman, etc.).
- `cbrkit.sim.embed`: Embedding-based similarity with caching (Sentence Transformers, BM25, remote providers).
- `cbrkit.sim.graphs`: Graph similarity algorithms (A*, VF2, greedy, LAP, QAP, etc.).
- `cbrkit.sim.taxonomy`: Similarity of values by their nodes in a taxonomy (Wu-Palmer, weights, levels, paths).
- `cbrkit.sim.llm`: Similarity judged by a language model of the host (`Critique`).
- `cbrkit.sim.pooling`: Pooling functions of an `Aggregator`, one class of `POOLINGS` per pooling such as `Mean` or `KMax`.

Top-Level Measures and Helpers:
- `AttributeValue`: Applies a measure to every attribute of attribute-value data
  and aggregates the local similarities, nested for object-oriented cases.
- `Aggregator`: Combines local similarities into a global one with a pooling function
  and weights by key or position.
- `explain`: Breaks a similarity down by attribute or graph mapping for user interfaces,
  by the measure that computed it if that is `Explainable`.
- `Combine`: Combines several named measures of the same values and aggregates their similarities.
- `Cache`: Wraps a similarity function with a result cache in a store the caller owns.
- `Transpose` / `transpose_value` / `transpose_singleton`: Transforms inputs before passing them
  to a similarity function.
- `AttributeTable`: Measure chosen by a shared attribute value or type of a pair.

`Combine` and `AttributeTable` serialize by their kind,
while `Cache` and `Transpose` hold code or runtime state and are built in Python only.

Custom measures are plain functions, such as one that rates partial matches of colors.
To serialize a custom measure, for instance in a `cbrkit.spec` sidecar,
write it as a `cbrkit.schema.serializable` frozen, keyword-only dataclass with a `kind` field
and a Google-style docstring, which gives its JSON Schema,
and extend the model that holds it, such as `cbrkit.spec.Config`, with `cbrkit.schema.extend`.

Example:
    >>> def color(case: str, query: str) -> float:
    ...     return 1.0 if case == query else 0.5 if case in query or query in case else 0.0
    >>> similarity = AttributeValue(
    ...     attributes={
    ...         "price": numbers.Linear(max_distance=100000),
    ...         "color": color,
    ...         "maker": AttributeValue(attributes={"name": strings.JaroWinkler(), "country": generic.Equality()}),
    ...     },
    ...     aggregator=Aggregator(weights={"price": 2}),
    ... )
    >>> case = {"price": 20000, "color": "light blue", "maker": {"name": "Audi", "country": "DE"}}
    >>> query = {"price": 30000, "color": "blue", "maker": {"name": "Audi", "country": "DE"}}
    >>> round(similarity([(case, query)])[0].value, 3)
    0.825
"""

from typing import Annotated, TypeVar

from ..schema import Family
from ..typing import AnySimFunc, Float
from . import (
    collections,
    dates,
    embed,
    generic,
    graphs,
    llm,
    numbers,
    pooling,
    strings,
    taxonomy,
)
from ._explain import Explainable, explain
from .aggregator import Aggregator
from .attribute_value import AttributeValue
from .wrappers import (
    AttributeTable,
    Cache,
    Combine,
    Transpose,
    transpose_singleton,
    transpose_value,
)

MEASURES = Family(
    "measure",
    (
        AttributeValue,
        AttributeTable,
        Combine,
        collections.DTW,
        collections.Jaccard,
        collections.IsolatedMapping,
        collections.OptimalMapping,
        collections.SequenceCorrectness,
        collections.SequenceMapping,
        collections.SmithWaterman,
        collections.TWED,
        dates.DateLinear,
        embed.Embedding,
        generic.Equality,
        generic.Static,
        generic.Table,
        generic.TypeEquality,
        graphs.Astar,
        graphs.BruteForce,
        graphs.DFS,
        graphs.GraphDTW,
        graphs.GraphSmithWaterman,
        graphs.GraphTWED,
        graphs.Greedy,
        graphs.LAP,
        graphs.QAP,
        graphs.VF2Networkx,
        graphs.VF2Rustworkx,
        llm.Critique,
        numbers.Exponential,
        numbers.Linear,
        numbers.LinearInterval,
        numbers.Sigmoid,
        numbers.Threshold,
        strings.Glob,
        strings.Indel,
        strings.Jaro,
        strings.JaroWinkler,
        strings.Levenshtein,
        strings.Ngram,
        strings.Regex,
        strings.WordCoverage,
        taxonomy.Taxonomy,
    ),
)
"""The built-in similarity measures, which serialize by their kind, for instance in `cbrkit.spec`."""

# classic generic aliases, since pydantic would name a `type` alias by its type arguments in JSON Schemas
_V = TypeVar("_V")
_S = TypeVar("_S", bound=Float, default=float)

Measure = Annotated[AnySimFunc[_V, _S], MEASURES]
"""A similarity or batch similarity function that serializes by kind as one of `MEASURES`, for the fields of classes above `cbrkit.sim`."""

__all__ = [
    "MEASURES",
    "Aggregator",
    "AttributeTable",
    "AttributeValue",
    "Cache",
    "Combine",
    "Explainable",
    "Measure",
    "Transpose",
    "collections",
    "dates",
    "embed",
    "explain",
    "generic",
    "graphs",
    "llm",
    "numbers",
    "pooling",
    "strings",
    "taxonomy",
    "transpose_singleton",
    "transpose_value",
]
