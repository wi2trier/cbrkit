"""Similarity measures for different data types with aggregation and utility functions.

CBRkit provides built-in similarity measures for standard data types as well as
utilities for combining, caching, and transforming them.
All similarity functions follow the signature ``sim = f(x, y)`` or the batch
variant ``sims = f([(x1, y1), ...])``.
Built-in measures are provided through generator functions that return a
configured similarity function.

Submodules:
    ``cbrkit.sim.numbers``: Numeric similarity (linear, exponential, threshold, sigmoid, step).
    ``cbrkit.sim.strings``: String similarity (Levenshtein, Jaro, Jaro-Winkler, spaCy, NLTK).
    ``cbrkit.sim.collections``: Collection and sequence similarity (Jaccard, Dice, etc.).
    ``cbrkit.sim.generic``: Generic similarity (equality, static, tables).
    ``cbrkit.sim.embed``: Embedding-based similarity with caching (Sentence Transformers, OpenAI).
    ``cbrkit.sim.graphs``: Graph similarity algorithms (A*, VF2, greedy, LAP, etc.).
    ``cbrkit.sim.taxonomy``: Taxonomy-based similarity (Wu-Palmer and others).
    ``cbrkit.sim.pooling``: Pooling functions for aggregating multiple values.

Top-Level Functions:
    ``attribute_value``: Computes similarity for attribute-value data by applying
    per-attribute measures and aggregating them into a global score.
    ``aggregator``: Creates an aggregation function that combines multiple local
    similarity scores into a single global score using a pooling strategy.
    ``combine``: Combines multiple similarity functions and aggregates results.
    ``cache``: Wraps a similarity function with result caching.
    ``transpose`` / ``transpose_value``: Transforms inputs before passing them
    to a similarity function.
    ``table`` / ``dynamic_table`` / ``type_table`` / ``attribute_table``: Lookup-based
    similarity dispatching.

Example:
    Define an attribute-value similarity measure::

        import cbrkit

        sim_func = cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "color": cbrkit.sim.generic.equality(),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        )
"""

from . import collections, embed, generic, graphs, numbers, pooling, strings, taxonomy
from .aggregator import aggregator
from .pooling import PoolingName
from .attribute_value import AttributeValueSim, attribute_value
from .wrappers import (
    attribute_table,
    cache,
    combine,
    dynamic_table,
    table,
    transpose,
    transpose_value,
    type_table,
)

__all__ = [
    "transpose",
    "transpose_value",
    "cache",
    "combine",
    "table",
    "dynamic_table",
    "type_table",
    "attribute_table",
    "collections",
    "generic",
    "numbers",
    "strings",
    "attribute_value",
    "graphs",
    "embed",
    "taxonomy",
    "pooling",
    "aggregator",
    "PoolingName",
    "AttributeValueSim",
]
