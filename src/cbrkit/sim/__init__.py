"""
CBRkit contains a selection of similarity measures for different data types.
Besides measures for standard data types like
numbers (`cbrkit.sim.numbers`),
strings (`cbrkit.sim.strings`),
lists/collections (`cbrkit.sim.collections`),
and generic data (`cbrkit.sim.generic`),
there is also a measure for attribute-value data.
Additionally, the module contains an aggregator to combine multiple local measures into a global score.
"""

from . import collections, embed, generic, graphs, numbers, strings, taxonomy
from .aggregator import PoolingName, aggregator
from .attribute_value import AttributeValueSim, attribute_value
from .wrappers import (
    attribute_table,
    cache,
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
    "aggregator",
    "PoolingName",
    "AttributeValueSim",
]
