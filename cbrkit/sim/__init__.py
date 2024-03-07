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

from . import collections, generic, numbers, strings
from ._aggregator import PoolingName, aggregator
from ._attribute_value import AttributeValueData, AttributeValueSim, attribute_value

__all__ = [
    "collections",
    "generic",
    "numbers",
    "strings",
    "attribute_value",
    "aggregator",
    "PoolingName",
    "AttributeValueData",
    "AttributeValueSim",
]
