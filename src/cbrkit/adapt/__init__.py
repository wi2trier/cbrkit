"""
CBRkit contains a selection of adaptation functions for different data types.
Besides functions for standard data types like
numbers (`cbrkit.adapt.numbers`),
strings (`cbrkit.adapt.strings`),
and generic data (`cbrkit.adapt.generic`),
there is also a function for attribute-value data.
"""

from . import generic, numbers, strings
from .attribute_value import attribute_value

__all__ = [
    "generic",
    "strings",
    "numbers",
    "attribute_value",
]
