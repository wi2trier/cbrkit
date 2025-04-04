from ...helpers import optional_dependencies
from . import astar, greedy
from .brute_force import brute_force
from .common import (
    ElementMatcher,
    GraphSim,
    default_element_matcher,
    type_element_matcher,
)
from .isomorphism import isomorphism
from .precompute import precompute

with optional_dependencies():
    from .alignment import dtw

with optional_dependencies():
    from .alignment import smith_waterman

__all__ = [
    "astar",
    "greedy",
    "brute_force",
    "isomorphism",
    "precompute",
    "dtw",
    "smith_waterman",
    "GraphSim",
    "ElementMatcher",
    "default_element_matcher",
    "type_element_matcher",
]
