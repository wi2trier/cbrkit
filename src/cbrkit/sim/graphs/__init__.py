from ...helpers import optional_dependencies
from . import astar
from .brute_force import brute_force
from .common import (
    BaseGraphSimFunc,
    ElementMatcher,
    GraphSim,
    SearchGraphSimFunc,
    SearchState,
    SemanticEdgeSim,
    default_element_matcher,
    type_element_matcher,
)
from .greedy import greedy
from .isomorphism import isomorphism
from .precompute import precompute

with optional_dependencies():
    from .alignment import dtw

with optional_dependencies():
    from .alignment import smith_waterman

with optional_dependencies():
    from .ged import ged

__all__ = [
    "astar",
    "greedy",
    "ged",
    "brute_force",
    "isomorphism",
    "precompute",
    "dtw",
    "smith_waterman",
    "GraphSim",
    "ElementMatcher",
    "default_element_matcher",
    "type_element_matcher",
    "SemanticEdgeSim",
    "BaseGraphSimFunc",
    "SearchGraphSimFunc",
    "SearchState",
]
