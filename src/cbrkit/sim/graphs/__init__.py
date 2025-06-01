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
)
from .greedy import greedy
from .lsap import lsap
from .precompute import precompute
from .vf2 import vf2

with optional_dependencies():
    from .alignment import dtw

with optional_dependencies():
    from .alignment import smith_waterman

with optional_dependencies():
    from .dfs import dfs

__all__ = [
    "astar",
    "brute_force",
    "dfs",
    "greedy",
    "lsap",
    "precompute",
    "vf2",
    "dtw",
    "smith_waterman",
    "GraphSim",
    "ElementMatcher",
    "SemanticEdgeSim",
    "BaseGraphSimFunc",
    "SearchGraphSimFunc",
    "SearchState",
]
