"""Graph similarity algorithms for structural matching and comparison.

This module provides various algorithms for computing similarity between
graph structures represented as `cbrkit.model.graph.Graph` instances.
Each algorithm takes node and edge similarity functions and returns a
global graph similarity score.

Algorithms:
- `astar`: A* search for optimal graph edit distance.
  Guarantees optimal results when `partial_mapping` is enabled unless `beam_width` or
  `pathlength_weight` is used, but may be slow for large graphs.
- `vf2`: VF2 algorithm for (sub)graph isomorphism.
  Available in pure Python (`vf2`), NetworkX (`vf2_networkx`),
  and RustWorkX (`vf2_rustworkx`) variants.
- `greedy`: Fast greedy matching that pairs nodes by highest similarity.
- `lap`: Linear Assignment Problem solver using the Hungarian algorithm.
  Fast but approximate, since it folds the edge costs into the node costs.
- `qap`: Quadratic Assignment Problem solved exactly as a binary linear program.
  Same result as `astar` with exact settings and usually much faster, but slower than
  `lap`.
- `brute_force`: Exhaustive search over all possible node matchings.
  Only practical for small graphs.
- `dfs`: Depth-first search based matching (requires `graphs` extra).
- `dtw`: Dynamic Time Warping for sequential graph alignment.
- `twed`: Time Warp Edit Distance for sequential graph alignment.
  A metric alternative to `dtw` that deletes unmatched nodes instead of warping them.
- `smith_waterman`: Smith-Waterman local alignment for sequential graphs.

Initialization Functions:
- `init_empty`: Initializes the search state with no pre-matched nodes.
- `init_unique_matches`: Initializes with uniquely matchable node pairs.

Types:
- `GraphSim`: Protocol for graph similarity functions.
- `ElementMatcher`: Protocol for node/edge matching predicates.
- `SemanticEdgeSim`: Semantic edge similarity function type.
- `BaseGraphSimFunc`: Base class for graph similarity functions.
- `SearchGraphSimFunc`: Base class for search-based graph similarity.
- `SearchState` / `SearchStateInit`: Search state types.

Example:
    >>> from cbrkit.sim.generic import equality
    >>> graph_sim = astar.build(node_sim_func=equality())
"""

from ...helpers import optional_dependencies
from . import astar
from .alignment import dtw, smith_waterman, twed
from .brute_force import brute_force
from .common import (
    BaseGraphSimFunc,
    ElementMatcher,
    GraphSim,
    SearchGraphSimFunc,
    SearchState,
    SearchStateInit,
    SemanticEdgeSim,
    init_empty,
    init_unique_matches,
)
from .greedy import greedy
from .lap import lap
from .qap import qap
from .vf2 import vf2, vf2_networkx, vf2_rustworkx

with optional_dependencies():
    from .dfs import dfs

__all__ = [
    "BaseGraphSimFunc",
    "ElementMatcher",
    "GraphSim",
    "SearchGraphSimFunc",
    "SearchState",
    "SearchStateInit",
    "SemanticEdgeSim",
    "astar",
    "brute_force",
    "dfs",
    "dtw",
    "greedy",
    "init_empty",
    "init_unique_matches",
    "lap",
    "qap",
    "smith_waterman",
    "twed",
    "vf2",
    "vf2_networkx",
    "vf2_rustworkx",
]
