"""Graph similarity algorithms for structural matching and comparison.

This module provides various algorithms for computing similarity between
graph structures represented as `cbrkit.model.graph.Graph` instances.
Each algorithm takes node and edge similarity functions and returns a
global graph similarity score.
Every algorithm is a batch function (`cbrkit.typing.BatchSimFunc`) that scores the node
and edge pairs of all graph pairs, such as a whole case base, in one call of the node and
edge measures, and then maps every graph pair on its own with `BaseGraphSimFunc.solve`.
`cbrkit.helpers.unbatchify` turns one into a function of a single graph pair.
Search budgets such as `max_expansions`, `max_iterations`, and `timeout` are unbounded when None.

Algorithms:
- `Astar`: A* search for optimal graph edit distance.
  Guarantees optimal results when `partial_mapping` is enabled unless `beam_width` or
  `pathlength_weight` is used, but may be slow for large graphs.
  `max_expansions` bounds the search deterministically.
- `VF2Rustworkx` / `VF2Networkx`: VF2 algorithm for (sub)graph isomorphism
  via RustWorkX or NetworkX (requires `graphs` extra).
- `Greedy`: Fast greedy matching that pairs nodes by highest similarity.
- `LAP`: Linear Assignment Problem solver using the Hungarian algorithm.
  Fast but approximate, since it folds the edge costs into the node costs.
- `QAP`: Quadratic Assignment Problem solved exactly as a binary linear program.
  Same result as `astar` with exact settings and usually much faster, but slower than
  `LAP`.
  `max_expansions` bounds the solver deterministically, falling back to the `LAP` mapping
  when that is better.
- `BruteForce`: Exhaustive search over all possible node matchings.
  Only practical for small graphs.
- `DFS`: Depth-first search based matching (requires `graphs` extra).
- `GraphDTW`: Dynamic Time Warping for sequential graph alignment.
- `GraphTWED`: Time Warp Edit Distance for sequential graph alignment.
  A metric alternative to `GraphDTW` that deletes unmatched nodes instead of warping them.
- `GraphSmithWaterman`: Smith-Waterman local alignment for sequential graphs.

The classes of `DFS`, `VF2Rustworkx` and `VF2Networkx` are always available,
so specifications name them like any other measure,
and only computing a similarity with them needs the `graphs` extra.

The edit costs act as weights of the graph elements, so every algorithm returns a
similarity between zero and one.
With the default costs, it measures how well the case covers the query.
Every result carries an `exact` flag that is only set when the algorithm proved the
mapping optimal: `BruteForce`, `QAP` when the solver finished, and `astar` with
exact settings when the search completed within its budget.
All other algorithms, including `DFS` whose edit paths follow networkx semantics,
are approximate.

`cbrkit.sim.explain` turns a `cbrkit.model.GraphSim` into a serializable explanation with the mapped
nodes and edges, their scores (per attribute for `AttributeValue` node measures),
the unmapped query elements, and the node and edge coverage.

Matchers (`GRAPH_MATCHERS`):
- `AnyElement`: Allows to map any pair of elements, the default.
- `SameAttribute`: Allows to map elements with equal values of an attribute, such as their type.

Initialization Functions (`SEARCH_INITS`):
- `InitEmpty`: Initializes the search state with no pre-matched nodes.
- `InitUniqueMatches`: Initializes with uniquely matchable node pairs.
- `InitFromMeasure`: Initializes with the mapping of another graph measure, such as `LAP`.

Edge Measures (`EDGE_MEASURES`):
- `SemanticEdge`: Scales the aggregated similarity of the endpoints by that of the edge values, the default.

Types:
- `PairSimilarities`: Similarities of all legal node and edge pairs of a graph pair.
- `ElementMatcher`: Protocol for node/edge matching predicates.
- `EdgeSimFunc`: Protocol for the similarities of edge pairs given those of their endpoints.
- `BaseGraphSimFunc`: Base class for graph similarity functions.
- `SearchGraphSimFunc`: Base class for search-based graph similarity.
- `SearchState` / `SearchStateInit`: Search state types.

Example:
    >>> from cbrkit.sim.generic import Equality
    >>> graph_sim = Astar(node_measure=Equality())
"""

from . import astar
from ._base import BaseGraphSimFunc, PairSimilarities
from ._dfs import DFS
from ._matchers import (
    EDGE_MEASURES,
    GRAPH_MATCHERS,
    AnyElement,
    EdgeSimFunc,
    ElementMatcher,
    SameAttribute,
    SemanticEdge,
)
from ._search import (
    SEARCH_INITS,
    InitEmpty,
    InitFromMeasure,
    InitUniqueMatches,
    SearchGraphSimFunc,
    SearchState,
    SearchStateInit,
)
from ._vf2 import VF2Networkx, VF2Rustworkx
from .alignment import GraphDTW, GraphSmithWaterman, GraphTWED
from .astar import Astar
from .brute_force import BruteForce
from .greedy import Greedy
from .lap import LAP
from .qap import QAP

__all__ = [
    "DFS",
    "EDGE_MEASURES",
    "GRAPH_MATCHERS",
    "LAP",
    "QAP",
    "SEARCH_INITS",
    "AnyElement",
    "Astar",
    "BaseGraphSimFunc",
    "BruteForce",
    "EdgeSimFunc",
    "ElementMatcher",
    "GraphDTW",
    "GraphSmithWaterman",
    "GraphTWED",
    "Greedy",
    "InitEmpty",
    "InitFromMeasure",
    "InitUniqueMatches",
    "PairSimilarities",
    "SameAttribute",
    "SearchGraphSimFunc",
    "SearchState",
    "SearchStateInit",
    "SemanticEdge",
    "VF2Networkx",
    "VF2Rustworkx",
    "astar",
]
