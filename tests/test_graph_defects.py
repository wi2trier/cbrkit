"""Regression tests for the repaired graph similarity measures.

Every test pins down a defect that was only observable at runtime, so the docstring
of each test names the behaviour that used to be wrong.
The `G` identifiers refer to the audit these fixes came from.
"""

import random
from collections.abc import Callable
from typing import Any

import pytest

import cbrkit
from cbrkit.model.graph import Graph, from_dict
from cbrkit.sim.graphs import SemanticEdgeSim, astar, init_empty

type AnyGraph = Graph[Any, Any, Any, Any]


def graph(nodes: dict[str, Any], edges: dict[str, tuple[str, str, Any]]) -> AnyGraph:
    return from_dict(
        {
            "nodes": nodes,
            "edges": {
                key: {"source": source, "target": target, "value": value}
                for key, (source, target, value) in edges.items()
            },
            "value": None,
        }
    )


def equality(x: Any, y: Any) -> float:
    return 1.0 if x == y else 0.0


def measures(*, partial_mapping: bool = False, **kwargs: Any) -> dict[str, Any]:
    return {
        "astar1": astar.build(
            heuristic_func=astar.h1(),
            selection_func=astar.select1(),
            partial_mapping=partial_mapping,
            **kwargs,
        ),
        "astar2": astar.build(
            heuristic_func=astar.h2(),
            selection_func=astar.select2(),
            partial_mapping=partial_mapping,
            **kwargs,
        ),
        "astar3": astar.build(partial_mapping=partial_mapping, **kwargs),
        "astar4": astar.build(
            heuristic_func=astar.h4(),
            selection_func=astar.select4(),
            partial_mapping=partial_mapping,
            **kwargs,
        ),
        "brute_force": cbrkit.sim.graphs.brute_force(**kwargs),
        "dfs": cbrkit.sim.graphs.dfs(**kwargs),
        "greedy": cbrkit.sim.graphs.greedy(**kwargs),
        "lap": cbrkit.sim.graphs.lap(**kwargs),
        "vf2_networkx": cbrkit.sim.graphs.vf2_networkx(**kwargs),
        "vf2_rustworkx": cbrkit.sim.graphs.vf2_rustworkx(**kwargs),
    }


MEASURES = measures(node_sim_func=equality)


def random_graph(rng: random.Random) -> AnyGraph:
    nodes = {f"n{i}": rng.choice("ABC") for i in range(rng.randint(1, 4))}
    keys = list(nodes)
    edges = {
        f"e{i}": (rng.choice(keys), rng.choice(keys), rng.choice("ABC"))
        for i in range(rng.randint(0, 4))
    }

    return graph(nodes, edges)


def is_legal(x: AnyGraph, y: AnyGraph, sim: Any) -> bool:
    """Checks legality as defined by Bergmann and Gil (2014)."""
    node_mapping, edge_mapping = sim.node_mapping, sim.edge_mapping

    if len(set(node_mapping.values())) != len(node_mapping):
        return False

    if len(set(edge_mapping.values())) != len(edge_mapping):
        return False

    return all(
        node_mapping.get(y.edges[y_key].source.key) == x.edges[x_key].source.key
        and node_mapping.get(y.edges[y_key].target.key) == x.edges[x_key].target.key
        for y_key, x_key in edge_mapping.items()
    )


PARALLEL_CASE = graph({"n0": "A", "n1": "B"}, {"e0": ("n0", "n1", "E")})
PARALLEL_QUERY = graph(
    {"n0": "A", "n1": "B"}, {"e0": ("n0", "n1", "E"), "e1": ("n0", "n1", "E")}
)


@pytest.mark.parametrize("name", MEASURES)
def test_parallel_edges_map_injectively(name: str) -> None:
    """G1: a single case edge used to be claimed by every parallel query edge."""
    sim = MEASURES[name](PARALLEL_CASE, PARALLEL_QUERY)

    assert is_legal(PARALLEL_CASE, PARALLEL_QUERY, sim)
    assert sim.value == pytest.approx(0.75)


def test_competing_edges_are_matched_optimally() -> None:
    """G1: competing candidates were claimed greedily, which is not the best pairing."""
    nodes = {"n0": "A", "n1": "B"}
    x = graph(nodes, {"e0": ("n0", "n1", "P"), "e1": ("n0", "n1", "Q")})
    y = graph(nodes, {"e0": ("n0", "n1", "P"), "e1": ("n0", "n1", "R")})
    sims = {("P", "P"): 0.9, ("P", "Q"): 0.8, ("R", "P"): 0.85, ("R", "Q"): 0.1}

    def edge_sim_func(case: Any, query: Any) -> float:
        return sims[query, case]

    func = cbrkit.sim.graphs.brute_force(
        node_sim_func=equality,
        edge_sim_func=SemanticEdgeSim(edge_sim_func=edge_sim_func),
    )
    _, edge_pair_sims = func.pair_similarities(x, y)
    mapping = func.induced_edge_mapping(x, y, {"n0": "n0", "n1": "n1"}, edge_pair_sims)

    assert dict(mapping) == {"e0": "e1", "e1": "e0"}


def test_astar_searches_competing_edges_from_default_initialization() -> None:
    """G1: the default initializer fixed competing edges before scoring them.

    The node matcher has to be discriminating, since only a mutually unique node
    mapping lets the initializer reach the edges at all.
    """
    nodes = {"n0": "A", "n1": "B"}
    x = graph(nodes, {"e0": ("n0", "n1", "P"), "e1": ("n0", "n1", "Q")})
    y = graph(nodes, {"e0": ("n0", "n1", "Q"), "e1": ("n0", "n1", "P")})
    kwargs: dict[str, Any] = {
        "node_sim_func": equality,
        "edge_sim_func": SemanticEdgeSim(edge_sim_func=equality),
        "node_matcher": lambda a, b: a == b,
    }

    sim = astar.build(**kwargs)(x, y)

    assert sim.value == pytest.approx(1.0)
    assert dict(sim.edge_mapping) == {"e0": "e1", "e1": "e0"}
    assert sim.value == cbrkit.sim.graphs.brute_force(**kwargs)(x, y).value


def test_edge_matching_does_not_trade_similarity_for_count() -> None:
    """G1: penalizing forbidden pairs made the solver prefer two cheap pairs to one good one."""
    nodes = {"n0": "A", "n1": "B"}
    x = graph(nodes, {"c0": ("n0", "n1", "C0"), "c1": ("n0", "n1", "C1")})
    y = graph(nodes, {"q0": ("n0", "n1", "Q0"), "q1": ("n0", "n1", "Q1")})
    sims = {("Q0", "C0"): 0.9, ("Q0", "C1"): 0.05, ("Q1", "C0"): 0.05}

    def edge_sim_func(case: Any, query: Any) -> float:
        return sims[query, case]

    func = cbrkit.sim.graphs.brute_force(
        node_sim_func=equality,
        edge_sim_func=SemanticEdgeSim(edge_sim_func=edge_sim_func),
        edge_matcher=lambda case, query: (query, case) in sims,
    )
    _, edge_pair_sims = func.pair_similarities(x, y)
    mapping = func.induced_edge_mapping(x, y, {"n0": "n0", "n1": "n1"}, edge_pair_sims)

    assert dict(mapping) == {"q0": "c0"}


@pytest.mark.parametrize(
    "costs",
    [{}, {"node_del_cost": 0.5}, {"node_del_cost": 2.0}, {"edge_del_cost": 0.25}],
)
def test_exact_measures_agree(costs: dict[str, float]) -> None:
    """G2/G3: A* had to map every mappable element and mis-normalized its heuristics.

    `qap` states the same problem as a binary linear program, so it has to agree with
    the search based measures on every input.
    """
    rng = random.Random(0)
    funcs = measures(node_sim_func=equality, partial_mapping=True, **costs)
    reference = funcs["brute_force"]

    for _ in range(25):
        x, y = random_graph(rng), random_graph(rng)
        optimum = reference(x, y).value

        for name, func in funcs.items():
            if not name.startswith("astar") and name != "qap":
                continue

            assert func(x, y).value == pytest.approx(optimum), name


# The only case node fits both query nodes, but only the second one carries an edge
# that can be mapped along with it, so the first one has to be declined.
PARTIAL_CASE = graph({"n0": "A"}, {"e0": ("n0", "n0", "B"), "e1": ("n0", "n0", "B")})
PARTIAL_QUERY = graph({"n0": "A", "n1": "A"}, {"e0": ("n1", "n1", "B")})


def test_partial_mapping_must_be_enabled_for_the_optimum() -> None:
    """G2: the smaller default search omits the branch required for the optimum."""
    default_func = astar.build(node_sim_func=equality)
    exact_func = astar.build(node_sim_func=equality, partial_mapping=True)
    default_sim = default_func(PARTIAL_CASE, PARTIAL_QUERY)
    exact_sim = exact_func(PARTIAL_CASE, PARTIAL_QUERY)

    assert default_sim.value == pytest.approx(1 / 3)
    assert exact_sim.value == pytest.approx(2 / 3)
    assert dict(exact_sim.node_mapping) == {"n1": "n0"}


def expanded_states(enabled: bool) -> tuple[float, int]:
    """Runs A* on the partial mapping example and counts the states it expands."""
    visited: list[int] = []

    class counting(astar.build[Any, Any, Any, Any]):
        def compute_priority(
            self,
            x: AnyGraph,
            y: AnyGraph,
            state: Any,
            node_pair_sims: Any,
            edge_pair_sims: Any,
        ) -> float:
            visited.append(1)

            return super().compute_priority(x, y, state, node_pair_sims, edge_pair_sims)

    sim = counting(node_sim_func=equality, partial_mapping=enabled)(
        PARTIAL_CASE, PARTIAL_QUERY
    )

    return sim.value, len(visited)


def test_partial_mapping_can_be_disabled() -> None:
    """G2: the extra successor is optional, since it enlarges the search space."""
    optimal, optimal_states = expanded_states(True)
    pruned, pruned_states = expanded_states(False)

    assert optimal == pytest.approx(2 / 3)
    assert pruned == pytest.approx(1 / 3)
    assert pruned_states < optimal_states


def test_zero_similarity_node_is_mapped() -> None:
    """G4: `dfs` tested the similarity for truthiness, so zero meant infeasible."""
    x = graph({"a": "A", "b": "B"}, {"e": ("a", "b", None)})
    y = graph({"a": "A", "b": "Z"}, {"e": ("a", "b", None)})
    sim = cbrkit.sim.graphs.dfs(node_sim_func=equality)(x, y)

    assert dict(sim.node_mapping) == {"a": "a", "b": "b"}


def test_vf2_visits_every_subgraph_once() -> None:
    """G5: the fallback enumerated removal sequences, which grows factorially."""
    visited: list[frozenset[str]] = []

    class counting(cbrkit.sim.graphs.vf2_networkx[Any, Any, Any, Any]):
        def node_mappings(self, x: AnyGraph, y: AnyGraph) -> Any:
            visited.append(frozenset(y.nodes.keys()))

            return super().node_mappings(x, y)

    x = graph({"c0": "Z"}, {})
    y = graph({f"n{i}": "A" for i in range(6)}, {})
    counting(node_sim_func=equality, node_matcher=lambda a, b: a == b)(x, y)

    assert len(visited) == len(set(visited)) == 2**6


@pytest.mark.parametrize("name", MEASURES)
def test_node_pairs_are_scored_once(name: str) -> None:
    """G6: `brute_force` and `vf2` rescored every pair for every candidate mapping."""
    calls: list[tuple[Any, Any]] = []

    def node_sim_func(case: Any, query: Any) -> float:
        calls.append((case, query))

        return equality(case, query)

    x = graph(
        {f"c{i}": chr(65 + i) for i in range(4)},
        {"e0": ("c0", "c1", None), "e1": ("c1", "c2", None)},
    )
    y = graph(
        {f"q{i}": chr(65 + i) for i in range(4)},
        {"f0": ("q0", "q1", None), "f1": ("q1", "q2", None)},
    )
    measures(node_sim_func=node_sim_func)[name](x, y)

    assert len(calls) == len(set(calls)) == len(x.nodes) * len(y.nodes)


def test_dead_end_branch_keeps_its_mapping() -> None:
    """G8: an expansion without successors used to fall back to the initial state."""

    class give_up_after_one:
        """Selects one element and then declines to select any further one."""

        def __call__(
            self, x: AnyGraph, y: AnyGraph, s: Any, node_sims: Any, edge_sims: Any
        ) -> Any:
            if len(s.open_y_nodes) + len(s.open_y_edges) < len(y.nodes) + len(y.edges):
                return None

            return astar.select1()(x, y, s, node_sims, edge_sims)

    x = graph({"n0": "A", "n1": "B"}, {"e0": ("n0", "n1", None)})
    y = graph({"m0": "A", "m1": "B"}, {"f0": ("m0", "m1", None)})
    sim = astar.build(
        node_sim_func=equality,
        selection_func=give_up_after_one(),
        init_func=init_empty(),
    )(x, y)

    assert sim.node_mapping


def test_pathlength_weight_favors_long_paths() -> None:
    """G9: edit costs below the defaults made the priority negative, inverting it."""
    x = graph({"n0": "A", "n1": "B"}, {"e0": ("n0", "n1", None)})
    y = graph({"m0": "A", "m1": "B"}, {"f0": ("m0", "m1", None)})
    func = astar.build(node_sim_func=equality, pathlength_weight=2, node_del_cost=0.5)
    node_sims, edge_sims = func.pair_similarities(x, y)
    root = func.init_search_state(x, y)
    deeper = func.expand(x, y, root, node_sims, edge_sims)[0]

    root_priority = func.compute_priority(x, y, root, node_sims, edge_sims)
    deeper_priority = func.compute_priority(x, y, deeper, node_sims, edge_sims)

    assert root_priority < 0.0
    assert deeper_priority < root_priority


@pytest.mark.parametrize("name", MEASURES)
def test_empty_query_is_fully_satisfied(name: str) -> None:
    """G10: two empty graphs used to score zero instead of one."""
    empty = graph({}, {})

    assert MEASURES[name](empty, empty).value == 1.0


def test_vf2_backends_agree() -> None:
    """G11: networkx found induced isomorphisms and dropped parallel edges."""
    rng = random.Random(11)
    matcher: Callable[[Any, Any], bool] = lambda a, b: a == b

    for _ in range(50):
        x, y = random_graph(rng), random_graph(rng)
        networkx = cbrkit.sim.graphs.vf2_networkx(
            node_sim_func=equality, node_matcher=matcher
        )(x, y)
        rustworkx = cbrkit.sim.graphs.vf2_rustworkx(
            node_sim_func=equality, node_matcher=matcher
        )(x, y)

        assert networkx.value == pytest.approx(rustworkx.value)


def test_connected_edges_are_ordered() -> None:
    """G12: a set was returned, so the greedy edge cost depended on iteration order."""
    func = cbrkit.sim.graphs.lap(node_sim_func=equality)
    g = graph(
        {"alpha": "A", "beta": "B"},
        {f"e{i}": ("alpha", "beta", None) for i in range(4)},
    )

    assert func.connected_edges(g, "alpha") == list(g.edges.keys())
