from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import cbrkit

type Graph[N] = cbrkit.model.graph.Graph[str, N, None, None]

ALGORITHMS: dict[
    str,
    Callable[
        ..., cbrkit.typing.AnySimFunc[Graph[str | int], cbrkit.sim.graphs.GraphSim[str]]
    ],
] = {
    "astar": cbrkit.sim.graphs.astar.build,
    "brute_force": cbrkit.sim.graphs.brute_force,
    "ged": cbrkit.sim.graphs.ged,
    "greedy": cbrkit.sim.graphs.greedy,
    # TODO: needs separate test as it is stricter than the others
    # "isomorphism": cbrkit.sim.graphs.isomorphism,
}


@dataclass
class Setup[N]:
    casebase: cbrkit.typing.Casebase[str, Graph[N]]
    query: Graph[N]
    node_sim_func: cbrkit.typing.AnySimFunc[N, cbrkit.typing.Float]
    node_matcher: cbrkit.sim.graphs.ElementMatcher[N]
    baseline: cbrkit.typing.Casebase[str, Any]


@pytest.fixture
def setup_v1() -> Setup[str | int]:
    return Setup(
        casebase={
            key: cbrkit.model.graph.from_dict(value)
            for key, value in cbrkit.loaders.file("./data/graphs-v1.json").items()
        },
        query=cbrkit.model.graph.from_dict(
            {
                "nodes": {
                    "node1": "A string value",
                    "node2": 42,
                },
                "edges": {
                    "edge1": {"source": "node1", "target": "node2", "value": None}
                },
                "value": None,
            }
        ),
        node_sim_func=cbrkit.sim.type_table(
            {
                str: cbrkit.sim.generic.equality(),
                int: cbrkit.sim.numbers.linear_interval(0, 200),
            }
        ),
        node_matcher=cbrkit.sim.graphs.type_element_matcher,
        baseline={},
    )


@pytest.mark.parametrize(
    "func_name",
    ["astar", "brute_force", "ged", "greedy", "isomorphism"],
)
def test_v1(setup_v1, func_name) -> None:
    func = ALGORITHMS[func_name]
    retriever = cbrkit.retrieval.build(
        func(
            node_sim_func=setup_v1.node_sim_func,
            node_matcher=setup_v1.node_matcher,
        )
    )
    result = cbrkit.retrieval.apply_query(
        setup_v1.casebase,
        setup_v1.query,
        retriever,
    )

    assert result.similarities["first"].value == 1.0
    assert result.similarities["second"].value < 1.0


@pytest.fixture
def setup_v2() -> Setup[dict[str, str]]:
    return Setup(
        casebase={
            key: cbrkit.model.graph.from_dict(value)
            for key, value in cbrkit.loaders.file("./data/graphs-v2.json").items()
        },
        query=cbrkit.model.graph.from_dict(
            {
                "nodes": {
                    "q1": {"type": "Person", "text": "Alice"},
                    "q2": {"type": "Company", "text": "Acme Corp"},
                },
                "edges": {"qe1": {"source": "q1", "target": "q2", "value": None}},
                "value": None,
            }
        ),
        node_sim_func=cbrkit.sim.attribute_value(
            attributes={"text": cbrkit.sim.generic.equality()},
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        node_matcher=lambda x, y: x["type"] == y["type"],
        baseline=cbrkit.loaders.file("./data/graphs-v2-baseline.json"),
    )


@pytest.mark.parametrize(
    "func_name", ["astar", "brute_force", "ged", "greedy", "isomorphism"]
)
def test_v2(setup_v2, func_name):
    func = ALGORITHMS[func_name]
    retriever = cbrkit.retrieval.build(
        func(
            node_sim_func=setup_v2.node_sim_func,
            node_matcher=setup_v2.node_matcher,
        )
    )
    result = cbrkit.retrieval.apply_query(
        setup_v2.casebase,
        setup_v2.query,
        retriever,
    )

    for key, sim in result.similarities.items():
        baseline = setup_v2.baseline[key]

        assert sim.value == baseline["value"], key
        assert sim.node_mapping == baseline["node_mapping"], key
        assert sim.edge_mapping == baseline["edge_mapping"], key
        assert sim.node_similarities == baseline["node_similarities"], key
        assert sim.edge_similarities == baseline["edge_similarities"], key
