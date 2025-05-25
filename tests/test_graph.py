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
    "isomorphism": cbrkit.sim.graphs.isomorphism,
}


@dataclass
class Setup[N]:
    casebase: cbrkit.typing.Casebase[str, Graph[N]]
    query: Graph[N]
    node_sim_func: cbrkit.typing.AnySimFunc[N, cbrkit.typing.Float]
    node_matcher: cbrkit.sim.graphs.ElementMatcher[N]
    baseline: cbrkit.typing.Casebase[str, Any]


@pytest.fixture(scope="session")
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
        node_matcher=lambda x, y: type(x) is type(y),
        baseline={},
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS.keys())
def test_v1(setup_v1, algorithm) -> None:
    func = ALGORITHMS[algorithm]
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

    assert result.similarities["first"].value == pytest.approx(1.0)
    assert result.similarities["second"].value < 1.0


@pytest.fixture(scope="session")
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
            attributes={
                "text": lambda x, y: 1.0 if x == y else 0.2,
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        node_matcher=lambda y, x: y["type"] == x["type"],
        baseline=cbrkit.loaders.file("./data/graphs-v2-baseline.json"),
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS.keys())
def test_v2(setup_v2, algorithm):
    func = ALGORITHMS[algorithm]
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

        if algorithm == "isomorphism" and not baseline["edge_mapping"]:
            # If there is no edge mapping in the baseline, the algorithm does not find any mapping
            continue

        # assert sim.value == pytest.approx(baseline["value"]), key
        assert sim.node_mapping == baseline["node_mapping"], key
        assert sim.edge_mapping == baseline["edge_mapping"], key
        assert sim.node_similarities == pytest.approx(baseline["node_similarities"]), (
            key
        )
        assert sim.edge_similarities == pytest.approx(baseline["edge_similarities"]), (
            key
        )
