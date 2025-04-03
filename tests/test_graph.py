import cbrkit


def test_astar():
    casebase: cbrkit.typing.Casebase[
        str, cbrkit.model.graph.Graph[str, str | int, None, str]
    ] = {
        key: cbrkit.model.graph.from_dict(value)
        for key, value in cbrkit.loaders.file("./data/graphs.json").items()
    }

    query: cbrkit.model.graph.Graph[str, str | int, None, str] = (
        cbrkit.model.graph.from_dict(
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
        )
    )

    node_sim = cbrkit.sim.type_table(
        {
            str: cbrkit.sim.generic.equality(),
            int: cbrkit.sim.numbers.linear_interval(0, 200),
        },
        default=cbrkit.sim.generic.static(0.0),
    )

    graph_sim = cbrkit.sim.graphs.astar.build(
        cbrkit.sim.graphs.astar.g1(node_sim),
        cbrkit.sim.graphs.astar.h2(node_sim),
        node_matcher=cbrkit.sim.graphs.type_element_matcher,
    )
    retriever = cbrkit.retrieval.build(graph_sim)

    result = cbrkit.retrieval.apply_query(casebase, query, retriever)

    assert result.similarities["first"].value == 1.0
    assert result.similarities["second"].value < 1.0
