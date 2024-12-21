import cbrkit


def test_astar():
    casebase: cbrkit.typing.Casebase[
        str, cbrkit.sim.graphs.Graph[str, str | int, None, str]
    ] = {
        k: cbrkit.sim.graphs.from_dict(v)
        for k, v in cbrkit.loaders.file("./data/graphs.json").items()
    }

    query: cbrkit.sim.graphs.Graph[str, str | int, None, str] = (
        cbrkit.sim.graphs.from_dict(
            {
                "nodes": {
                    "node1": {"data": "A string value"},
                    "node2": {"data": 42},
                },
                "edges": {
                    "edge1": {"source": "node1", "target": "node2", "data": None}
                },
                "data": "Some query",
            }
        )
    )

    node_sim: cbrkit.typing.BatchSimFunc[
        cbrkit.sim.graphs.Node[str, str | int], float
    ] = cbrkit.sim.transpose(
        cbrkit.sim.generic.type_table(
            {
                str: cbrkit.sim.generic.equality(),
                int: cbrkit.sim.numbers.linear_interval(0, 200),
            },
            default=cbrkit.sim.generic.static(0.0),
        ),
        cbrkit.helpers.unpack_value,
    )

    graph_sim = cbrkit.sim.graphs.astar.build(
        cbrkit.sim.graphs.astar.g1(node_sim),
        cbrkit.sim.graphs.astar.h2(node_sim),
    )
    retriever = cbrkit.retrieval.build(graph_sim)

    result = cbrkit.retrieval.apply_query(casebase, query, retriever)

    assert result.similarities["first"].value == 1.0
    assert result.similarities["second"].value < 1.0
