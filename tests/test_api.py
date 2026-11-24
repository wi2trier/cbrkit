"""REST API of a system, in process without any network."""

import pytest

import cbrkit
from cbrkit import spec
from cbrkit.errors import Refused
from cbrkit.system import System

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cbrkit.system.api import app

CASES = {0: {"price": 10.0, "fuel": "petrol"}, 1: {"price": 20.0, "fuel": "diesel"}}


def client(token: str | None = None, **sections) -> TestClient:
    config = spec.infer(CASES, cbrkit.io.Source(format="csv")).model_copy(
        update=sections
    )

    return TestClient(
        app(
            System(config=config, storage=cbrkit.indexable.InMemory(CASES)),
            token=token,
            max_body=200,
        ),
        raise_server_exceptions=False,
    )


def test_routes_follow_the_specification():
    api = client(retain=None)
    schema = api.get("/openapi.json").json()
    request = schema["components"]["schemas"]["Case"]

    assert set(request["properties"]) == {"price", "fuel"}
    assert all(name.isalnum() for name in schema["components"]["schemas"])
    assert "/retrieve" in schema["paths"] and "/retain" not in schema["paths"]
    assert (
        api.post("/retrieve", json={"query": {"price": 19}, "limit": 1}).json()["hits"][
            0
        ]["key"]
        == 1
    )
    assert api.post("/retrieve", json={"query": {"color": "red"}}).status_code == 422
    assert api.post("/retrieve", json={"query": {}, "limit": 1000}).status_code == 422
    assert api.post("/retain", json={"query": {}, "solutions": []}).status_code == 404
    assert api.post("/retrieve", content=b"x" * 300).status_code == 413
    assert api.get("/config").json()["measure"]["kind"] == "attribute_value"


def test_graphs_are_validated_and_serialized_by_pydantic():
    graphs = {
        key: cbrkit.model.graph.from_dict({"nodes": {"x": {"type": "t", "v": v}}})
        for key, v in (("a", 1.0), ("b", 5.0))
    }
    config = spec.infer(graphs, cbrkit.io.Source(format="json", graphs=True))
    api = TestClient(
        app(System(config=config, storage=cbrkit.indexable.InMemory(graphs)))
    )
    query = {"nodes": {"x": {"type": "t", "v": 4.0}}}
    hit = api.post("/retrieve", json={"query": query, "limit": 1}).json()["hits"][0]

    assert hit["key"] == "b"
    assert hit["value"] == {
        "version": 1,
        "nodes": {"x": {"type": "t", "v": 5.0}},
        "edges": {},
        "value": None,
    }


def test_token_and_hosts_are_checked():
    api = client(
        token="secret",
        retain=cbrkit.retain.Retainer(),
        limits=spec.Limits(dry_run=True),
    )
    body = {"query": {"price": 15}, "solutions": [{"price": 15, "fuel": "gas"}]}

    assert api.post("/retain", json=body).status_code == 401
    assert api.get("/health").status_code == 200
    response = api.post(
        "/retain", json=body, headers={"Authorization": "Bearer secret"}
    )
    changes = response.json()["changes"]
    assert (changes["applied"], changes["items"][0]["op"]) == (False, "add")
    assert (
        api.post(
            "/retrieve",
            json={"query": {}},
            headers={"Host": "evil.example", "Authorization": "Bearer secret"},
        ).status_code
        == 400
    )


@pytest.mark.parametrize(
    ("error", "status"), [(Refused("refused"), 400), (ValueError("internal"), 500)]
)
def test_only_refusals_reach_the_client(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
):
    def fail(*args: object) -> None:
        raise error

    monkeypatch.setattr("cbrkit.system.core.run_query_async", fail)
    response = client().post("/retrieve", json={"query": {}})

    assert response.status_code == status
    assert ("internal" in response.text) is False
