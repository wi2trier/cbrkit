"""MCP server of a system, with an in-memory client."""

import asyncio

import pytest

import cbrkit
from cbrkit import spec
from cbrkit.system import System

fastmcp = pytest.importorskip("fastmcp")

from cbrkit.system.api import app
from cbrkit.system.mcp import server

CASES = {0: {"price": 10.0}, 1: {"price": 20.0}}


def test_tools_match_the_routes():
    config = spec.infer(CASES, cbrkit.io.Source(format="csv")).model_copy(
        update={
            "retain": cbrkit.retain.Retainer(),
            "maintain": cbrkit.maintain.Maintainer(
                competence=0.5, remove=frozenset({"auxiliary"})
            ),
        }
    )
    system = System(config=config, storage=cbrkit.indexable.InMemory(CASES))

    async def run():
        async with fastmcp.Client(server(system)) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            result = await client.call_tool(
                "retrieve", {"query": {"price": 19}, "limit": 1}
            )
            config = await client.read_resource("cbrkit://config")

            return tools, result, config

    tools, result, config = asyncio.run(run())
    routes = app(system).openapi()["paths"]

    assert {f"/{name}" for name in tools} == set(routes) - {"/health", "/config"}
    assert tools["retain"].annotations.read_only_hint is False
    assert tools["retrieve"].annotations.read_only_hint is True
    assert tools["retain"].annotations.destructive_hint is False
    assert tools["maintain"].annotations.destructive_hint is True
    assert "price" in str(tools["retrieve"].input_schema)
    assert result.structured_content["hits"][0]["key"] == 1
    assert spec.Config.model_validate_json(config[0].text) == system.config
