"""MCP server of a system, one tool per function of `cbrkit.system.operations.operations`.

Requires the `mcp` extra.
Tool names, descriptions, schemas, and annotations come from the operations,
so they match the routes of `cbrkit.system.api` and the commands of the `cbrkit` command line,
and the resource `cbrkit://config` holds the specification like `GET /config` of the REST API.
Error details are masked, except for refused operations and conflicting retain and maintenance runs.

Example:
    >>> server(System.load("cases.csv")).run()  # doctest: +SKIP
"""

from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.resources import TextResource
from fastmcp.tools import FunctionTool, ToolResult
from mcp_types import ToolAnnotations
from pydantic import AnyUrl

from ..errors import Conflict, Refused
from ..spec import dumps
from .core import System
from .operations import operations

__all__ = ["server"]

INSTRUCTIONS = """\
Case-based reasoning over one case base: retrieve similar cases for a query,
reuse and revise them into solutions, and retain accepted solutions if the case base allows it.
Maintaining the case base is a separate tool outside this cycle that needs no query.
Every result has the same shape: hits and the changes of a retain or maintenance run with the reason of every change.
The resource cbrkit://config holds the specification with the language models that produce the results.
The input schema of every tool lists the attributes of cases and queries."""


class _OperationTool(FunctionTool):
    """Reports refused operations and conflicting runs to the client, unlike other errors."""

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return await super().run(arguments)
        except (Refused, Conflict) as error:
            raise ToolError(str(error)) from error


def server(system: System[Any]) -> FastMCP:
    """An MCP server with the operations that the specification of a system enables.

    Run it over stdio with `server(system).run()`,
    or mount `server(system).http_app(path="/")` in an ASGI app as `cbrkit.system.api.app` does.

    Args:
        system: The system that runs the operations.
    """
    tools = [
        _OperationTool.from_function(
            operation,
            annotations=ToolAnnotations(
                read_only_hint=operation.read_only,
                destructive_hint=operation.destructive,
                idempotent_hint=operation.read_only,
                open_world_hint=False,
            ),
        )
        for operation in operations(system)
    ]
    mcp = FastMCP(
        "cbrkit",
        instructions=INSTRUCTIONS,
        tools=tools,
        mask_error_details=True,
        strict_input_validation=True,
    )
    mcp.add_resource(
        TextResource(
            uri=AnyUrl("cbrkit://config"),
            name="config",
            description="The specification that every tool follows,"
            " with the language models of the providers that produce the results.",
            mime_type="application/json",
            text=dumps(system.config),
        )
    )

    return mcp
