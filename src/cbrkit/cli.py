"""Command line of cbrkit, requires the `cli` extra.

`cbrkit infer` writes the sidecar of a case base, and every operation of `cbrkit.system.operations`
is a command that reads a request as JSON and prints the response as JSON,
like the routes of `cbrkit serve` and the tools of `cbrkit mcp`.
The documentation build renders the reference of every command below.

.. include:: ../../cli.md
"""

import asyncio
import inspect
import ipaddress
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import orjson
import typer
from pydantic import ValidationError

from . import io, spec
from .errors import CbrkitError
from .schema import parse_docstring
from .system import System
from .system.operations import OPERATIONS, operations

__all__ = ["app"]

app = typer.Typer(
    name="cbrkit",
    help="Case-based reasoning over a case base configured by its sidecar file.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

Casebase = Annotated[
    Path, typer.Argument(help="Case base file or folder.", exists=True)
]
ConfigPath = Annotated[
    Path | None,
    typer.Option("--config", help="Sidecar file, <casebase>.cbrkit.json by default."),
]


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)

    return typer.Exit(1)


def _read(request: str) -> str:
    if request == "-":
        return sys.stdin.read()

    if request.startswith("@"):
        return Path(request[1:]).read_text()

    return request


def _command(name: str) -> Callable[..., None]:
    def command(
        casebase: Casebase,
        request: Annotated[
            str,
            typer.Argument(
                help="Request as JSON, @file to read a file, or - for stdin."
            ),
        ] = "{}",
        config: ConfigPath = None,
    ) -> None:
        try:
            enabled = {
                operation.__name__: operation
                for operation in operations(System.load(casebase, config))
            }

            if name not in enabled:
                raise _fail(f"the specification does not enable {name}")

            operation = enabled[name]
            arguments = operation.request.model_validate_json(_read(request))
            response = asyncio.run(operation(**dict(arguments)))
        except (OSError, ValidationError, CbrkitError) as error:
            raise _fail(str(error)) from error

        typer.echo(response.model_dump_json(indent=2))

    return command


for _method in OPERATIONS:
    app.command(
        name=_method.__name__,
        help=parse_docstring(inspect.getdoc(_method) or "")[0],
    )(_command(_method.__name__))


@app.command()
def infer(
    casebase: Casebase,
    key_column: Annotated[
        str | None,
        typer.Option(help="Column or field keying the records, positions by default."),
    ] = None,
    embedder: Annotated[
        str | None,
        typer.Option(help="Embedder name for long texts, see the providers section."),
    ] = None,
    samples: Annotated[int, typer.Option(help="Sample values kept per element.")] = 20,
) -> None:
    """Infers the measure of a case base and writes or merges its sidecar file."""
    try:
        source = io.detect(casebase).model_copy(update={"key_column": key_column})
        path = io.companion(casebase, "sidecar")
        saved = spec.read(path) if path.exists() else None
        config = spec.infer(
            io.load(casebase, source),
            source,
            saved,
            embedder=embedder,
            samples=samples,
        )

        if config != saved:
            spec.write(path, config)
    except (OSError, ValidationError, CbrkitError) as error:
        raise _fail(str(error)) from error

    typer.echo(path)


@app.command()
def schema(
    output: Annotated[
        Path | None, typer.Option(help="File to write, stdout by default.")
    ] = None,
) -> None:
    """Prints the JSON Schema of sidecar files."""
    schema = orjson.dumps(
        spec.Config.model_json_schema(), option=orjson.OPT_INDENT_2
    ).decode()

    if output is None:
        typer.echo(schema)
    else:
        output.write_text(schema)


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return host == "localhost"


@app.command()
def serve(
    casebase: Casebase,
    config: ConfigPath = None,
    host: Annotated[str, typer.Option(help="Address to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8000,
    token: Annotated[
        str | None,
        typer.Option(
            envvar="CBRKIT_TOKEN",
            help="Bearer token required by every request, best given as CBRKIT_TOKEN to keep it out of the process list.",
        ),
    ] = None,
    max_body: Annotated[int, typer.Option(help="Largest request body in bytes.")] = 1
    << 20,
    mcp: Annotated[
        bool, typer.Option(help="Serve MCP at /mcp, requires the mcp extra.")
    ] = True,
    insecure: Annotated[
        bool,
        typer.Option(
            help="Allow a public address or write operations without a token."
        ),
    ] = False,
) -> None:
    """Serves the operations over REST and MCP, requires the api extra."""
    import uvicorn

    from .system.api import LOOPBACK_HOSTS
    from .system.api import app as api

    try:
        system = System.load(casebase, config)
    except (OSError, ValidationError, CbrkitError) as error:
        raise _fail(str(error)) from error

    # An empty token would accept the bare `Bearer ` header, so it is none.
    token = token or None

    if token is None and not insecure:
        if not _loopback(host):
            raise _fail("a public address needs --token or --insecure")

        if system.config.writes:
            raise _fail("write operations need --token or --insecure")

    hosts = LOOPBACK_HOSTS if _loopback(host) else ("*",)
    uvicorn.run(api(system, token, max_body, hosts, mcp), host=host, port=port)


@app.command("mcp")
def mcp_stdio(casebase: Casebase, config: ConfigPath = None) -> None:
    """Serves the operations as MCP tools over stdio, requires the mcp extra."""
    from .system.mcp import server

    try:
        system = System.load(casebase, config)
    except (OSError, ValidationError, CbrkitError) as error:
        raise _fail(str(error)) from error

    server(system).run(show_banner=False)
