"""REST API of a system, one `POST /<operation>` route per function of `cbrkit.system.operations.operations`.

`GET /config` serves the specification of the system, `GET /health` answers without a token.

Requires the `api` extra, and the `mcp` extra to mount the MCP server at `/mcp`.
Requests carry only cases, queries, keys and numbers, so a client can never name a file or code.
Invalid requests give 422, refused operations 400, retain and maintenance runs that lost every race against other writers 409,
and failures 500 without details.

Example:
    >>> import uvicorn  # doctest: +SKIP
    >>> uvicorn.run(app(System.load("cases.csv"), token="secret"))  # doctest: +SKIP
"""

import inspect
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..errors import Conflict, Refused
from ..spec import Config
from .core import Outcome, System
from .operations import Operation, operations

__all__ = ["app", "router"]

LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]", "testserver")
"""Hosts accepted by default, which keeps DNS rebinding away from a local server."""


@dataclass(slots=True, frozen=True)
class _Bearer:
    """Requires a bearer token on every request except the health check."""

    app: ASGIApp
    token: str

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] != "/health":
            given = Headers(scope=scope).get("authorization", "").encode()

            if not secrets.compare_digest(given, f"Bearer {self.token}".encode()):
                response = PlainTextResponse(
                    "Unauthorized", 401, headers={"WWW-Authenticate": "Bearer"}
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


def _endpoint(operation: Operation) -> Callable[[BaseModel], Awaitable[Outcome[Any]]]:
    """Runs an operation for a request model of its parameters, which FastAPI reads as the body."""

    async def endpoint(body: BaseModel) -> Outcome[Any]:
        return await operation(**dict(body))

    parameter = inspect.Parameter(
        "body", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=operation.request
    )
    vars(endpoint)["__signature__"] = inspect.Signature([parameter])

    return endpoint


def router(system: System[Any]) -> APIRouter:
    """The routes of the operations that the specification of a system enables.

    Args:
        system: The system that runs the operations.
    """
    routes = APIRouter()

    for operation in operations(system):
        routes.add_api_route(
            f"/{operation.__name__}",
            _endpoint(operation),
            methods=["POST"],
            response_model=inspect.signature(operation).return_annotation,
            operation_id=operation.__name__,
            summary=(operation.__doc__ or "").split("\n")[0],
            description=operation.__doc__,
            tags=["read" if operation.read_only else "write"],
        )

    return routes


async def _refused(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=400)


async def _conflict(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=409)


def app(
    system: System[Any],
    token: str | None = None,
    max_body: int = 1 << 20,
    hosts: Sequence[str] = LOOPBACK_HOSTS,
    mcp: bool = False,
) -> FastAPI:
    """A REST API, and optionally an MCP server at `/mcp`, for a system.

    Args:
        system: The system that runs the operations.
        token: Bearer token required by every request except `GET /health`, none if None.
        max_body: Largest request body in bytes.
        hosts: Accepted values of the Host header, `["*"]` for any.
        mcp: Mount the MCP server of `cbrkit.system.mcp` at `/mcp`, which requires the `mcp` extra.
    """
    mcp_app = None

    if mcp:
        from .mcp import server

        mcp_app = server(system).http_app(path="/")

    api = FastAPI(
        title="cbrkit",
        description="Case-based reasoning over the configured case base.",
        version=version("cbrkit"),
        lifespan=mcp_app.lifespan if mcp_app is not None else None,
    )
    api.include_router(router(system))
    api.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    api.add_api_route(
        "/config",
        lambda: system.config,
        methods=["GET"],
        response_model=Config,
        summary="The specification of the case base",
        description="The specification that every operation follows,"
        " with the language models of the providers that produce the results.",
    )
    api.add_exception_handler(Refused, _refused)
    api.add_exception_handler(Conflict, _conflict)

    if mcp_app is not None:
        api.mount("/mcp", mcp_app)

    api.add_middleware(RequestBodyLimitMiddleware, max_body_size=max_body)
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))

    if token is not None:
        api.add_middleware(_Bearer, token=token)

    return api
