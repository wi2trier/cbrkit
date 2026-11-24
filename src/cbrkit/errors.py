"""Exceptions of cbrkit that tell a refused request apart from a failure.

The REST API, the MCP server, and the command line report `Refused` and `Conflict` to the client,
and every other exception as a failure without details,
so a bug in cbrkit or a dependency never passes for an invalid request and never leaks its message.
Programming errors such as a wrong argument type keep the builtin exceptions.
"""

__all__ = ["CbrkitError", "Conflict", "MissingExtra", "Refused", "Unbound"]


class CbrkitError(Exception):
    """Base class of the exceptions of cbrkit."""


class Refused(CbrkitError, ValueError):
    """A request that is invalid for the case base or names an operation the specification does not enable."""


class Conflict(CbrkitError):
    """A storage holds other cases than those a mutation was planned on, so nothing was written."""


class Unbound(CbrkitError, LookupError):
    """A language model or embedder that is referred to by name, but whose name is bound to no function.

    `cbrkit.spec.bind` raises it for every unbound name, so a `cbrkit.system.System` fails on construction.
    """


class MissingExtra(ModuleNotFoundError):
    """An optional dependency that is not installed, the message names the extra that installs it.

    It is a failure of the installation rather than of a request, so it is no `CbrkitError`.
    """
