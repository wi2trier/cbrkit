"""The operations of a system as plain typed async functions, one table for REST, MCP, agents, and the command line.

`operations` turns every operation that the specification of a system enables into an async function
that runs the method of `cbrkit.system.System` of the same name and returns a `cbrkit.system.Outcome`.
Its name, docstring, and signature come from the method, and the signature is typed with the models
of the cases derived from the specification, the bounds of `limit`, and the outcome model named after the operation,
such as `RetrieveOutcome`, so the frameworks that read signatures accept the functions as they are:
FastMCP `Tool.from_function` or pydantic-ai `FunctionToolset`.
The attributes `read_only` and `destructive` mark the operations that write the case base
and those that may replace or remove stored cases, and `request` is the model of the arguments,
which the REST API and the command line validate requests with.

Example:
    >>> import asyncio
    >>> from cbrkit.indexable import InMemory
    >>> from cbrkit.io import Source
    >>> from cbrkit.spec import infer
    >>> cases = {0: {"price": 10.0}, 1: {"price": 20.0}}
    >>> system = System(config=infer(cases, Source(format="json")), storage=InMemory(cases))
    >>> retrieve = operations(system)[0]
    >>> retrieve.__name__, retrieve.read_only, retrieve.destructive
    ('retrieve', True, False)
    >>> [hit.key for hit in asyncio.run(retrieve(query={"price": 18.0}, limit=1)).hits]
    [1]
"""

import inspect
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from types import GenericAlias
from typing import Annotated, Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..model import Add, Changes, Hit, Planner, Remove, Review, SerializedFloat, Update
from ..schema import parse_docstring
from ..spec import Config, Phase, case_model
from .core import Key, Outcome, System

__all__ = ["OPERATIONS", "Method", "Operation", "operations"]


class Method(Protocol):
    """A method of `System` that runs an operation, bound to a system or not."""

    @property
    def __name__(self) -> str: ...

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Coroutine[Any, Any, Outcome[Any]]: ...


def _reads(_: Config) -> Planner[Any, Any] | None:
    return None


@dataclass(slots=True, frozen=True)
class _Entry:
    """An operation with the sections of a specification that enable it and that write the case base.

    Args:
        method: The method of `System` that runs it.
        enables: The section that enables it, disabled if None.
        writes: The section that plans the writes to the case base, read-only if None.
    """

    method: Method
    enables: Callable[[Config], Phase | None]
    writes: Callable[[Config], Planner[Any, Any] | None] = _reads


_ENTRIES: tuple[_Entry, ...] = (
    _Entry(System.retrieve, lambda config: config.retrieve),
    _Entry(System.reuse, lambda config: config.reuse),
    _Entry(System.revise, lambda config: config.revise),
    _Entry(System.retain, lambda config: config.retain, lambda config: config.retain),
    _Entry(
        System.cycle,
        lambda config: config.retrieve,
        lambda config: config.retain if config.cycle_retains else None,
    ),
    _Entry(
        System.maintain, lambda config: config.maintain, lambda config: config.maintain
    ),
)

OPERATIONS: tuple[Method, ...] = tuple(entry.method for entry in _ENTRIES)
"""Every operation as its method of `System`:
the four phases of the CBR cycle, the whole cycle, and the maintenance of the case base outside the cycle."""


class Operation(Protocol):
    """An operation of a system, a typed async function that runs a method of `System`.

    Attributes:
        read_only: Whether the operation leaves the case base unchanged.
        destructive: Whether the operation may replace or remove stored cases,
            see `cbrkit.model.Planner.destructive`,
            unless the limits of the specification enforce a dry run.
        request: The model of the arguments, which rejects unknown ones, named after the operation, such as `RetrieveRequest`.
    """

    __name__: str
    read_only: bool
    destructive: bool
    request: type[BaseModel]

    async def __call__(self, *args: Any, **kwargs: Any) -> Outcome[Any]: ...


def _concrete(generic: Any, *params: Any, **fields: Any) -> type[BaseModel]:
    return create_model(generic.__name__, __base__=generic[params], **fields)


def _outcome(case: type[BaseModel]) -> type[BaseModel]:
    """The `Outcome` with the cases of a specification, of concrete models named like their classes.

    A parametrized generic model would be named with its type arguments in OpenAPI and MCP schemas,
    such as `Hit_Key_Case_Float_`.
    """
    change = cast(Any, Annotated)[
        _concrete(Add, Key, case)
        | _concrete(Update, Key, case)
        | _concrete(Remove, Key),
        Field(discriminator="op"),
    ]
    changes = _concrete(
        Changes,
        Key,
        case,
        items=(GenericAlias(Sequence, (change,)), ()),
        reviews=(GenericAlias(Sequence, (_concrete(Review, Key),)), ()),
    )

    return _concrete(
        Outcome,
        case,
        hits=(
            GenericAlias(Sequence, (_concrete(Hit, Key, case, SerializedFloat),)),
            (),
        ),
        changes=(changes | None, None),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True)

    if isinstance(value, list):
        return [_plain(item) for item in value]

    return value


def operations(system: System[Any]) -> list[Operation]:
    """The operations that the specification of a system enables, typed with the models of its cases.

    Args:
        system: The system that runs the operations.
    """
    query = case_model(system.config)
    types: dict[str, Any] = {
        "query": query,
        "solutions": GenericAlias(list, (query,)),
        "limit": Annotated[int, Field(ge=1, le=system.config.limits.max_limit)],
        "uses": dict[Key, Annotated[int, Field(ge=0)]],
    }
    outcome = _outcome(case_model(system.config, stored=True))

    return [
        _operation(
            getattr(system, entry.method.__name__),
            entry.writes(system.config),
            system.config.limits.dry_run,
            types,
            outcome,
        )
        for entry in _ENTRIES
        if entry.enables(system.config) is not None
    ]


def _annotation(parameter: inspect.Parameter, types: Mapping[str, Any]) -> Any:
    if parameter.name not in types:
        return parameter.annotation

    if parameter.default is None:
        return types[parameter.name] | None

    return types[parameter.name]


def _operation(
    method: Method,
    planner: Planner[Any, Any] | None,
    dry_run: bool,
    types: Mapping[str, Any],
    outcome: type[BaseModel],
) -> Operation:
    name = method.__name__
    description, docs = parse_docstring(inspect.getdoc(method) or "")
    parameters = [
        parameter.replace(
            annotation=cast(Any, Annotated)[
                _annotation(parameter, types),
                Field(description=docs.get(parameter.name)),
            ]
        )
        for parameter in inspect.signature(method).parameters.values()
    ]
    returns = cast(
        type[Outcome[Any]],
        create_model(f"{name.title()}Outcome", __base__=outcome),
    )
    signature = inspect.Signature(parameters, return_annotation=returns)
    fields: dict[str, Any] = {
        parameter.name: (
            parameter.annotation,
            ... if parameter.default is inspect.Parameter.empty else parameter.default,
        )
        for parameter in parameters
    }
    request = create_model(
        f"{name.title()}Request", __config__=ConfigDict(extra="forbid"), **fields
    )

    async def operation(*args: Any, **kwargs: Any) -> Outcome[Any]:
        arguments = signature.bind(*args, **kwargs).arguments
        result = await method(
            **{key: _plain(value) for key, value in arguments.items()}
        )

        # the result holds the cases as stored, the declared model those of the specification
        return returns.model_validate(result.model_dump())

    operation.__name__ = operation.__qualname__ = name
    operation.__doc__ = description
    operation.__annotations__ = {
        **{parameter.name: parameter.annotation for parameter in parameters},
        "return": returns,
    }
    vars(operation).update(
        __signature__=signature,
        read_only=planner is None,
        destructive=planner is not None and not dry_run and planner.destructive,
        request=request,
    )

    return cast(Operation, operation)
