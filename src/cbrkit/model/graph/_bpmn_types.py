"""The element types of the generated BPMN schema classes, which need no xsdata, so inferring a specification needs no `bpmn` extra."""

from collections.abc import Mapping
from functools import cache

from ._bpmn_schema import model as bpmn

__all__ = ["BPMN_NODES", "BpmnHierarchy", "bpmn_class", "bpmn_hierarchy"]

BPMN_NODES: tuple[type[bpmn.TBaseElement], ...] = (
    bpmn.TFlowNode,
    bpmn.TDataObjectReference,
    bpmn.TDataStoreReference,
    bpmn.TParticipant,
)
"""The elements that become nodes: flow nodes, data references, and pools without a process."""

type BpmnHierarchy = tuple[str, tuple[BpmnHierarchy, ...]]
"""The local name of the elements of a generated type and the hierarchies of its subtypes."""


@cache
def _classes() -> Mapping[str, type[bpmn.TBaseElement]]:
    """The generated classes by the local names of their elements and types, which their `Meta.name` holds."""
    return {
        vars(cls)["Meta"].name: cls
        for cls in vars(bpmn).values()
        if isinstance(cls, type) and issubclass(cls, bpmn.TBaseElement)
    }


@cache
def bpmn_class(tag: str | None) -> type[bpmn.TBaseElement]:
    """The generated class of a BPMN element by its local name, such as the `type` of a node or edge.

    Examples:
        >>> issubclass(bpmn_class("userTask"), bpmn.TTask), bpmn_class("unknown")
        (True, <class 'cbrkit.model.graph._bpmn_schema.model.TBaseElement'>)
    """
    return _classes().get(tag or "", bpmn.TBaseElement)


@cache
def _element_names() -> Mapping[type[bpmn.TBaseElement], str]:
    """The local names of the elements of the generated types.

    The generated class of an element derives from its type and has no subclasses.
    """
    return {
        base: name
        for name, cls in _classes().items()
        if not cls.__subclasses__()
        for base in cls.__bases__
        if issubclass(base, bpmn.TBaseElement)
    }


def bpmn_hierarchy(
    cls: type[bpmn.TBaseElement] = bpmn.TBaseElement, /
) -> BpmnHierarchy:
    """The hierarchy of the element types below a generated type that nodes of `from_bpmn` can have, see `BPMN_NODES`.

    Examples:
        >>> name, children = bpmn_hierarchy(bpmn.TGateway)
        >>> name, [child for child, _ in children]
        ('gateway', ['complexGateway', 'eventBasedGateway', 'exclusiveGateway', 'inclusiveGateway', 'parallelGateway'])
    """
    names = _element_names()

    return names[cls], tuple(
        bpmn_hierarchy(sub)
        for sub in cls.__subclasses__()
        if sub in names
        and any(issubclass(sub, node) or issubclass(node, sub) for node in BPMN_NODES)
    )
