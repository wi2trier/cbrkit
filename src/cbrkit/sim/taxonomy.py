"""Similarity of values by their nodes in a taxonomy, see `Taxonomy`.

A taxonomy is a tree of categories whose nodes have the attributes `name`, `weight`, and `children`
(see `SerializedTaxonomyNode`), and a node without weight or children may be given by its name only.
`load` reads a taxonomy from a json or yaml file, which is also the place to change it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol, assert_never, override

from pydantic import BaseModel, Field

from ..helpers import frozendict
from ..io import load as load_file
from ..schema import Family, Unit, serializable
from ..typing import FilePath, SimFunc

__all__ = [
    "Levels",
    "Paths",
    "SerializedTaxonomyNode",
    "Taxonomy",
    "TaxonomyNode",
    "TaxonomySimFunc",
    "TaxonomyStrategy",
    "TaxonomyTree",
    "Weights",
    "WuPalmer",
    "load",
]


class SerializedTaxonomyNode(BaseModel):
    """A node of a taxonomy with its children, as stored in a file.

    Args:
        name: Name of the node, unique in the taxonomy.
        weight: Similarity of two nodes below this one, used by `Weights` with user weights.
        children: The child nodes, each given by a node or its name.
    """

    name: str
    weight: float = 1.0
    children: list[SerializedTaxonomyNode | str] = Field(default_factory=list)


def load(path: FilePath) -> SerializedTaxonomyNode:
    """Loads a taxonomy from a json or yaml file.

    Args:
        path: Path of the file.
    """
    return SerializedTaxonomyNode.model_validate(load_file(path))


@dataclass(slots=True, frozen=True, eq=False)
class TaxonomyNode:
    """A node of a parsed taxonomy, compared by identity.

    Args:
        name: Name of the node.
        weight: Weight of the node.
        depth: Number of edges from the root.
        parent: The parent node, None for the root.
    """

    name: str
    weight: float
    depth: int
    parent: TaxonomyNode | None

    @property
    def level(self) -> int:
        """The 1-based level of the node."""
        return self.depth + 1


@dataclass(slots=True, frozen=True)
class TaxonomyTree:
    """A parsed taxonomy whose nodes are found by their names, see `parse`.

    Args:
        root: The root node.
        nodes: Every node by its name.
        max_depth: The largest depth of a node.
    """

    root: TaxonomyNode
    nodes: Mapping[str, TaxonomyNode]
    max_depth: int

    @classmethod
    def parse(cls, data: SerializedTaxonomyNode | str) -> TaxonomyTree:
        """Parses a serialized taxonomy.

        Raises:
            ValueError: If two nodes have the same name.

        Examples:
            >>> taxonomy = TaxonomyTree.parse(SerializedTaxonomyNode(name="root", children=["a"]))
            >>> taxonomy.nodes["a"].parent is taxonomy.root, taxonomy.max_depth
            (True, 1)
        """
        nodes: dict[str, TaxonomyNode] = {}
        pending: list[tuple[SerializedTaxonomyNode | str, TaxonomyNode | None]] = [
            (data, None)
        ]

        while pending:
            item, parent = pending.pop()
            item = SerializedTaxonomyNode(name=item) if isinstance(item, str) else item

            if item.name in nodes:
                raise ValueError(f"Duplicate taxonomy node: {item.name}")

            node = TaxonomyNode(
                item.name,
                item.weight,
                0 if parent is None else parent.depth + 1,
                parent,
            )
            nodes[node.name] = node
            pending.extend((child, node) for child in item.children)

        root = nodes[data if isinstance(data, str) else data.name]

        return cls(root, frozendict(nodes), max(node.depth for node in nodes.values()))

    def lca(self, node1: TaxonomyNode, node2: TaxonomyNode) -> TaxonomyNode:
        """The lowest common ancestor of two nodes."""
        while node1 is not node2:
            if node1.parent is None or node2.parent is None:
                return self.root

            if node1.depth > node2.depth:
                node1 = node1.parent
            else:
                node2 = node2.parent

        return node1


type TaxonomyStrategy = Literal["optimistic", "pessimistic", "average"]
"""How a measure treats two nodes of which one is the lowest common ancestor of both."""


class TaxonomySimFunc(Protocol):
    """Similarity of two nodes of a taxonomy, given by their names."""

    def __call__(self, case: str, query: str, taxonomy: TaxonomyTree) -> float: ...


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class WuPalmer(TaxonomySimFunc):
    """Wu & Palmer similarity measure of two nodes in a taxonomy.

    Examples:
        >>> sim = Taxonomy(taxonomy=load("./data/cars-taxonomy.yaml"), measure=WuPalmer())
        >>> sim("audi", "porsche")
        0.5
        >>> sim("audi", "bmw")
        0.0
    """

    kind: Literal["wu_palmer"] = field(default="wu_palmer", repr=False)

    @override
    def __call__(self, case: str, query: str, taxonomy: TaxonomyTree) -> float:
        node1 = taxonomy.nodes[case]
        node2 = taxonomy.nodes[query]
        lca = taxonomy.lca(node1, node2)

        if node1.depth + node2.depth == 0:
            return 1.0

        return (2 * lca.depth) / (node1.depth + node2.depth)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Weights(TaxonomySimFunc):
    """Weight-based similarity measure of two nodes in a taxonomy.

    The weights are defined by the user in the taxonomy file or automatically calculated based on the depth of the nodes.

    Args:
        source: The source of the weights to use. One of "auto" or "user".
        strategy: The strategy to use in case one of the node is the lowest common ancestor (lca).
            One of "optimistic", "pessimistic", or "average".

    ![user weights](/assets/taxonomy/user-weights.png)

    ![auto weights](/assets/taxonomy/auto-weights.png)

    Examples:
        >>> sim = Taxonomy(taxonomy=load("./data/cars-taxonomy.yaml"), measure=Weights(source="auto", strategy="optimistic"))
        >>> sim("audi", "Volkswagen AG")
        1.0
        >>> sim("audi", "bmw")
        0.0
    """

    source: Literal["auto", "user"]
    strategy: TaxonomyStrategy
    kind: Literal["weights"] = field(default="weights", repr=False)

    @override
    def __call__(self, case: str, query: str, taxonomy: TaxonomyTree) -> float:
        node1 = taxonomy.nodes[case]
        node2 = taxonomy.nodes[query]
        lca = taxonomy.lca(node1, node2)
        max_depth = taxonomy.max_depth

        if max_depth == 0:
            return 1.0

        weight: float = lca.weight if self.source == "user" else lca.depth / max_depth

        if lca is not node1 and lca is not node2:
            return weight

        match self.strategy, self.source:
            case "optimistic", _:
                return 1.0
            case "pessimistic", _:
                return weight
            case "average", "user":
                return (node1.weight + node2.weight) / 2
            case "average", "auto":
                return (node1.depth + node2.depth) / (2 * max_depth)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Levels(TaxonomySimFunc):
    """Node levels similarity measure of two nodes in a taxonomy.

    The similarity is calculated based on the levels of the nodes.

    Args:
        strategy: The strategy to use in case one of the node is the lowest common ancestor (lca).
            One of "optimistic", "pessimistic", or "average".

    ![node levels](/assets/taxonomy/node-levels.png)

    Examples:
        >>> sim = Taxonomy(taxonomy=load("./data/cars-taxonomy.yaml"), measure=Levels(strategy="optimistic"))
        >>> sim("audi", "Volkswagen AG")
        1.0
        >>> sim("audi", "bmw")
        0.3333333333333333
    """

    strategy: TaxonomyStrategy
    kind: Literal["levels"] = field(default="levels", repr=False)

    @override
    def __call__(self, case: str, query: str, taxonomy: TaxonomyTree) -> float:
        node1 = taxonomy.nodes[case]
        node2 = taxonomy.nodes[query]
        lca = taxonomy.lca(node1, node2)

        match self.strategy:
            case "optimistic":
                return lca.level / min(node1.level, node2.level)
            case "pessimistic":
                return lca.level / max(node1.level, node2.level)
            case "average":
                return lca.level / ((node1.level + node2.level) / 2)
            case _:
                assert_never(self.strategy)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Paths(TaxonomySimFunc):
    """Path steps similarity measure of two nodes in a taxonomy.

    The similarity is calculated based on the steps up and down from the lowest common ancestor (lca).

    Args:
        weight_up: The weight to use for the steps up.
        weight_down: The weight to use for the steps down.

    ![path steps](/assets/taxonomy/path.png)

    Examples:
        >>> sim = Taxonomy(taxonomy=load("./data/cars-taxonomy.yaml"), measure=Paths())
        >>> sim("audi", "Volkswagen AG")
        0.8333333333333334
        >>> sim("audi", "bmw")
        0.3333333333333333
    """

    weight_up: float = 1.0
    weight_down: float = 1.0
    kind: Literal["paths"] = field(default="paths", repr=False)

    @override
    def __call__(self, case: str, query: str, taxonomy: TaxonomyTree) -> float:
        node1 = taxonomy.nodes[case]
        node2 = taxonomy.nodes[query]
        lca = taxonomy.lca(node1, node2)

        steps_up = node1.depth - lca.depth
        steps_down = node2.depth - lca.depth

        weighted_steps = (steps_up * self.weight_up) + (steps_down * self.weight_down)
        max_weighted_steps = (taxonomy.max_depth * self.weight_up) + (
            taxonomy.max_depth * self.weight_down
        )

        if max_weighted_steps == 0:
            return 1.0

        return (max_weighted_steps - weighted_steps) / max_weighted_steps


TAXONOMY_FUNCS = Family(
    "taxonomy function",
    (WuPalmer, Weights, Levels, Paths),
)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Taxonomy(SimFunc[str, float]):
    """Similarity of two values by their nodes in a taxonomy.

    Args:
        taxonomy: The taxonomy, see `load` to read it from a file.
        measure: Similarity of two nodes of the taxonomy.
        default: Similarity of values that are not part of the taxonomy, which raise a `KeyError` if None.

    Examples:
        >>> sim = Taxonomy(taxonomy=load("./data/cars-taxonomy.yaml"), default=0.0)
        >>> sim("audi", "unknown")
        0.0
    """

    taxonomy: SerializedTaxonomyNode
    measure: Annotated[TaxonomySimFunc, TAXONOMY_FUNCS] = WuPalmer()
    default: Unit | None = None
    kind: Literal["taxonomy"] = field(default="taxonomy", repr=False)
    _tree: TaxonomyTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tree", TaxonomyTree.parse(self.taxonomy))

    @override
    def __call__(self, case: str, query: str) -> float:
        for value in (case, query):
            if value not in self._tree.nodes:
                if self.default is None:
                    raise KeyError(f"Value not part of the taxonomy: {value}")

                return self.default

        return self.measure(case, query, self._tree)
