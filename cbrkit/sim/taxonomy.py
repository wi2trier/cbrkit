from dataclasses import dataclass, field
from typing import Optional, Protocol, TypedDict, cast

from cbrkit.loaders import data as load_data
from cbrkit.typing import FilePath, SimPairFunc


class SerializedNode(TypedDict, total=False):
    key: str
    sim: float
    children: list["SerializedNode | str"]


@dataclass
class TaxonomyNode:
    key: str
    weight: float | None
    depth: int
    parent: Optional["TaxonomyNode"]
    children: dict[str, "TaxonomyNode"] = field(default_factory=dict)


class Taxonomy:
    root: TaxonomyNode
    nodes: dict[str, TaxonomyNode]

    def __init__(self, path: FilePath) -> None:
        root_data = cast(SerializedNode, load_data(path))
        self.nodes = {}
        self.root = self._load(root_data)

    def _load(
        self,
        data: SerializedNode | str,
        parent: TaxonomyNode | None = None,
        depth: int = 0,
    ) -> TaxonomyNode:
        if isinstance(data, str):
            data = {"key": data}

        assert "key" in data, "Missing key in some node"

        node = TaxonomyNode(
            key=data["key"],
            weight=data.get("weight"),
            depth=depth,
            parent=parent,
        )

        for child in data.get("children", []):
            child_node = self._load(child, node, depth + 1)
            node.children[child_node.key] = child_node

        self.nodes[node.key] = node

        return node

    def lca(self, node1: TaxonomyNode, node2: TaxonomyNode) -> TaxonomyNode:
        while node1 != node2:
            if node1.parent is None or node2.parent is None:
                return self.root

            if node1.depth > node2.depth:
                node1 = node1.parent
            else:
                node2 = node2.parent

        return node1


class TaxonomyFunc(Protocol):
    def __call__(self, taxonomy: Taxonomy, x: str, y: str) -> float:
        ...


def wu_palmer() -> TaxonomyFunc:
    def wrapped_func(taxonomy: Taxonomy, x: str, y: str) -> float:
        node1 = taxonomy.nodes[x]
        node2 = taxonomy.nodes[y]
        lca = taxonomy.lca(node1, node2)

        return (2 * lca.depth) / (node1.depth + node2.depth)

    return wrapped_func


_taxonomy_func = wu_palmer()


def load(
    path: FilePath, measure: TaxonomyFunc = _taxonomy_func
) -> SimPairFunc[str, float]:
    taxonomy = Taxonomy(path)

    def wrapped_func(x: str, y: str) -> float:
        return measure(taxonomy, x, y)

    return wrapped_func
