from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, cast

from cbrkit import load, model


class SerializedNode(TypedDict, total=False):
    key: str
    weight: float
    children: dict[str, "SerializedNode"]


TaxonomyMeasureName = Literal["wu_palmer"]
TaxonomyMeasureFunc = Callable[["Taxonomy", str, str], model.SimilarityValue]
TaxonomyMeasure = TaxonomyMeasureName | TaxonomyMeasureFunc


@dataclass
class TaxonomyNode:
    key: str
    weight: float | None
    depth: int
    parent: "TaxonomyNode" | None
    children: dict[str, "TaxonomyNode"] = field(default_factory=dict)


class Taxonomy:
    root: TaxonomyNode
    nodes: dict[str, TaxonomyNode]

    def __init__(self, path: model.FilePath) -> None:
        if isinstance(path, str):
            path = Path(path)

        root_data = cast(SerializedNode, load.data_loaders[path.suffix](path))
        self.root = self._load(root_data)

    def _load(
        self, data: SerializedNode, parent: TaxonomyNode | None = None, depth: int = 0
    ) -> TaxonomyNode:
        assert "key" in data, "Missing key in some node"

        node = TaxonomyNode(
            key=data["key"],
            weight=data.get("weight"),
            depth=depth,
            parent=parent,
        )

        for key, child in data.get("children", {}).items():
            node.children[key] = self._load(child, node, depth + 1)

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

    def similarity(
        self,
        key1: str,
        key2: str,
        measure: TaxonomyMeasure | None = None,
    ) -> model.SimilarityValue:
        if measure is None:
            measure = "wu_palmer"

        if isinstance(measure, str):
            measure = measures[measure]

        return measure(self, key1, key2)


def wu_palmer(taxonomy: Taxonomy, key1: str, key2: str) -> model.SimilarityValue:
    node1 = taxonomy.nodes[key1]
    node2 = taxonomy.nodes[key2]

    lca = taxonomy.lca(node1, node2)

    return (2 * lca.depth) / (node1.depth + node2.depth)


measures: dict[str, TaxonomyMeasureFunc] = {
    "wu_palmer": wu_palmer,
}