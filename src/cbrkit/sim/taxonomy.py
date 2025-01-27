"""
A taxonomy is a hierarchical structure of categories, where each category is a node in the taxonomy.
The nodes are connected by parent-child relationships, and the root node is the top-level category.
Each node may have the attributes `name`, `weight`, and `children` (see `cbrkit.sim.taxonomy.SerializedTaxonomyNode`).
To simplify the creation of a taxonomy, the `cbrkit.sim.taxonomy.load` function can be used to load a taxonomy from a file (toml, json, or yaml).
For nodes without a `weight` or `children`, it is also possible to pass its name as a string instead of a dictionary.

**Important:** If loading the taxonomy from a file, changes like adding children or weights need to be made there, not in the code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, override

from pydantic import BaseModel, Field

from ..loaders import file as load_file

__all__ = [
    "SerializedTaxonomyNode",
    "TaxonomyNode",
    "Taxonomy",
    "TaxonomyStrategy",
    "TaxonomySimFunc",
    "wu_palmer",
    "weights",
    "levels",
    "paths",
    "build",
]


class SerializedTaxonomyNode(BaseModel):
    name: str
    weight: float = 1.0
    children: list["SerializedTaxonomyNode | str"] = Field(default_factory=list)


SerializedTaxonomyNode.model_rebuild()


@dataclass(slots=True)
class TaxonomyNode:
    name: str
    weight: float
    depth: int
    parent: "TaxonomyNode | None"
    children: dict[str, "TaxonomyNode"] = field(default_factory=dict)

    @property
    def level(self) -> int:
        return self.depth + 1


@dataclass(slots=True)
class Taxonomy:
    """Load a taxonomy and return a function that measures the similarity.

    The taxonomy is loaded from the given path and expected to conform to the following structure:

    ```yaml
    name: ROOT
    weight: 0.0
    children:
      - name: CHILD1
        weight: 0.4
        children:
          - name: GRANDCHILD1
            weight: 0.8
            children:
              - name: GREATGRANDCHILD1
              - name: GREATGRANDCHILD2
          - name: GRANDCHILD2
      - name: CHILD2
        weight: 0.6
        children: []
    ```

    The `name` field is required for each node, and the `children` field is optional.
    The `weight` field is optional and can be used to assign a weight/similarity value to the node.
    If not set, the default value is 1.0.
    The `weight` is only used by the measure `user_weights` and ignored otherwise.
    """

    root_name: str
    nodes: dict[str, TaxonomyNode]

    def __init__(self, data: SerializedTaxonomyNode) -> None:
        self.nodes = {}
        self.root_name = self.parse(data).name

    @property
    def root(self) -> TaxonomyNode:
        return self.nodes[self.root_name]

    @property
    def max_depth(self) -> int:
        return max(node.depth for node in self.nodes.values())

    @property
    def max_level(self) -> int:
        return max(node.level for node in self.nodes.values())

    def parse(
        self,
        data: SerializedTaxonomyNode | str,
        parent: TaxonomyNode | None = None,
        depth: int = 0,
    ) -> TaxonomyNode:
        if isinstance(data, str):
            data = SerializedTaxonomyNode(name=data)

        node = TaxonomyNode(
            name=data.name,
            weight=data.weight,
            depth=depth,
            parent=parent,
        )

        for child in data.children:
            child_node = self.parse(child, node, depth + 1)
            node.children[child_node.name] = child_node

        self.nodes[node.name] = node

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


TaxonomyStrategy = Literal["optimistic", "pessimistic", "average"]


@dataclass(slots=True, frozen=True)
class TaxonomySimFunc(Protocol):
    def __call__(self, x: str, y: str, taxonomy: Taxonomy) -> float: ...


@dataclass(slots=True, frozen=True)
class wu_palmer(TaxonomySimFunc):
    """Wu & Palmer similarity measure of two nodes in a taxonomy.

    Examples:
        >>> sim = build("./data/cars-taxonomy.yaml", wu_palmer())
        >>> sim("audi", "porsche")
        0.5
        >>> sim("audi", "bmw")
        0.0
    """

    @override
    def __call__(self, x: str, y: str, taxonomy: Taxonomy) -> float:
        node1 = taxonomy.nodes[x]
        node2 = taxonomy.nodes[y]
        lca = taxonomy.lca(node1, node2)

        return (2 * lca.depth) / (node1.depth + node2.depth)


@dataclass(slots=True, frozen=True)
class weights(TaxonomySimFunc):
    """Weight-based similarity measure of two nodes in a taxonomy.

    The weights are defined by the user in the taxonomy file or automatically calculated based on the depth of the nodes.

    Args:
        source: The source of the weights to use. One of "auto" or "user".
        strategy: The strategy to use in case one of the node is the lowest common ancestor (lca).
            One of "optimistic", "pessimistic", or "average".

    ![user weights](../../../../assets/taxonomy/user-weights.png)

    ![user weights](../../../../assets/taxonomy/auto-weights.png)

    Examples:
        >>> sim = build("./data/cars-taxonomy.yaml", weights("auto", "optimistic"))
        >>> sim("audi", "Volkswagen AG")
        1.0
        >>> sim("audi", "bmw")
        0.0
    """

    source: Literal["auto", "user"]
    strategy: TaxonomyStrategy

    @override
    def __call__(self, x: str, y: str, taxonomy: Taxonomy) -> float:
        node1 = taxonomy.nodes[x]
        node2 = taxonomy.nodes[y]
        lca = taxonomy.lca(node1, node2)
        max_depth = taxonomy.max_depth

        weight: float = lca.weight if self.source == "user" else lca.depth / max_depth

        if lca == node1 or lca == node2:
            # pessimistic not needed: weight of lca already used
            if self.strategy == "optimistic":
                weight = 1.0
            elif self.strategy == "average" and self.source == "user":
                weight = (node1.weight + node2.weight) / 2
            elif self.strategy == "average" and self.source == "auto":
                weight1 = node1.depth / max_depth
                weight2 = node2.depth / max_depth
                weight = (weight1 + weight2) / 2

        return weight


@dataclass(slots=True, frozen=True)
class levels(TaxonomySimFunc):
    """Node levels similarity measure of two nodes in a taxonomy.

    The similarity is calculated based on the levels of the nodes.

    Args:
        strategy: The strategy to use in case one of the node is the lowest common ancestor (lca).
            One of "optimistic", "pessimistic", or "average".

    ![node levels](../../../../assets/taxonomy/node-levels.png)

    Examples:
        >>> sim = build("./data/cars-taxonomy.yaml", levels("optimistic"))
        >>> sim("audi", "Volkswagen AG")
        1.0
        >>> sim("audi", "bmw")
        0.3333333333333333
    """

    strategy: TaxonomyStrategy

    @override
    def __call__(self, x: str, y: str, taxonomy: Taxonomy) -> float:
        node1 = taxonomy.nodes[x]
        node2 = taxonomy.nodes[y]
        lca = taxonomy.lca(node1, node2)

        if self.strategy == "optimistic":
            return lca.level / min(node1.level, node2.level)
        elif self.strategy == "pessimistic":
            return lca.level / max(node1.level, node2.level)
        elif self.strategy == "average":
            return lca.level / ((node1.level + node2.level) / 2)

        return 0.0


@dataclass(slots=True, frozen=True)
class paths(TaxonomySimFunc):
    """Path steps similarity measure of two nodes in a taxonomy.

    The similarity is calculated based on the steps up and down from the lowest common ancestor (lca).

    Args:
        weight_up: The weight to use for the steps up.
        weight_down: The weight to use for the steps down.

    ![path steps](../../../../assets/taxonomy/path.png)

    Examples:)
        >>> sim = build("./data/cars-taxonomy.yaml", paths())
        >>> sim("audi", "Volkswagen AG")
        0.8333333333333334
        >>> sim("audi", "bmw")
        0.3333333333333333
    """

    weight_up: float = 1.0
    weight_down: float = 1.0

    @override
    def __call__(self, x: str, y: str, taxonomy: Taxonomy) -> float:
        node1 = taxonomy.nodes[x]
        node2 = taxonomy.nodes[y]
        lca = taxonomy.lca(node1, node2)

        steps_up = node1.depth - lca.depth
        steps_down = node2.depth - lca.depth

        weighted_steps = (steps_up * self.weight_up) + (steps_down * self.weight_down)
        max_weighted_steps = (taxonomy.max_depth * self.weight_up) + (
            taxonomy.max_depth * self.weight_down
        )

        return (max_weighted_steps - weighted_steps) / max_weighted_steps


default_taxonomy_func = wu_palmer()


@dataclass(slots=True, init=False)
class build:
    """Build a taxonomy similarity function.

    Args:
        taxonomy: The taxonomy to use for the similarity measure.

    Returns:
        A function that measures the similarity between two nodes in the taxonomy.
    """

    taxonomy: Taxonomy = field(repr=False)
    func: TaxonomySimFunc

    def __init__(
        self,
        taxonomy: Taxonomy | SerializedTaxonomyNode | str | Path,
        func: TaxonomySimFunc = default_taxonomy_func,
    ) -> None:
        if isinstance(taxonomy, Taxonomy):
            self.taxonomy = taxonomy
        elif isinstance(taxonomy, str | Path):
            loaded_content = load_file(taxonomy)
            serialized_taxonomy = SerializedTaxonomyNode.model_validate(loaded_content)
            self.taxonomy = Taxonomy(serialized_taxonomy)
        else:
            self.taxonomy = Taxonomy(taxonomy)

        self.func = func

    def __call__(self, x: str, y: str) -> float:
        return self.func(x, y, self.taxonomy)
