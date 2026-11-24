"""Core data models for CBR result structures and graph representations.

This module provides the data classes used throughout CBRkit to represent
the results of the CBR cycle, similarity values, explanations and graphs.

Result Classes:
- `QueryResult`: Cases with their `similarities` to one query, their `ranking`
  and the best cases with their similarities as `hits()`.
- `Step`: Outputs of one function for every query, such as a `QueryResult`,
  a read-only mapping of every query key to its output.
- `Result`: Results of every function of the cycle,
  a read-only mapping of every query key to the result of the last function.
- `Hit`: A case as the `value` of its key with its similarity to a query as its `score`.

Similarity Values:
- `AttributeValueSim`, `SequenceSim`, `GraphSim` and `CritiqueSim` are float-like similarities
  with per-attribute scores, alignments, or node and edge mappings,
  subclasses of `cbrkit.typing.Sim` that `SerializedFloat` serializes by their kind in pydantic models.

Changes:
- `Changes` holds the items `Add`, `Update` and `Remove` that retain, maintenance, and applications make to a storage,
  each with its `Rule` and reason, and the `Review` of the stored cases they are based on.
  An `Add` with a key inserts the case under it and conflicts with a stored case of that key,
  one without a key gets a key that the storage allocates.
  A `Planner` such as `cbrkit.maintain.Maintainer` plans purely from the stored cases and the inputs of a run,
  `Changes.apply` writes the changes as one mutation of a storage,
  and `replan` reads the storage, plans, and applies, planning again if the storage changed in between.

Explanations:
- `Explanation` and `GraphExplanation` break a similarity down for user interfaces,
  see `cbrkit.sim.explain`.

Graph Module:
- The `cbrkit.model.graph` submodule provides a `Graph` data structure
  with `Node` and `Edge` types, validating serialization helpers (`to_dict`,
  `from_dict`, `from_json`), BPMN 2.0 documents (`parse_bpmn`, `render_bpmn`) and their graphs
  (`from_bpmn`, `bpmn_document`) with the `bpmn` extra, and conversions to NetworkX, RustWorkX, and Graphviz.

Example:
    >>> result = QueryResult(
    ...     query="my query",
    ...     casebase={"c1": "case1", "c2": "case2"},
    ...     similarities={"c1": 0.9, "c2": 0.5},
    ... )
    >>> result.ranking
    ('c1', 'c2')
    >>> result.hits(limit=1)[0].value
    'case1'
"""

from . import case, changes, explanation, graph, result, similarity
from .case import Case, problem, solution, with_solution
from .changes import (
    Add,
    Category,
    Change,
    Changes,
    Planner,
    Remove,
    Review,
    Rule,
    Update,
    replan,
)
from .explanation import Explanation, GraphExplanation
from .result import Hit, QueryResult, Result, Step
from .similarity import (
    Alignment,
    AttributeValueSim,
    ChangeSim,
    CritiqueSim,
    GraphSim,
    SequenceSim,
    SerializedFloat,
)

__all__ = [
    "Add",
    "Alignment",
    "AttributeValueSim",
    "Case",
    "Category",
    "Change",
    "ChangeSim",
    "Changes",
    "CritiqueSim",
    "Explanation",
    "GraphExplanation",
    "GraphSim",
    "Hit",
    "Planner",
    "QueryResult",
    "Remove",
    "Result",
    "Review",
    "Rule",
    "SequenceSim",
    "SerializedFloat",
    "Step",
    "Update",
    "case",
    "changes",
    "explanation",
    "graph",
    "problem",
    "replan",
    "result",
    "similarity",
    "solution",
    "with_solution",
]
