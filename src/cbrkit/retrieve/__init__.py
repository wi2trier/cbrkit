"""Case retrieval with similarity-based, sparse, and embedding-based methods.

This module provides the retrievers, which `cbrkit.cycle` applies to queries.
Retrievers compute similarity scores between a query and cases in a casebase,
then return ranked results.
Multiple retrievers can be composed sequentially (MAC/FAC pattern) or combined
in parallel with score aggregation or rank fusion.

Building Retrievers:
- `Retriever`: Creates a retriever from a similarity function.
  Flattens all batches into pairwise comparisons and optionally parallelizes
  the similarity computations within batches.
- `Embedding`: Retrieves by the similarity of embeddings, embedding every case once per call.
- `BM25`: Retrieves by BM25 text relevance (requires `bm25` extra).
- `fuse`: Merges several rankings of hits into one by reciprocal rank fusion,
  for instance the results of several queries.

Wrappers:
- `Combine`: Merges results from multiple retrievers using an aggregator.
- `Fuse`: Merges the rankings of multiple retrievers by reciprocal rank fusion.
- `Distribute`: Parallelizes retrieval across batches by calling the wrapped
  retriever separately for each (casebase, query) pair.
- `Transpose` / `transpose_value`: Transforms cases/queries before retrieval.

The wrappers are asynchronous and wrap synchronous and asynchronous functions of the cycle alike,
awaiting those with effects and running pure ones in a thread of their own, as `cbrkit.cycle.run_async` does.

Grouped Retrieval (scoring a relation over several cases at once, e.g. whether a
query lies between a pair of cases):
- `as_groups`: Views a casebase as single-case groups, the entry point.
- `Group`: Combines those cases into wider groups inside a retriever
  chain, so a cheap retriever can narrow down the cases before the expansion.

Queries are single-case groups too.
`cbrkit.helpers.broadcast_getter` as the `value_getter` of `cbrkit.sim.AttributeValue`
compares the tuple of values of a group with the query value per attribute,
`cbrkit.sim.collections.IsolatedMapping` compares a query with the closest group member,
and `cbrkit.sim.transpose_singleton` runs a classical measure on single-case groups,
for instance in a first retriever that narrows down the cases before `Group` expands them.
Relations such as betweenness or analogical proportions are domain-specific measures of a group.
Later phases treat a group as an ordinary case, and tuple keys need a conversion before JSON serialization.

`Combine` aggregates similarities and thus assumes them comparable,
while `Fuse` and `fuse` merge rankings, for instance of a dense and a sparse retriever or of several phrasings of one question.
`Retriever` runs chunks of pairs on a `concurrent.futures.Executor` that the caller owns,
such as a `ProcessPoolExecutor`, a `ThreadPoolExecutor` on free-threaded Python, or an `InterpreterPoolExecutor`,
and `Distribute` runs every (casebase, query) batch of a wrapped retriever as a task of its own.

Every ranking function, such as `Retriever`, `Embedding`, `BM25`, `Combine`, `Fuse`, the rerankers,
the storage retrievers, `cbrkit.reuse.Reuser` and `cbrkit.revise.Reviser`,
is a `Ranking` that keeps the best cases by its `limit` and `min_similarity`, after normalizing them if `normalize_scores` is set.

Storage Retrieval:
- The storages of `cbrkit.indexable` (LanceDB, pgvector, sqlite-vec, zvec) are searched
  by their own retrievers, such as `cbrkit.indexable.PgvectorRetriever(storage=...)`.

Re-ranking (`cbrkit.retrieve.rerank`, chain after a retriever):
- `rerank.CrossEncoder`: sentence-transformers cross-encoder (requires `transformers` extra).
- `rerank.HTTP`: HTTP `/rerank` endpoint, e.g. vLLM (requires `http` extra).

Example:
    >>> from cbrkit.cycle import run_query
    >>> from cbrkit.sim.generic import Equality
    >>> retriever = Retriever(measure=Equality(), limit=1)
    >>> result = run_query({"a": "hello", "b": "world"}, "hello", retriever)
    >>> result.ranking[0]
    'a'
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_exports
from .._phase import Ranking
from . import rerank
from ._embedding import Embedding
from .build import Retriever, as_groups
from .fusion import Fuse, fuse
from .wrappers import (
    Combine,
    Distribute,
    Group,
    Transpose,
    transpose_value,
)

if TYPE_CHECKING:
    from ._bm25 import BM25

__all__ = [
    "BM25",
    "Combine",
    "Distribute",
    "Embedding",
    "Fuse",
    "Group",
    "Ranking",
    "Retriever",
    "Transpose",
    "as_groups",
    "fuse",
    "rerank",
    "transpose_value",
]

lazy_exports(__name__)
