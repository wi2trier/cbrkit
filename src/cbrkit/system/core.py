"""A CBR system built from a specification, see `System`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..cycle import run_query_async
from ..errors import Refused
from ..helpers import to_thread
from ..io import FileStorage, companion
from ..model import Changes, Hit, Planner, QueryResult, SerializedFloat, replan
from ..model.graph import Graph
from ..retain import Retainer, changed
from ..retrieve import Retriever
from ..reuse import Reuser
from ..revise import Reviser
from ..schema import Model
from ..sim import Measure
from ..sim import explain as explain_similarity
from ..spec import Config, Embedders, Models, Phase, bind, providers, read
from ..typing import Casebase, FilePath, Storage

__all__ = [
    "Key",
    "Outcome",
    "System",
]

type Key = int | str
"""Key of a case."""

type _Step[V] = Retriever[V, SerializedFloat] | Reuser[V, Any] | Reviser[V, Any]


class Outcome[C](Model):
    """The result of every operation of a system.

    `model_dump(mode="json")` gives plain JSON, with graph cases in their serialized form,
    which the REST API, the MCP server, and the command line return.

    Args:
        hits: Cases by descending similarity to the query, such as retrieved cases or accepted solutions,
            keyed by their position in the request, or the cases as stored by a retain run.
        changes: Changes of a retain or maintenance run with their reasons
            and the review of the stored cases, None for the other operations.
    """

    hits: Sequence[Hit[Key, C, SerializedFloat]] = ()
    changes: Changes[Key, C] | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class System[V]:
    """A CBR system built from a specification that retrieves, reuses, revises, retains, and maintains cases.

    A phase without its section is off, so its operation does what the cycle does without it:
    reuse proposes the retrieved cases as they are, revise accepts every solution,
    and retain and maintain change nothing, so the case base is only written with a `retain` or `maintain` section.
    The methods are the four phases after Aamodt and Plaza (1994), the whole cycle,
    and the maintenance of the case base outside the cycle,
    take the query first if they need one, and return the same `Outcome`: hits and the changes of a run.
    They are asynchronous, since they read and write the storage and may embed or generate,
    and run the pure phases in a worker thread, see `cbrkit.cycle.run_async`.
    The configuration, with the language models of the providers, describes how they were produced.
    They take and return cases as the storage holds them, such as mappings or graphs,
    of type `V`, which pydantic validates from and serializes to JSON,
    so they serve as the operations of `cbrkit.system.api`, `cbrkit.system.mcp`, and the command line.

    Args:
        config: The specification with the similarity and the sections of the cycle.
        storage: The cases, such as `cbrkit.io.FileStorage` or `cbrkit.indexable.InMemory`.
        embedders: Embedding functions by the names the measures refer to, see `cbrkit.spec.bind`.
        models: Language models by the names the measures and adaptations refer to, see `cbrkit.spec.bind`.

    Raises:
        cbrkit.errors.Unbound: If an embedder or language model of the specification is bound to no function.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.io import Source
        >>> from cbrkit.spec import infer
        >>> cases = {0: {"price": 10.0}, 1: {"price": 20.0}, 2: {"price": 40.0}}
        >>> system = System(config=infer(cases, Source(format="json")), storage=InMemory(cases))
        >>> outcome = asyncio.run(system.retrieve({"price": 18.0}, limit=2))
        >>> [hit.key for hit in outcome.hits]
        [1, 0]
        >>> outcome.model_dump(mode="json")["hits"][0]["value"]
        {'price': 20.0}
        >>> [hit.key for hit in asyncio.run(system.retrieve(key=0, limit=1)).hits]
        [1]
    """

    config: Config
    storage: Storage[Any, V]
    embedders: Embedders = field(default_factory=dict)
    models: Models = field(default_factory=dict)
    _bound: Config = field(init=False, repr=False)
    _retainer: Retainer[Any, V] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bound = bind(self.config, self.embedders, self.models)
        object.__setattr__(self, "_bound", bound)
        object.__setattr__(self, "_retainer", bound.retainer(self.storage))

    @classmethod
    def load(
        cls,
        casebase: FilePath,
        config: FilePath | None = None,
        embedders: Embedders | None = None,
        models: Models | None = None,
    ) -> System[Any]:
        """Loads a case base with its sidecar file and binds the providers of its specification, see `cbrkit.spec.providers`.

        Args:
            casebase: Path of the case base file or folder.
            config: Path of the sidecar, `<casebase>.cbrkit.json` if None.
            embedders: Embedding functions of the host by name, which take precedence over the providers section.
            models: Language models of the host by name, which take precedence over the providers section.
        """
        spec = read(config or companion(casebase, "sidecar"))
        embedders, models = providers(spec, embedders, models, casebase=casebase)

        return cls(
            config=spec,
            storage=FileStorage(path=Path(casebase), source=spec.source),
            embedders=embedders,
            models=models,
        )

    async def _outcome(
        self,
        result: QueryResult[Any, V, SerializedFloat] | None = None,
        changes: Changes[Any, V] | None = None,
        explain: Measure[V, Any] | None = None,
    ) -> Outcome[V]:
        """The outcome of a run, whose hits `explain` breaks down if given, the measure of the phase that scored them."""
        hits = [] if result is None else result.hits()

        if explain is not None and result is not None:
            hits = await to_thread(self._explain, hits, result.query, explain)

        return Outcome(hits=hits, changes=changes)

    def _explain(
        self, hits: Sequence[Hit[Any, V, SerializedFloat]], query: V, measure: Measure[V, Any]
    ) -> list[Hit[Any, V, SerializedFloat]]:
        return [
            hit.model_copy(
                update={
                    "explanation": explain_similarity(
                        hit.value, query, hit.score, measure
                    )
                }
            )
            for hit in hits
        ]

    def _solutions(self, solutions: Sequence[V]) -> dict[int, V]:
        if len(solutions) > self.config.limits.max_limit:
            raise Refused("There are more solutions than the maximal limit")

        return dict(enumerate(solutions))

    def _retriever(self, limit: int | None) -> Retriever[V, SerializedFloat]:
        retriever = self._bound.phase(self._bound.retrieve)
        bound = min(
            limit or retriever.limit or self.config.limits.max_limit,
            self.config.limits.max_limit,
        )

        return replace(retriever, limit=bound)

    def _steps(
        self, limit: int | None, *sections: Reuser[V, Any] | Reviser[V, Any] | None
    ) -> list[_Step[V]]:
        """The retriever of the limit and the configured sections, in order and with their measures filled in."""
        return [
            self._retriever(limit),
            *(self._bound.phase(section) for section in sections if section is not None),
        ]

    def _check_cost(self, casebase: Casebase[Any, V], *queries: V) -> None:
        """Refuses comparing the queries with the cases if their graphs compare more node pairs than the limits allow."""
        budget = self.config.limits.max_node_pairs

        if budget is None:
            return

        # Every query meets every case, so the pairs are the product of their node counts.
        pairs = sum(len(q.nodes) for q in queries if isinstance(q, Graph)) * sum(
            len(case.nodes) for case in casebase.values() if isinstance(case, Graph)
        )

        if pairs > budget:
            raise Refused(f"The graphs compare {pairs} node pairs, more than {budget}")

    async def _resolve(
        self, query: V | None, key: Key | None
    ) -> tuple[Casebase[Any, V], V]:
        """The cases to compare and the query, a stored case if a key is given, which is left out of the cases.

        Raises:
            cbrkit.errors.Refused: If neither or both of query and key are given, the key is unknown,
                or a graph query compares more node pairs than the limits allow.
        """
        stored = await self.storage.get_index()

        if query is not None and key is None:
            casebase = stored
        elif query is None and key is not None:
            if key not in stored:
                raise Refused(f"There is no case {key!r}")

            query = stored[key]
            casebase = {k: v for k, v in stored.items() if k != key}
        else:
            raise Refused("Pass either a query or the key of a stored case")

        self._check_cost(casebase, query)

        return casebase, query

    async def _changes[**P](
        self,
        planner: Planner[Any, V, P],
        dry_run: bool,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Changes[Any, V]:
        """The changes of a run, applied with `cbrkit.model.replan` or only planned in a dry run."""
        if dry_run:
            stored = await self.storage.get_index()

            return await to_thread(planner.plan, stored, *args, **kwargs)

        return await replan(planner, self.storage, *args, **kwargs)

    async def retrieve(
        self,
        query: V | None = None,
        key: Key | None = None,
        limit: int | None = None,
        explain: bool = False,
    ) -> Outcome[V]:
        """Retrieves the cases that are most similar to a query or to a stored case.

        Args:
            query: The query, attributes that are left out do not count.
            key: Key of a stored case to use as the query instead, which is left out of the hits.
            limit: Maximal number of cases, the configured limit if None.
            explain: Break the similarity of every case down by attribute or graph element.

        Raises:
            cbrkit.errors.Refused: If neither or both of query and key are given, the key is unknown,
                or a graph query compares more node pairs than the limits allow.
        """
        casebase, query = await self._resolve(query, key)
        retriever = self._retriever(limit)
        result = await run_query_async(casebase, query, retriever)

        return await self._outcome(result, explain=_measure(retriever, explain))

    async def reuse(
        self,
        query: V | None = None,
        key: Key | None = None,
        limit: int | None = None,
        explain: bool = False,
    ) -> Outcome[V]:
        """Retrieves the most similar cases and adapts them to a query or to a stored case.

        Args:
            query: The query, attributes that are left out do not count.
            key: Key of a stored case to use as the query instead, which is left out of the hits.
            limit: Maximal number of cases, the configured limit if None.
            explain: Break the similarity of every case down by attribute or graph element.

        Without a reuse section, the retrieved cases are proposed as they are.

        Raises:
            cbrkit.errors.Refused: If neither or both of query and key are given, the key is unknown,
                or a graph query compares more node pairs than the limits allow.
        """
        casebase, query = await self._resolve(query, key)
        steps = self._steps(limit, self._bound.reuse)
        result = await run_query_async(casebase, query, *steps)

        return await self._outcome(result, explain=_measure(steps[-1], explain))

    async def revise(
        self, query: V, solutions: Sequence[V], explain: bool = False
    ) -> Outcome[V]:
        """Repairs and assesses solutions of a query, keeping the accepted ones.

        Args:
            query: The query that the solutions solve.
            solutions: The solutions, keyed by their position in the hits.
            explain: Break the similarity of every solution down by attribute or graph element.

        Without a revise section, every solution is accepted and ranked by its similarity to the query.

        Raises:
            cbrkit.errors.Refused: If there are more solutions than the maximal limit of retrieve.
        """
        reviser = self._bound.phase(self._bound.revise or Retriever())
        casebase = self._solutions(solutions)
        self._check_cost(casebase, query)
        result = await run_query_async(casebase, query, reviser)

        return await self._outcome(result, explain=_measure(reviser, explain))

    async def retain(
        self, query: V, solutions: Sequence[V], dry_run: bool = False
    ) -> Outcome[V]:
        """Stores solutions of a query in the case base as they are, such as solutions that revise accepted or a person confirmed.

        The changes add them or merge them into the stored cases they duplicate, each change with its reason,
        and carry the keys of the stored cases.
        If another run changes the case base in the meantime, the run is planned again.

        Args:
            query: The query that the solutions solve.
            solutions: The solutions to store.
            dry_run: Plan without changing the case base, the configuration may enforce it.

        Without a retain section, nothing is stored.

        Raises:
            cbrkit.errors.Refused: If there are more solutions than the maximal limit,
                or comparing graph solutions with the stored cases compares more node pairs than the limits allow.
        """
        if self._retainer is None:
            return await self._outcome(changes=Changes())

        solved = self._solutions(solutions)
        changes = await self._store(self._retainer, solved, query, dry_run)

        return await self._outcome(changes=changes)

    async def _store(
        self,
        retainer: Retainer[Any, V],
        solutions: Casebase[Any, V],
        query: V,
        dry_run: bool,
        stored: Casebase[Any, V] | None = None,
    ) -> Changes[Any, V]:
        """The changes of storing solutions of a query, applied or only planned in a dry run.

        Args:
            stored: The cases the solutions are compared with for duplicates, read from the storage if None.
        """
        if retainer.duplicate is not None:
            cases = await self.storage.get_index() if stored is None else stored
            self._check_cost(cases, *solutions.values())

        return await self._changes(
            retainer, dry_run or self.config.limits.dry_run, [(solutions, query)]
        )

    async def maintain(
        self, uses: Mapping[Key, int] | None = None, dry_run: bool = False
    ) -> Outcome[V]:
        """Reviews the stored cases and restores the case base, outside the CBR cycle and without a query.

        The changes merge or remove duplicates, remove inconsistent cases,
        and remove cases that the competence of the case base does not need, each change with its reason,
        and the review assesses every stored case if a competence threshold is configured.
        When to maintain, such as periodically, is up to the caller.
        If another run changes the case base in the meantime, the run is planned again.

        Args:
            uses: How often every case was used, which decides between cases of equal competence.
            dry_run: Plan without changing the case base, the configuration may enforce it.

        Without a maintain section, nothing changes.
        """
        if self._bound.maintain is None:
            return await self._outcome(changes=Changes())

        if uses is not None:
            # JSON object keys are strings, so they are matched with the stored keys as text
            keys = {str(key): key for key in await self.storage.get_index()}
            uses = {keys.get(str(key), key): count for key, count in uses.items()}

        changes = await self._changes(
            self._bound.phase(self._bound.maintain),
            dry_run or self.config.limits.dry_run,
            uses,
        )

        return await self._outcome(changes=changes)

    async def cycle(
        self,
        query: V | None = None,
        key: Key | None = None,
        limit: int | None = None,
        explain: bool = False,
        dry_run: bool = False,
    ) -> Outcome[V]:
        """Runs the cycle for a query or a stored case: retrieves cases, then reuses, revises, and retains them as configured.

        The hits are the cases of the last phase before retain, ranked and scored by it.
        Only a reuse section turns retrieved cases into solutions of the query,
        so the solutions it proposes, one by default, and revise accepts are retained
        if the specification has both a reuse and a retain section (`Config.cycle_retains`),
        leaving out those that reuse passed on unchanged, such as by a null adaptation (`cbrkit.retain.changed`),
        and the changes are those of that retain run.

        Args:
            query: The query, attributes that are left out do not count.
            key: Key of a stored case to use as the query instead, which is left out of the retrieved cases.
            limit: Maximal number of retrieved cases, the configured limit if None.
            explain: Break the similarity of every case down by attribute or graph element.
            dry_run: Plan the changes of retain without changing the case base, the configuration may enforce it.

        Raises:
            cbrkit.errors.Refused: If neither or both of query and key are given, the key is unknown,
                or graphs compare more node pairs than the limits allow.
        """
        casebase, query = await self._resolve(query, key)
        steps = self._steps(limit, self._bound.reuse, self._bound.revise)
        result = await run_query_async(casebase, query, *steps)
        changes = None

        if self._retainer is not None and self.config.cycle_retains:
            changes = await self._store(
                self._retainer, changed(result.casebase, casebase), query, dry_run, casebase
            )

        return await self._outcome(result, changes, _measure(steps[-1], explain))


def _measure[V](phase: Phase, explain: bool) -> Measure[V, Any] | None:
    """The measure that scored the cases of a phase, if they are to be explained."""
    return phase.measure if explain else None
