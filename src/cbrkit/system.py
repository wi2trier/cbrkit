from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel

from .cycle import apply_query as _cycle_apply_query
from .helpers import produce_factory
from .model.result import QueryResultStep
from .retain import apply_query as _retain_apply_query
from .retrieval import apply_query as _retrieval_apply_query
from .reuse import apply_query as _reuse_apply_query
from .revise import apply_query as _revise_apply_query
from .typing import (
    Float,
    MaybeFactories,
    MaybeFactory,
    RetainerFunc,
    RetrieverFunc,
    ReuserFunc,
    ReviserFunc,
)

__all__ = [
    "System",
    "CasebaseSpec",
]

type CasebaseSpec[K, V] = Iterable[K] | Mapping[K, V] | None


@dataclass(slots=True, frozen=True)
class System[
    K,
    V: BaseModel,
    S: Float = float,
    R1: BaseModel | None = None,
    R2: BaseModel | None = None,
    R3: BaseModel | None = None,
    R4: BaseModel | None = None,
]:
    casebase: MaybeFactory[Mapping[K, V]] = field(default_factory=dict)
    retriever_factory: (
        Callable[
            [R1 | None],
            MaybeFactories[RetrieverFunc[K, V, S]],
        ]
        | None
    ) = None
    reuser_factory: (
        Callable[
            [R2 | None],
            MaybeFactories[ReuserFunc[K, V, S]],
        ]
        | None
    ) = None
    reviser_factory: (
        Callable[
            [R3 | None],
            MaybeFactories[ReviserFunc[K, V, S]],
        ]
        | None
    ) = None
    retainer_factory: (
        Callable[
            [R4 | None],
            MaybeFactories[RetainerFunc[K, V, S]],
        ]
        | None
    ) = None

    def _load_casebase(self, spec: CasebaseSpec[K, V]) -> Mapping[K, V]:
        casebase = produce_factory(self.casebase)

        if spec is None:
            return casebase
        elif isinstance(spec, Mapping):
            return cast(Mapping[K, V], spec)
        elif isinstance(spec, Iterable):
            return {key: casebase[key] for key in spec}

        raise ValueError("Invalid casebase specification.")

    def retrieve(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        config: R1 | None = None,
    ) -> QueryResultStep[K, V, S]:
        if not self.retriever_factory:
            raise ValueError("Retriever factory is not defined.")

        return _retrieval_apply_query(
            self._load_casebase(casebase),
            query,
            self.retriever_factory(config),
        ).default_query

    def reuse(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        config: R2 | None = None,
    ) -> QueryResultStep[K, V, S]:
        if not self.reuser_factory:
            raise ValueError("Reuser factory is not defined.")

        return _reuse_apply_query(
            self._load_casebase(casebase),
            query,
            self.reuser_factory(config),
        ).default_query

    def revise(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        config: R3 | None = None,
    ) -> QueryResultStep[K, V, S]:
        """Revise solutions by assessing quality and optionally repairing them.

        Args:
            query: The query to revise solutions for.
            casebase: Optional casebase specification.
            config: Optional reviser configuration.

        Returns:
            The revised query result step.
        """
        if not self.reviser_factory:
            raise ValueError("Reviser factory is not defined.")

        return _revise_apply_query(
            self._load_casebase(casebase),
            query,
            self.reviser_factory(config),
        ).default_query

    def retain(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        config: R4 | None = None,
    ) -> QueryResultStep[K, V, S]:
        """Retain cases in the casebase.

        Args:
            query: The query whose case may be retained.
            casebase: Optional casebase specification.
            config: Optional retainer configuration.

        Returns:
            The retained query result step.
        """
        if not self.retainer_factory:
            raise ValueError("Retainer factory is not defined.")

        return _retain_apply_query(
            self._load_casebase(casebase),
            query,
            self.retainer_factory(config),
        ).default_query

    def cycle(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        retriever_config: R1 | None = None,
        reuser_config: R2 | None = None,
        reviser_config: R3 | None = None,
        retainer_config: R4 | None = None,
    ) -> QueryResultStep[K, V, S]:
        if not self.retriever_factory:
            raise ValueError("Retriever factory is not defined.")

        reusers = self.reuser_factory(reuser_config) if self.reuser_factory else []
        revisers = self.reviser_factory(reviser_config) if self.reviser_factory else []
        retainers = (
            self.retainer_factory(retainer_config) if self.retainer_factory else []
        )

        return _cycle_apply_query(
            self._load_casebase(casebase),
            query,
            self.retriever_factory(retriever_config),
            reusers,
            revisers,
            retainers,
        ).retain.final_step.default_query
