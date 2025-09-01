from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Mapping

from pydantic import BaseModel

import cbrkit
from cbrkit.typing import Float, MaybeFactories

__all__ = [
    "System",
    "CasebaseSpec",
]

type CasebaseSpec[K, V] = Iterable[K] | Mapping[K, V] | None


@dataclass(slots=True, frozen=True)
class System[
    K,
    V: BaseModel,
    S: Float,
    R1: BaseModel | None,
    R2: BaseModel | None,
]:
    casebase: Mapping[K, V]
    retriever_factory: (
        Callable[
            [R1],
            MaybeFactories[cbrkit.typing.RetrieverFunc[K, V, S]],
        ]
        | None
    ) = None
    reuser_factory: (
        Callable[
            [R2],
            MaybeFactories[cbrkit.typing.ReuserFunc[K, V, S]],
        ]
        | None
    ) = None

    def _load_casebase(self, spec: CasebaseSpec[K, V]) -> Mapping[K, V]:
        if spec is None:
            return self.casebase
        elif isinstance(spec, Mapping):
            return spec
        elif isinstance(spec, Iterable):
            return {key: self.casebase[key] for key in spec}

        raise ValueError("Invalid casebase specification.")

    def retrieve(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        config: R1 = None,
    ) -> cbrkit.model.QueryResultStep[K, V, S]:
        if not self.retriever_factory:
            raise ValueError("Retriever factory is not defined.")

        return cbrkit.retrieval.apply_query(
            self._load_casebase(casebase),
            query,
            self.retriever_factory(config),
        ).default_query

    def reuse(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        config: R2 = None,
    ) -> cbrkit.model.QueryResultStep[K, V, S]:
        if not self.reuser_factory:
            raise ValueError("Reuser factory is not defined.")

        return cbrkit.reuse.apply_query(
            self._load_casebase(casebase),
            query,
            self.reuser_factory(config),
        ).default_query

    def cycle(
        self,
        query: V,
        *,
        casebase: CasebaseSpec[K, V] = None,
        retriever_config: R1 = None,
        reuser_config: R2 = None,
    ) -> cbrkit.model.QueryResultStep[K, V, S]:
        if not self.retriever_factory or not self.reuser_factory:
            raise ValueError("Retriever or reuser factory is not defined.")

        return cbrkit.cycle.apply_query(
            self._load_casebase(casebase),
            query,
            self.retriever_factory(retriever_config),
            self.reuser_factory(reuser_config),
        ).final_step.default_query
