"""The sidecar configuration of a case base and its binding to the functions of the host.

The measures and the phases of the cycle are the classes of `cbrkit.sim`, `cbrkit.adapt`, and `cbrkit.retrieve` to `cbrkit.maintain`,
serialized by their fields and the `kind` of their family,
so this module only adds what the sidecar needs beyond them.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import orjson
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from ..io import Source, atomic_write, companion
from ..maintain import Maintainer
from ..retain import Retainer
from ..retrieve import Retriever
from ..reuse import Reuser
from ..revise import Reviser
from ..schema import Model
from ..sim import Measure
from ..sim.embed import Embedder, EmbeddingCache, SQLiteStore
from ..sim.llm import LanguageModel
from ..typing import (
    AnyCbrFunc,
    EmbeddingFunc,
    FilePath,
    Float,
    GenerationFunc,
    NumpyArray,
    Storage,
)

__all__ = [
    "Config",
    "Embedders",
    "Limits",
    "Models",
    "Phase",
    "Provenance",
    "Providers",
    "bind",
    "dumps",
    "providers",
    "read",
    "write",
]


class Provenance(Model):
    """Who decided on the measure at a JSON pointer of the similarity, and why.

    Args:
        source: Who took the decision.
        model: Identifier of the language model for decisions of an `llm`.
        reason: Short justification.
        element: JSON pointer of the compared element in the profile of the case base,
            such as `/attributes/price`.
        absent: Set when the element no longer occurs in the case base.
    """

    source: Literal["heuristic", "llm", "user"] = "heuristic"
    model: str | None = None
    reason: str = ""
    element: str = ""
    absent: bool = False


class Limits(Model):
    """Bounds of what a request to a `cbrkit.system.System` may cost and change.

    Args:
        max_limit: Largest number of cases or solutions a request may ask for.
        max_node_pairs: Largest number of pairs of query and case nodes that a retrieval of a graph compares,
            summed over the stored cases, unbounded if None.
        dry_run: Plan the changes of retain and maintenance runs without changing the case base,
            which no request can override.
    """

    max_limit: Annotated[int, Field(ge=1)] = 100
    max_node_pairs: Annotated[int, Field(ge=1)] | None = None
    dry_run: bool = False


type ModelName = Annotated[str, Field(pattern=r"^[\w.-]+(/[\w.-]+)?:.+$")]
"""A Pydantic AI model identifier of a provider and a model, such as `openai:text-embedding-3-small`."""


class Providers(Model):
    """Models of the host by the names the measures refer to, without any secrets.

    Keys and endpoints come from the environment variables of the providers.

    Args:
        embedders: Pydantic AI embedding model of every embedder name of the measures,
            such as `openai:text-embedding-3-small` or `sentence-transformers:all-MiniLM-L6-v2`.
        models: Pydantic AI model of every language model name of the measures and adaptations,
            such as `anthropic:claude-sonnet-4-6`.
        cache: Keep the embeddings of the cases in the cache companion of the case base (`<name>.cbrkit.sqlite`),
            and those of queries only in memory, see `cbrkit.sim.embed.EmbeddingCache`.
    """

    embedders: dict[str, ModelName] = Field(default_factory=dict)
    models: dict[str, ModelName] = Field(default_factory=dict)
    cache: bool = False


type Phase = (
    Retriever[Any, Any]
    | Reuser[Any, Any]
    | Reviser[Any, Any]
    | Retainer[Any, Any]
    | Maintainer[Any, Any]
)
"""A phase of the cycle or the maintenance, the sections of a configuration, see `Config.phase`."""


class Config(Model):
    """Root of a sidecar file with every decision needed for the CBR cycle of a case base.

    The phases are the classes of the cycle, and a phase without a measure uses `measure`, see `Config.phase`,
    except revise, whose quality of solutions is no similarity of problems and needs its own.
    Its JSON Schema is `Config.model_json_schema()`,
    and `cbrkit.schema.extend(Config, ...)` gives the model of sidecars with custom kinds.

    Args:
        version: Version of the sidecar format, which cbrkit reads it by.
        schema_ref: URI of the JSON Schema that editors validate the sidecar with,
            stored as `$schema` and kept as given.
        source: How the case base is stored.
        fingerprint: Hash of the shape of the case base the decisions were last merged with.
        measure: The similarity measure of the cases, any measure of `cbrkit.sim` by its kind.
        provenance: Provenance of the measures in `measure` by their JSON pointer,
            `""` for the whole measure and for example `/attributes/price` for an attribute.
        retrieve: How cases are retrieved, see `cbrkit.retrieve.Retriever`.
        reuse: How retrieved cases are adapted, see `cbrkit.reuse.Reuser`, no reuse if None.
        revise: How solutions are repaired and assessed, see `cbrkit.revise.Reviser`,
            every solution is accepted if None.
        retain: How accepted solutions are stored, see `cbrkit.retain.Retainer`, never if None.
        maintain: How the case base is maintained outside the cycle, see `cbrkit.maintain.Maintainer`, never if None.
        limits: Bounds of what a request may cost and change.
        providers: Models of the host by the names the measures refer to.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    version: Literal[2] = 2
    schema_ref: str | None = Field(
        default=None, alias="$schema", exclude_if=lambda value: value is None
    )
    source: Source
    fingerprint: str
    measure: Measure[Any, Any]
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    # unparametrized, so that JSON Schemas name the sections by their classes
    retrieve: Retriever = Retriever(limit=10)  # pyright: ignore[reportMissingTypeArgument]
    reuse: Reuser | None = None  # pyright: ignore[reportMissingTypeArgument]
    revise: Reviser | None = None  # pyright: ignore[reportMissingTypeArgument]
    retain: Retainer | None = None  # pyright: ignore[reportMissingTypeArgument]
    maintain: Maintainer | None = None  # pyright: ignore[reportMissingTypeArgument]
    limits: Limits = Limits()
    providers: Providers = Providers()

    @property
    def cycle_retains(self) -> bool:
        """Whether the cycle retains, which needs a reuse section, since only reuse turns retrieved cases into solutions of the query."""
        return self.reuse is not None and self.retain is not None

    @property
    def writes(self) -> bool:
        """Whether requests may change the case base, through a retain or maintain section without an enforced dry run."""
        return (
            self.retain is not None or self.maintain is not None
        ) and not self.limits.dry_run

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (
            self.retain is not None or self.maintain is not None
        ) and not self.source.writable:
            raise ValueError(f"Cases cannot be written to {self.source.format} files")

        if (self.retrieve.limit or 0) > self.limits.max_limit:
            raise ValueError("The limit exceeds the maximal limit")

        if self.revise is not None and self.revise.measure is None:
            raise ValueError(
                "The revise section needs a measure of the quality of solutions,"
                " since the measure of the configuration compares problems"
            )

        return self

    def steps(
        self, storage: Storage[Any, Any] | None = None
    ) -> tuple[AnyCbrFunc[Any, Any, Any], ...]:
        """The phases of the cycle that the configuration has, in the order in which they run, with their measures filled in.

        Retrieve is always configured, reuse and revise only with their sections,
        and retain with its section and a reuse section (`cycle_retains`) only with the storage that it writes,
        storing the solutions that reuse proposes and revise accepts, except those it passed on unchanged.
        Bind the providers of the host with `bind` first.

        Args:
            storage: The storage of the cases, which a retain section writes.

        Examples:
            >>> from cbrkit.cycle import run_query
            >>> from cbrkit.spec import infer
            >>> cases = {"a": {"price": 10.0}, "b": {"price": 20.0}}
            >>> config = infer(cases, Source(format="json"))
            >>> run_query(cases, {"price": 12.0}, *config.steps()).ranking
            ('a',)
        """
        steps: tuple[AnyCbrFunc[Any, Any, Any], ...] = tuple(
            self.phase(p)
            for p in (self.retrieve, self.reuse, self.revise)
            if p is not None
        )
        retainer = (
            self.retainer(storage)
            if storage is not None and self.cycle_retains
            else None
        )

        return steps if retainer is None else (*steps, retainer)

    def retainer[K, V](self, storage: Storage[K, V]) -> Retainer[K, V] | None:
        """The retain phase of the configuration that writes a storage, with its measure filled in, None without a retain section.

        Args:
            storage: The storage of the cases that it writes.
        """
        if self.retain is None:
            return None

        return replace(self.phase(self.retain), storage=storage)

    def phase[P: Phase](self, section: P) -> P:
        """A phase of the configuration with its measure filled in where the phase has none.

        Args:
            section: One of its phases, such as `config.retain`.

        Examples:
            >>> from cbrkit.spec import infer
            >>> config = infer({0: {"price": 10.0}}, Source(format="json"))
            >>> config.phase(config.retrieve).measure == config.measure
            True
        """
        if section.measure is not None:
            return section

        return replace(section, measure=self.measure)


def dumps(config: Config) -> str:
    """Serializes a configuration deterministically with sorted keys."""
    return (
        orjson.dumps(
            config.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        ).decode()
        + "\n"
    )


def read(path: FilePath, model: type[Config] = Config) -> Config:
    """Reads a sidecar file.

    Args:
        path: Path of the sidecar.
        model: The configuration model, such as one extended by custom kinds with `cbrkit.schema.extend`.
    """
    return model.model_validate_json(Path(path).read_bytes())


def write(path: FilePath, config: Config) -> None:
    """Writes a sidecar file atomically."""
    atomic_write(Path(path), dumps(config))


type Embedders = Mapping[str, EmbeddingFunc[str, Any]]
"""Embedding functions of the host by the names that `cbrkit.sim.embed.Embedder` refers to."""

type Models = Mapping[str, GenerationFunc]
"""Language models of the host by the names that `cbrkit.sim.llm.LanguageModel` refers to."""


class _Measure(RootModel[Measure[Any, Any]]):
    """Any similarity measure, serialized by its kind if it is one of `cbrkit.sim.MEASURES`."""


def _bind[M: BaseModel](model: M, embedders: Embedders, models: Models) -> M:
    """Validates a model again with the embedders and language models of the host, which every name it refers to needs.

    Raises:
        cbrkit.errors.Unbound: If an embedder or language model is bound to no function.
    """
    context = {Embedder: embedders, LanguageModel: models}

    return type(model).model_validate(model.model_dump(), context=context)


def providers(
    config: Config,
    embedders: Embedders | None = None,
    models: Models | None = None,
    *,
    casebase: FilePath | None = None,
) -> tuple[Embedders, Models]:
    """The functions of the host completed by the providers section for the names that the host passes no function for.

    Embedders are bound to `cbrkit.sim.embed.PydanticAI`, cached next to the case base if `cache` is set,
    and language models to `cbrkit.sim.llm.PydanticAI`, which needs the `pydantic-ai` extra
    and takes the keys from the environment variables of the providers,
    so only a host that trusts the sidecar uses it.
    Pass the result to `bind` or `cbrkit.system.System`.

    Args:
        config: The configuration.
        embedders: Embedding functions of the host by name, which take precedence over the providers section.
        models: Language models of the host by name, which take precedence over the providers section.
        casebase: Path of the case base, next to which the providers cache the embeddings.

    Raises:
        AttributeError: If a name needs Pydantic AI but the `pydantic-ai` extra is missing.
    """
    bound_embedders: dict[str, EmbeddingFunc[str, Any]] = dict(embedders or {})
    bound_models: dict[str, GenerationFunc] = dict(models or {})

    for name, model in config.providers.embedders.items():
        if name in bound_embedders:
            continue

        from pydantic_ai import embeddings

        from ..sim.embed import PydanticAI as PydanticAIEmbedder

        func: EmbeddingFunc[str, NumpyArray] = PydanticAIEmbedder(
            embeddings.Embedder(model), input_type="document"
        )

        if config.providers.cache and casebase is not None:
            func = EmbeddingCache(
                embedder=func, store=SQLiteStore(companion(casebase, "cache"), model)
            )

        bound_embedders[name] = func

    for name, model in config.providers.models.items():
        if name not in bound_models:
            from ..sim.llm import PydanticAI

            bound_models[name] = PydanticAI(model)

    return bound_embedders, bound_models


def bind(
    config: Config, embedders: Embedders | None = None, models: Models | None = None
) -> Config:
    """Returns a configuration whose embedders and language models are bound to functions.

    Every name that the configuration refers to must be bound by the arguments,
    also one bound before, so a missing provider fails here instead of in the middle of a request.
    Complete the functions of the host by the providers section with `providers` first.

    Args:
        config: The configuration.
        embedders: Embedding functions referenced by `cbrkit.sim.embed.Embedder` names.
            Wrap them in `cbrkit.sim.embed.EmbeddingCache` to embed every text only once.
        models: Language models referenced by `cbrkit.sim.llm.LanguageModel` names.

    Raises:
        cbrkit.errors.Unbound: If an embedder or language model of the configuration is bound to no function.

    Examples:
        >>> from cbrkit.sim import AttributeValue
        >>> from cbrkit.sim.embed import Cosine, Embedder, Embedding
        >>> measure = AttributeValue(attributes={"text": Embedding(embedder=Embedder("e"), metric=Cosine())})
        >>> config = Config(source=Source(format="json"), fingerprint="", measure=measure)
        >>> bind(config, {"e": print}).measure.attributes["text"].embedder.func is print
        True
    """
    return _bind(config, embedders or {}, models or {})


def _build(measure: Measure[Any, Any], embedders: Embedders) -> Measure[Any, Float]:
    """A measure with the embedders of the host bound, see `bind`."""
    return _bind(_Measure(measure), embedders, {}).root
