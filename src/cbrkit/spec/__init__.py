"""Declarative, serializable similarity specifications.

A specification stores every decision needed for retrieval, so that retrieving is
deterministic and nothing is inferred at query time.
Its `measure` is any measure of `cbrkit.sim` in its serialized form,
so every measure of `cbrkit.sim.MEASURES` and those added by `cbrkit.schema.extend(Config, ...)` can be used,
and the JSON Schema of the sidecar is derived from the measure classes and their docstrings.
Each decision records its provenance by the JSON pointer of its measure:
a `heuristic`, an `llm` with its model, or a `user`, together with a short reason.
Hosts keep the configuration as a JSON sidecar next to the case base
(`<name>.cbrkit.json`), one of the `cbrkit.io.Companion` files of a case base besides its embedding cache and its lock.
Sidecars are UTF-8 JSON with sorted keys, an indent of two spaces, LF line endings, and a trailing newline.
Case bases are read and written by `cbrkit.io`, which also describes how one is stored (`cbrkit.io.Source`).

Functions:
- `infer` gives the configuration of a case base: heuristic decisions for its elements,
  added to those of a saved configuration without rewriting them.
- `bind` binds the embedders and language models of the host to a configuration,
  whose `Config.steps` are the phases of the cycle and whose `Config.phase` fills the measure into a phase,
  and `providers` completes the functions of the host by the providers section.
- `read`, `write`, and `dumps` read and write sidecars, `Config.model_validate_json` parses one,
  and `Config.model_json_schema` exports its JSON Schema.
- `refine` lets a bound `cbrkit.sim.llm.LanguageModel` refine decisions,
  choosing from the deterministic leaf measures of `leaves` with validation and fallback,
  and `mark_user` records the decisions that a user changed in an edited configuration.
- `case_model` derives the Pydantic model of the cases from the measures.

Modules:
- `_config`: the sidecar model `Config` with its sections, its file I/O, `providers`, and `bind`.
- `_infer`: `infer`, and `_profile` the statistics it decides by.
- `_refine`: `refine`, `leaves`, `Rejection`, and `mark_user`.
- `_cases`: `case_model`.
- `pointer`: the JSON pointers of decisions into measures, profiles, and their JSON, public for hosts that render a configuration.

The sections `retrieve`, `reuse`, `revise`, `retain`, and `maintain` are the phase classes
`cbrkit.retrieve.Retriever`, `cbrkit.reuse.Reuser`, `cbrkit.revise.Reviser`, `cbrkit.retain.Retainer`,
and `cbrkit.maintain.Maintainer` themselves, which use `measure` unless they name another one,
except revise, which needs its own measure of the quality of solutions.
They configure the CBR cycle of `cbrkit.system.System` and the maintenance of its case base outside the cycle,
`limits` bounds what a request may cost and change,
and `providers` maps the embedder and language model names to Pydantic AI models without secrets.
`cbrkit.sim.explain` breaks a similarity down for user interfaces.

Decisions are keyed by the JSON pointer of their measure, `""` for the whole measure
and for example `/attributes/price` for an attribute, whose weight lives in the `Aggregator` of its `AttributeValue`.
An `Embedding` measure only names its `cbrkit.sim.embed.Embedder`, which the host binds,
so the specification holds no provider settings or secrets.
Serializing a measure that holds a custom function, such as a `value_getter` or a measure of no family,
raises instead of silently dropping it, so a sidecar always reproduces the retrieval.

Heuristics of `infer`:
identifiers and links get a weight of zero, booleans are compared by equality,
numbers and dates linearly over their 5th to 95th percentile, categorical strings by case-insensitive equality,
short strings by Levenshtein, long texts by embeddings (or n-grams without an embedder),
lists by Jaccard, and objects recursively.
Graphs are compared with `cbrkit.sim.graphs.LAP`, nodes and edges per type with an `cbrkit.sim.AttributeTable`,
and the types of BPMN elements with a taxonomy of the BPMN element hierarchy.
Graphs count as BPMN graphs by the element types of their nodes, also without the `bpmn` extra.
A saved configuration keeps the decisions for elements that disappeared, marked as `absent`.

Refinement:
the model only picks deterministic leaf measures and weights for decisions not taken by a user.
Each proposed measure is built and run on sample values,
and a proposal that fails, leaves `[0, 1]`, or does not rate equal values as fully similar is rejected,
which keeps the previous decision.

Sections of the sidecar besides `source` and `measure`, each optional:
- `retrieve`: the number of cases of a request and the minimal similarity.
- `reuse`: an `adaptation` of `cbrkit.adapt` by its kind, such as `attribute_value` or the `rewrite`
  of a language model, scored by the measure of the configuration unless it names another one.
- `revise`: a required `measure` of the quality of solutions, such as the `critique` of a language model,
  the `min_similarity` of an accepted solution, and an optional `repair` of the failing ones.
- `retain`: stores accepted solutions, leaving out or merging those that `duplicate` a stored case.
- `maintain`: reviews and restores the case base outside the cycle.
- `limits`: the largest number of cases of a request, a budget of compared graph nodes,
  and a `dry_run` that only plans the changes of retain and maintenance runs.
- `providers`: Pydantic AI model names of the `embedders` and language `models`,
  such as `openai:text-embedding-3-small`, with keys from the environment variables of the providers.

Besides the measure, `infer` fills the sections that need no decision about the domain:
retrieve, reuse by a null adaptation that proposes the most similar case as it is, and retain for writable sources,
while revise and maintain stay off.
A section that is None turns its phase off, and without `retain` and `maintain` the case base is never written.

Example:
    >>> import cbrkit
    >>> from cbrkit.io import Source
    >>> cases = {
    ...     "a": {"name": "Golf", "price": 20000, "fuel": "petrol"},
    ...     "b": {"name": "Polo", "price": 15000, "fuel": "diesel"},
    ...     "c": {"name": "Passat", "price": 35000, "fuel": "petrol"},
    ... }
    >>> config = infer(cases, Source(format="json"))
    >>> config.measure.attributes["price"]
    Linear(max_distance=18000.0, min_distance=0.0)
    >>> query = {"price": 21000, "fuel": "Petrol"}
    >>> result = cbrkit.cycle.run_query(cases, query, config.phase(config.retrieve))
    >>> result.ranking[:2]
    ('a', 'c')
    >>> explanation = cbrkit.sim.explain(cases["a"], query, result.similarities["a"], config.measure)
    >>> list(explanation.attributes)
    ['price', 'fuel']
"""

from ._cases import case_model
from ._config import (
    Config,
    Embedders,
    Limits,
    Models,
    Phase,
    Provenance,
    Providers,
    bind,
    dumps,
    providers,
    read,
    write,
)
from ._infer import infer
from ._refine import Rejection, leaves, mark_user, refine

__all__ = [
    "Config",
    "Embedders",
    "Limits",
    "Models",
    "Phase",
    "Provenance",
    "Providers",
    "Rejection",
    "bind",
    "case_model",
    "dumps",
    "infer",
    "leaves",
    "mark_user",
    "providers",
    "read",
    "refine",
    "write",
]
