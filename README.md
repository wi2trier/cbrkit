<!-- markdownlint-disable MD033 MD041 -->
<h1><p align="center">CBRkit</p></h1>

<p align="center">
  <img width="256px" alt="cbrkit logo" src="https://raw.githubusercontent.com/wi2trier/cbrkit/main/assets/logo.png" />
</p>

<p align="center">
  <a href="https://pypi.org/project/cbrkit/">PyPI</a> |
  <a href="https://wi2trier.github.io/cbrkit/">Docs</a> |
  <a href="https://wi2trier.github.io/cbrkit/llms.txt">llms.txt</a> |
  <a href="https://github.com/wi2trier/cbrkit/tree/main/examples">Examples</a>
</p>

<p align="center">
  Customizable Case-Based Reasoning (CBR) toolkit for Python with a CLI, REST API and MCP server.
</p>

<p align="center">
  <a href="https://youtu.be/27dG4MagDhE">CBRkit Presentation</a>
  <br/>
  <i>ICCBR 2024 Best Student Paper</i>
</p>

---

<!-- PDOC_START -->

CBRkit is a modular toolkit for Case-Based Reasoning in Python.
It loads case bases, compares cases with built-in or custom similarity measures, and runs the whole CBR cycle: retrieve, reuse, revise, retain, plus case base maintenance.
A case base with a declarative sidecar specification is served over a command line, a REST API, and an MCP server without writing any code.
Everything is typed, and the descriptions of the JSON Schemas, the OpenAPI document, and the MCP tools come from the docstrings of the code, so agents see the same documentation as you.
CBRkit covers case-based reasoning only, agents and retrieval-augmented generation (RAG) over case bases build on it in [exar](https://github.com/wi2trier/exar).

This page gives an overview, every module documents its details with tested examples.
The [documentation](https://wi2trier.github.io/cbrkit/) is also available as [llms.txt](https://wi2trier.github.io/cbrkit/llms.txt) for language models, and the [demo project](https://github.com/wi2trier/cbrkit-demo) shows a complete application.

## Modules

- `cbrkit.io`: Reads and writes case bases as csv, json, jsonl, parquet and yaml files, reads toml, xlsx and xml files, folders of JSON, BPMN (with the `bpmn` extra) or text files, a single text file, data frames, and SQL queries, and serves files as storages, refusing malformed or oversized ones with `cbrkit.errors.Refused`.
- `cbrkit.sim`: Similarity measures for numbers, dates, strings, collections, embeddings, graphs, and taxonomies, aggregated by `AttributeValue` and `Aggregator`.
- `cbrkit.adapt`: Adaptation functions that modify a case to suit a query, including a rewrite by a language model.
- `cbrkit.retrieve`: Retrievers by similarity, BM25 or embeddings, grouped retrieval, rank fusion, and rerankers.
- `cbrkit.reuse`, `cbrkit.revise`: Adapting and scoring retrieved cases, assessing and repairing solutions.
- `cbrkit.retain`, `cbrkit.maintain`: Storing accepted solutions and maintaining the whole case base, planned as typed `Changes` with reasons.
- `cbrkit.cycle`: Runs any sequence of retrievers, reusers, revisers, and retainers on one async engine.
- `cbrkit.indexable`: Storages for persistent indexes and retained cases (in memory, LanceDB, pgvector, SQLite, zvec), the SQL ones with a required `scope` of the rows they reach, which `narrow` restricts further and `Unscoped.TABLE` lifts for an indexer.
- `cbrkit.spec`: Declarative, serializable specifications of the measure and the phases of a case base, inferred by heuristics and refined by a language model.
- `cbrkit.system`: A CBR system built from a specification, its operations as typed functions for agents, served over REST and MCP.
- `cbrkit.eval`: Retrieval metrics such as precision, recall, and nDCG as typed objects.
- `cbrkit.model`, `cbrkit.typing`, `cbrkit.errors`, `cbrkit.helpers`, `cbrkit.schema`: Result models, protocols for custom functions, exceptions, helpers, and the serialized forms of all functions as immutable `Family` sets of dataclasses.
A `Family` is an immutable set of serializable dataclasses that share a `kind` tag, which validates and serializes its members by that tag and builds the JSON Schema offering all of them.

Classes use CapWords names with all-caps acronyms (`Levenshtein`, `HTTP`, `BM25`), and functions use snake_case.
Functions with effects, such as storages, embedders, language models and rerankers, are asynchronous, while measures and adaptations are synchronous and pure.
Every name has one public path, the export of its package (`cbrkit.retrieve.rerank.HTTP`).

## Installation

```shell
pip install cbrkit
pip install 'cbrkit[EXTRA,...]'
```

The core needs no extra, the optional dependencies are grouped as follows (`all` installs every one):

- **Entry points:** `cli` (command line), `api` (REST API), `mcp` (MCP server)
- **Language and embedding models:** `pydantic-ai` (every provider of Pydantic AI), `http` (rerank endpoints)
- **Text and embeddings:** `bm25`, `transformers` (Sentence Transformers with PyTorch)
- **Graphs:** `bpmn` (BPMN 2.0 documents), `graphs` (NetworkX and rustworkx), `graphviz`
- **Data:** `fastexcel` (xlsx), `sql` (SQLAlchemy), `lancedb`, `pgvector`, `sqlite-vec`, `zvec`

Names that need an extra, such as `cbrkit.sim.embed.SentenceTransformers`, are imported on first access, so `import cbrkit` stays fast, and a missing extra raises an `AttributeError` that names it.
Setting `CBRKIT_EAGER_IMPORTS=1` imports every installed backend right away, which tools like pdoc need.
Frozen applications built with PyInstaller need `--collect-submodules cbrkit` and `module_collection_mode={"cbrkit": "pyz+py"}`, because the lazy names are read from the sources.

## Quickstart: Library

A similarity measure compares a case and a query, `AttributeValue` applies one per attribute and aggregates them, and a retriever ranks the cases of a case base:

```pycon
>>> import cbrkit
>>> casebase = cbrkit.io.load("data/cars-1k.csv")
>>> measure = cbrkit.sim.AttributeValue(
...     attributes={
...         "price": cbrkit.sim.numbers.Linear(max_distance=100000),
...         "year": cbrkit.sim.numbers.Linear(max_distance=50),
...         "manufacturer": cbrkit.sim.strings.Levenshtein(),
...     },
...     aggregator=cbrkit.sim.Aggregator(weights={"price": 2}),
... )
>>> retriever = cbrkit.retrieve.Retriever(measure=measure, limit=3)
>>> query = {"price": 10000, "year": 2010, "manufacturer": "audi"}
>>> result = cbrkit.cycle.run_query(casebase, query, retriever)
>>> result.ranking
(990, 526, 496)
>>> result.similarities[990]
AttributeValueSim(value=0.99959, similarities=frozendict({'price': 0.99918, 'year': 1.0, 'manufacturer': 1.0}))

```

Custom measures are plain functions `f(case, query)`, and batch functions score a list of pairs at once (see `cbrkit.helpers`).
The numeric, date, string and generic measures score whole columns with numpy and rapidfuzz, and an executor spreads large case bases over processes (see `cbrkit.retrieve`).
The other phases follow the same protocol, so `cbrkit.cycle.run_query(storage, query, retriever, reuser, reviser, retainer)` runs the whole cycle in order (see `cbrkit.cycle`).

`run_query` is `asyncio.run` around `run_query_async`, the engine that applications with an event loop await.
It awaits the functions with effects and runs the pure ones in a worker thread, from which measures that embed or generate await their models (see `cbrkit.helpers.wait`).
Retrieved cases solve other problems, so a reuser adapts them to the query and proposes the best solution, or as many as its `limit` keeps.
A retainer stores the solutions that reach it in its storage as one mutation, planned again if the storage changed in between, and the result holds the changes with their reasons:

```pycon
>>> storage = cbrkit.indexable.InMemory(casebase)
>>> reuser = cbrkit.reuse.Reuser(adaptation=lambda case, query: {**case, **query}, measure=measure)
>>> retainer = cbrkit.retain.Retainer(storage=storage, measure=measure)
>>> result = cbrkit.cycle.run_query(storage, query, retriever, reuser, retainer)
>>> [(change.op, change.key) for change in result.changes.items]
[('add', 999)]

```

Maintenance reviews the whole case base outside the cycle and plans its changes purely, which `cbrkit.model.replan(maintainer, storage)` applies (see `cbrkit.maintain`).

Cases and queries are always of the same type, and any value is its own problem and solution.
`cbrkit.model.Case` splits a case into a problem and a solution, and a query into a problem without a solution:
retrievers and retainers compare the problems, reusers reuse the adapted solution for the problem of the query, and revisers assess the whole case.

## Quickstart: Specification and Command Line

`cbrkit.spec` infers a serializable configuration from the case base, which applications keep as a sidecar file next to it and edit by hand or with a language model:

```pycon
>>> from cbrkit import spec
>>> source = cbrkit.io.detect("data/cars-1k.csv")
>>> config = spec.infer(cbrkit.io.load("data/cars-1k.csv", source), source)
>>> config.measure.attributes["price"]
Linear(max_distance=12808.8, min_distance=0.0)
>>> cbrkit.cycle.run_query(casebase, query, config.phase(config.retrieve)).ranking[:3]
(990, 526, 496)
>>> cbrkit.cycle.run_query(casebase, query, *config.steps()).ranking
(990,)

```

The sidecar `<name>.cbrkit.json` is UTF-8 JSON with sorted keys that editors validate through an optional `$schema` key.
It is one of the companions beside a case base, together with the embedding cache `<name>.cbrkit.sqlite` and the lock `<name>.cbrkit.lock`, which hosts derive with `cbrkit.io.companion` and recognize with `cbrkit.io.subject` instead of spelling the names themselves.
`cbrkit.io.is_casebase` tells which files and folders may be case bases.
Hosts that let users edit the sidecar offer the measures of `spec.leaves()` and record the changed decisions with `spec.mark_user`, so a later refinement by a language model keeps them.
Given the saved sidecar, `spec.infer` returns it as it is or adds inferred decisions for new elements, and `spec.refine` refines it by a language model.
The phases of the cycle are the sections of the sidecar, such as a `cbrkit.retain.Retainer` under `retain`, so the sidecar and the code configure them with the same fields.
Custom measures serialize like built-in ones once they are `cbrkit.schema.serializable` and the sidecar is read with `cbrkit.schema.extend(spec.Config, Family("measure", (MyMeasure,)))`.

The `cli` extra runs the same steps without code, every command reads a JSON request (inline, `@file`, or `-` for stdin) and prints its outcome as JSON:

```shell
cbrkit infer data/cars-1k.csv          # writes data/cars-1k.csv.cbrkit.json
cbrkit retrieve data/cars-1k.csv '{"query": {"price": 20000, "fuel": "gas"}, "limit": 3, "explain": true}'
cbrkit retain data/cars-1k.csv @request.json
cbrkit maintain data/cars-1k.csv '{"dry_run": true}'
cbrkit schema                           # JSON Schema of the sidecar file
```

Optional sections of the sidecar enable reuse, revise, retain, and maintenance, and name the Pydantic AI models of embedders and language models, see `cbrkit.spec`.
`cbrkit.spec.providers` binds these names with the `pydantic-ai` extra, and hosts pass their own functions by name instead, see `cbrkit.system`.
Operations that the sidecar does not enable fail with exit code 1, and the case base is never written without a `retain` or `maintain` section.

## Quickstart: Serving over REST and MCP

The operations of a system are plain typed async functions of `cbrkit.system.operations.operations`, so the REST API, the MCP server, the command line, and the toolsets of agents offer the same names, descriptions, schemas, and behavior:

```shell
cbrkit serve data/cars-1k.csv                       # REST at http://127.0.0.1:8000, MCP at /mcp
CBRKIT_TOKEN=secret cbrkit serve data/cars-1k.csv --host 0.0.0.0
cbrkit mcp data/cars-1k.csv                         # MCP over stdio for local agents
```

Requests only carry cases, queries, keys and numbers, never paths or code, and every change of the case base is planned on the server from the sidecar.
A public address or a sidecar that writes the case base needs a token, which `CBRKIT_TOKEN` keeps out of the process list.
`cbrkit.system` describes the routes, tools, status codes and security defaults, and `cbrkit.system.api.app` returns the ASGI app to compose with your own application, see [examples/cars_system.py](https://github.com/wi2trier/cbrkit/blob/main/examples/cars_system.py).

## Logging

CBRkit logs through the module loggers of the `logging` module and configures no handler, so enable it in your application:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("cbrkit.cycle").setLevel(logging.DEBUG)
```

## Roadmap

### Next

- Release 2.0 and mark it as stable once Hivegent runs on it.
- Publish the JSON Schema of specifications (`cbrkit schema`) at a stable versioned URL, register it in SchemaStore for `*.cbrkit.json`, and let `infer` set it as the `$schema` of new specifications.

### Later

- Replace the lazy exports with `lazy from` imports and the first-party `frozendict` with the builtin once Python 3.15 is the minimum.
- Let `spec.mark_user` keep the reason given by a client.
