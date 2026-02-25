<!-- markdownlint-disable MD033 MD041 -->
<h1><p align="center">CBRkit</p></h1>

<p align="center">
  <img width="256px" alt="cbrkit logo" src="https://raw.githubusercontent.com/wi2trier/cbrkit/main/assets/logo.png" />
</p>

<p align="center">
  <a href="https://pypi.org/project/cbrkit/">PyPI</a> |
  <a href="https://wi2trier.github.io/cbrkit/">Docs</a> |
  <a href="https://github.com/wi2trier/cbrkit/tree/main/tests/test_retrieve.py">Example</a>
</p>

<p align="center">
  Customizable Case-Based Reasoning (CBR) toolkit for Python with a built-in API and CLI.
</p>

<p align="center">
  <a href="https://youtu.be/27dG4MagDhE">CBRkit Presentation</a>
  <br/>
  <i>ICCBR 2024 Best Student Paper</i>
</p>

---

<!-- PDOC_START -->

CBRkit is a customizable and modular toolkit for Case-Based Reasoning (CBR) in Python.
It provides a set of tools for loading cases and queries, defining similarity measures, and retrieving cases based on a query.
The toolkit is designed to be flexible and extensible, allowing you to define custom similarity measures or use built-in ones.
Retrieval pipelines are declared by composing these metrics, and the toolkit provides utility functions for applying them on a casebase.
Additionally, it offers ready-to-use API and CLI interfaces for easy integration into your projects.
The library is fully typed, enabling autocompletion and type checking in modern IDEs like VSCode and PyCharm.

To get started, we provide a [demo project](https://github.com/wi2trier/cbrkit-demo) which contains a casebase and a predefined retriever.
Further examples can be found in our [tests](./tests/test_retrieve.py) and [documentation](https://wi2trier.github.io/cbrkit/).
The following modules are part of CBRkit:

- `cbrkit.loaders`: Functions for loading cases and queries from various file formats and data sources.
- `cbrkit.dumpers`: Functions for exporting data to JSON, YAML, CSV, TOML, and Markdown.
- `cbrkit.sim`: Similarity measures for common data types with utility functions such as `cache`, `combine`, `transpose`, etc.
  - `cbrkit.sim.strings`: String similarity measures (Levenshtein, Jaro, spaCy, etc.).
  - `cbrkit.sim.numbers`: Numeric similarity measures (linear, exponential, threshold).
  - `cbrkit.sim.collections`: Similarity measures for collections and sequences (Jaccard, etc.).
  - `cbrkit.sim.embed`: Embedding-based similarity functions with caching support.
  - `cbrkit.sim.graphs`: Graph similarity algorithms including A\*, VF2, greedy, LAP, and more.
  - `cbrkit.sim.taxonomy`: Taxonomy-based similarity functions (Wu-Palmer, etc.).
  - `cbrkit.sim.generic`: Generic similarity functions (equality, tables, static).
  - `cbrkit.sim.attribute_value`: Similarity for attribute-value based data.
  - `cbrkit.sim.pooling`: Functions for aggregating multiple similarity values.
  - `cbrkit.sim.aggregator`: Combines multiple local measures into global scores.
- `cbrkit.adapt`: Adaptation functions for adapting cases based on a query.
- `cbrkit.retrieval`: Retrieval pipelines with BM25, embedding-based retrieval, re-ranking (Cohere, Voyage AI, Sentence Transformers), and more.
- `cbrkit.reuse`: Reuse pipelines that apply adaptation and score the results.
- `cbrkit.revise`: Revision pipelines for assessing and optionally repairing solutions.
- `cbrkit.retain`: Retention pipelines for storing solved cases back into the casebase.
- `cbrkit.cycle`: Full CBR cycle orchestration across all four phases.
- `cbrkit.system`: High-level `System` class for composing all CBR phases into a single object.
- `cbrkit.synthesis`: LLM-based synthesis for generating insights from cases (RAG), with providers for OpenAI, Anthropic, Cohere, Google, Ollama, and more.
- `cbrkit.eval`: Evaluation metrics for retrieval results including precision, recall, and custom metrics.
- `cbrkit.model`: Data models for results and graph structures.
- `cbrkit.typing`: Generic type definitions for defining custom functions.

## Installation

The library is available on [PyPI](https://pypi.org/project/cbrkit/), so you can install it with `pip`:

```shell
pip install cbrkit
```

It comes with several optional dependencies for certain tasks like NLP which can be installed with:

```shell
pip install cbrkit[EXTRA_NAME,...]
```

where `EXTRA_NAME` is one of the following:

- `all`: All optional dependencies
- **LLM providers:** `anthropic`, `cohere`, `google`, `ollama`, `openai`, `openai-agents`, `pydantic-ai`, `instructor`, `voyageai`
- **NLP / text processing:** `bm25`, `chunking`, `levenshtein`, `nltk`, `spacy`
- **ML / embeddings:** `transformers` (includes `pytorch` and `sentence-transformers`)
- **Graphs:** `graphs` (`networkx` and `rustworkx`), `graphviz`
- **Data backends:** `chromadb`, `lancedb`, `pandas`, `sql` (SQLAlchemy), `zvec`
- **Tools:** `cli` (CLI), `api` (REST API server), `eval` (evaluation metrics), `timeseries` (DTW, Smith-Waterman)

Alternatively, you can also clone this git repository and install CBRKit and its dependencies via uv: `uv sync --all-extras`

## Loading Cases

The first step is to load cases and queries.
We provide predefined functions for the following formats:

- csv
- json
- toml
- xml
- yaml
- txt (plain text)
- py (object inside of a python file)

Loading one of those formats can be done via the `file` function:

```python
import cbrkit
casebase = cbrkit.loaders.file("path/to/cases.[json,toml,yaml,xml,csv]")
```

You can also load all files from a directory or use the unified `path` function:

```python
# Load all files matching a glob pattern from a directory
casebase = cbrkit.loaders.directory("path/to/cases/", pattern="*.json")

# Unified path function: auto-detects whether path is a file or directory
casebase = cbrkit.loaders.path("path/to/cases.json")  # single file
casebase = cbrkit.loaders.path("path/to/cases/")      # directory
```

Additionally, CBRkit integrates with `polars` and `pandas` for loading data frames:

```python
import polars as pl
import cbrkit

df = pl.read_csv("path/to/cases.csv")
casebase = cbrkit.loaders.polars(df)
```

For database access, CBRkit provides `sqlite` and `sqlalchemy` loaders (the latter requires the `sql` extra):

```python
casebase = cbrkit.loaders.sqlite("path/to/database.db", "SELECT * FROM cases")
```

**Tip:** You can validate a loaded casebase against a Pydantic model using `cbrkit.loaders.validate()`:

```python
from pydantic import BaseModel

class Car(BaseModel):
    price: int
    year: int
    model: str

casebase = cbrkit.loaders.validate(casebase, Car)
```

## Defining Queries

CBRkit expects the type of the queries to match the type of the cases.
You may define a single query directly in Python as a dict:

```python
query = {"name": "John", "age": 25}
```

If you have a collection of queries, you can load them using the same loader functions as for the cases.

```python
 # for polars
queries = cbrkit.loaders.polars(pl.read_csv("path/to/queries.csv"))
# for any other supported file format
queries = cbrkit.loaders.file("path/to/queries.[json,toml,yaml,xml,csv]")
```

In case your query collection only contains a single entry, you can use the `singleton` function to extract it.

```python
query = cbrkit.helpers.singleton(queries)
```

## Exporting Data

CBRkit provides functions for exporting data through the `cbrkit.dumpers` module.
Supported formats include JSON, YAML, CSV, TOML, and Markdown.

```python
import cbrkit

# Export to a file (format is inferred from the extension)
cbrkit.dumpers.file("output.json", data)
cbrkit.dumpers.file("output.yaml", data)

# Export to a directory (one file per entry)
cbrkit.dumpers.directory("output/", data)

# Or use the unified path function
cbrkit.dumpers.path("output.json", data)  # writes a single file
cbrkit.dumpers.path("output/", data)      # writes a directory

# Format data as a Markdown code block
md = cbrkit.dumpers.markdown()(data)
```

## Similarity Measures and Aggregation

The next step is to define similarity measures for the cases and queries.
It is possible to define custom measures, use built-in ones, or combine both.

### Custom Similarity Measures

In CBRkit, a similarity measure is defined as a function that compares two arguments (a case and a query) and returns a similarity score: `sim = f(x, y)`.
It also supports pipeline-based similarity measures that are popular in NLP where a list of tuples is passed to the similarity measure: `sims = f([(x1, y1), (x2, y2), ...])`.
This generic approach allows you to define custom similarity measures for your specific use case.
For instance, the following function, which can be used to compare a string attribute of a case and a query, not only checks for strict equality, but also for partial matches (e.g., `x = "blue"` and `y = "light blue"`):

```python
def color_similarity(x: str, y: str) -> float:
    if x == y:
        return 1.0
    elif x in y or y in x:
        return 0.5

    return 0.0
```

**Please note:** CBRkit inspects the signature of custom similarity functions to perform some checks.
You need to make sure that the two parameters are named `x` and `y`, otherwise CBRkit will throw an error.

### Built-in Similarity Measures

CBRkit contains a comprehensive selection of built-in similarity measures for various data types in the module `cbrkit.sim`.
They are provided through **generator functions** that allow you to customize the behavior of the built-in measures.

#### String Similarity

```python
# Semantic similarity is covered by the `cbrkit.sim.embed` module.
# See below for details.

# Edit distance measures
levenshtein_sim = cbrkit.sim.strings.levenshtein()
jaro_sim = cbrkit.sim.strings.jaro()

# Exact matching
equality_sim = cbrkit.sim.generic.equality()
```

#### Number Similarity

```python
# Linear similarity with optional thresholds
linear_sim = cbrkit.sim.numbers.linear(max_distance=100)

# Exponential decay similarity
exp_sim = cbrkit.sim.numbers.exponential(alpha=0.1)

# Step functions
threshold_sim = cbrkit.sim.numbers.threshold(threshold=50)
```

#### Embedding-Based Similarity

```python
# Build a similarity function with embedding and scorer
embed_sim = cbrkit.sim.embed.build(
    conversion_func=cbrkit.sim.embed.sentence_transformers(
        model="all-MiniLM-L6-v2"
    ),
    sim_func=cbrkit.sim.embed.cosine()  # or dot(), angular(), euclidean(), manhattan()
)

# Using OpenAI embeddings
openai_sim = cbrkit.sim.embed.build(
    conversion_func=cbrkit.sim.embed.openai(
        model="text-embedding-3-small"
    ),
    sim_func=cbrkit.sim.embed.cosine()
)

# Caching embeddings for performance
cached_embed_func = cbrkit.sim.embed.cache(
    func=cbrkit.sim.embed.sentence_transformers(
        model="all-MiniLM-L6-v2"
    ),
    path="embeddings.sqlite3",
    table="strf/minilm"
)
cached_sim = cbrkit.sim.embed.build(
    conversion_func=cached_embed_func,
    sim_func=cbrkit.sim.embed.cosine()
)
```

#### Collection and Sequence Similarity

CBRkit provides similarity measures for collections and sequences in `cbrkit.sim.collections`:

```python
# Jaccard similarity for sets (requires the `nltk` extra)
jaccard_sim = cbrkit.sim.collections.jaccard()

# Optimal sequence mapping using A* search
seq_sim = cbrkit.sim.collections.mapping(cbrkit.sim.generic.equality())
```

Dynamic Time Warping and Smith-Waterman alignment are available with the `timeseries` extra.
See the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/sim/collections.html) for the full list.

#### Taxonomy-Based Similarity

```python
# Load taxonomy from file
taxonomy_sim = cbrkit.sim.taxonomy.build(
    path="taxonomy.yaml",
    measure=cbrkit.sim.taxonomy.wu_palmer(),
)
```

#### Utility Functions

```python
# Combining multiple similarity functions
combined_sim = cbrkit.sim.combine(
    sim_funcs=[sim1, sim2, sim3],
    aggregator=cbrkit.sim.aggregator(pooling="mean")
)

# Caching similarity results
cached_sim = cbrkit.sim.cache(base_sim_func)

# Transposing similarity functions
transposed_sim = cbrkit.sim.transpose(
    sim_func=number_sim,
    to_x=lambda s: float(s),
    to_y=lambda s: float(s)
)
```

**Please note:** Calling these functions returns a similarity function itself that has the signature `sim = f(x, y)`.

An overview of all available similarity measures can be found in the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/sim.html).

### Graph Similarity

CBRkit provides extensive support for graph similarity through various algorithms:

```python
# Using Graph Edit Distance (GED) with A* search
graph_sim = cbrkit.sim.graphs.astar.build(
    node_sim_func=cbrkit.sim.generic.equality(),
    node_matcher=lambda n1, n2: n1 == n2,
    edge_matcher=lambda e1, e2: e1 == e2,
)
```

Available graph algorithms include `astar`, `vf2`, `greedy`, `lap`, `brute_force`, `dfs`, `dtw`, and `smith_waterman`.
See the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/sim/graphs.html) for a full list of algorithms and their parameters.

### Global Similarity and Aggregation

When dealing with cases that are not represented through elementary data types like strings, we need to aggregate individual measures to obtain a global similarity score.
We provide a predefined `aggregator` that transforms a list of similarities into a single score.
It can be used with custom and/or built-in measures.

```python
similarities = [0.8, 0.6, 0.9]
aggregator = cbrkit.sim.aggregator(pooling="mean")
global_similarity = aggregator(similarities)
```

For the common use case of attribute-value based data, CBRkit provides a predefined global similarity measure that can be used as follows:

```python
cbrkit.sim.attribute_value(
    attributes={
        "price": cbrkit.sim.numbers.linear(),
        "color": color_similarity # custom measure
        ...
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)
```

The `attribute_value` function lets you define measures for each attribute of the cases/queries as well as the aggregation function.
It also allows to use custom measures like the `color_similarity` function defined above.

**Please note:** The custom measure is not executed (i.e., there are **no** parenthesis at the end), but instead passed as a reference to the `attribute_value` function.

You may even nest similarity functions to create measures for object-oriented cases:

```python
cbrkit.sim.attribute_value(
    attributes={
        "manufacturer": cbrkit.sim.attribute_value(
            attributes={
                "name": cbrkit.sim.strings.spacy(model="en_core_web_lg"),
                "country": cbrkit.sim.strings.levenshtein(),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        "color": color_similarity # custom measure
        ...
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)
```

## CBR Cycle Phases

All four phases of the CBR cycle — retrieval, reuse, revise, and retain — follow the same unified protocol `CbrFunc` (defined in `cbrkit.typing`).
Each phase function takes a casebase and a query, and returns an updated casebase together with a score map.
The casebase in the output may differ from the input depending on the phase (e.g., adapted cases in reuse, newly stored cases in retain).
The score map assigns a floating-point score to each case in the output casebase, with phase-specific semantics:

- **Retrieval**: Similarity scores between cases and the query.
- **Reuse**: Quality scores of adapted cases compared to the query.
- **Revise**: Assessment scores evaluating solution correctness.
- **Retain**: Fitness scores for retained cases.

This uniform interface makes it easy to compose phases into pipelines and to swap implementations.
The phase-specific type aliases `RetrieverFunc`, `ReuserFunc`, `ReviserFunc`, and `RetainerFunc` are provided for clarity but are structurally identical to `CbrFunc`.

Each phase result has the following attributes:

- `similarities`: A dictionary containing the scores for each case.
- `ranking`: A list of case keys sorted by their score.
- `casebase`: The casebase containing the output cases.

## Retrieval

The first phase is to retrieve cases based on the loaded queries.
The `cbrkit.retrieval` module provides utility functions for this purpose.
You first build a retrieval pipeline by specifying a global similarity function and optionally a limit for the number of retrieved cases.

```python
retriever = cbrkit.retrieval.build(
    cbrkit.sim.attribute_value(...)
)
```

This retriever can then be applied on a casebase to retrieve cases for a given query.

```python
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

Our result has the following attributes:

- `similarities`: A dictionary containing the similarity scores for each case.
- `ranking` A list of case indices sorted by their similarity score.
- `casebase` The casebase containing only the retrieved cases (useful for downstream tasks).

An example using the provided `cars-1k` dataset can be found under [examples/cars_retriever.py](https://github.com/wi2trier/cbrkit/blob/main/examples/cars_retriever.py).

In some cases, it is useful to combine multiple retrieval pipelines, for example when applying the MAC/FAC pattern where a cheap pre-filter is applied to the whole casebase before a more expensive similarity measure is applied on the remaining cases.
To use this pattern, first create the corresponding retrievers using the builder:

```python
retriever1 = cbrkit.retrieval.dropout(..., min_similarity=0.5, limit=20)
retriever2 = cbrkit.retrieval.dropout(..., limit=10)
```

Then apply all of them sequentially by passing them as a list or tuple to the `apply_query` function:

```python
result = cbrkit.retrieval.apply_query(casebase, query, (retriever1, retriever2))
```

The result has the following two attributes:

- `final_step`: Result of the last retriever in the list.
- `steps`: A list of results for each retriever in the list.

Both `final_step` and each entry in `steps` have the same attributes as discussed previously.
The returned result also has these entries which are an alias for the corresponding entries in `final_step` (i.e., `result.ranking == result.final_step.ranking`).

## Adaptation Functions

Case adaptation is a crucial step in the CBR cycle that allows us to modify retrieved cases to better suit the current query.
CBRkit offers both built-in adaptation functions for common scenarios and the flexibility to define custom adaptation logic.

**Please note:** `cbrkit.adapt` contains the built-in adaption functions. To apply these (or custom adaption functions) to your actual casebase, please refer to [Reuse](#reuse).

### Custom Adaptation Functions

In CBRkit, an adaptation function is defined as a function that takes two arguments (a case and a query) and returns an adapted case: `adapted = f(case, query)`.
For more complex scenarios, CBRkit also supports two additional types of adaptation functions:

- Map functions that operate on the entire casebase: `adapted = f(casebase, query)`
- Reduce functions that select and adapt a single case: `key, adapted = f(casebase, query)`

This generic approach allows you to define custom adaptation functions for your specific use case.
For instance, the following function replaces a case value with the query value if they differ:

```python
def replace_adapter(case: str, query: str) -> str:
    return query if case != query else case
```

**Please note:** CBRkit inspects the signature of custom adaptation functions to determine their type.
Make sure that the parameters are named either `case` and `query` for pair functions, or `casebase` and `query` for map/reduce functions.

### Built-in Adaptation Functions

CBRkit contains adaptation functions for common data types like strings and numbers in the module `cbrkit.adapt`.
They are provided through **generator functions** that allow you to customize the behavior of the built-in functions.
For example, a number aggregator can be obtained as follows:

```python
# pooling can be a string like "mean", "min", "max", "sum", etc. or a custom PoolingFunction
number_adapter = cbrkit.adapt.numbers.aggregate(pooling="mean")
```

**Please note:** Calling the function `cbrkit.adapt.numbers.aggregate` returns an adaptation function that takes a collection of values and returns an adapted value.

For the common use case of attribute-value based data, CBRkit provides a predefined global adapter that can be used as follows:

```python
cbrkit.adapt.attribute_value(
    attributes={
        "price": cbrkit.adapt.numbers.aggregate(),
        "color": cbrkit.adapt.strings.regex("CASE_PATTERN", "QUERY_PATTERN", "REPLACEMENT"),
        ...
    }
)
```

The `attribute_value` function lets you define adaptation functions for each attribute of the cases.
You may even nest adaptation functions to handle object-oriented cases.

An overview of all available adaptation functions can be found in the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/adapt.html).

## Reuse

The reuse phase applies adaptation functions to retrieved cases and scores the adapted results.
The `cbrkit.reuse` module provides utility functions for this purpose.
You build a reuse pipeline by specifying an adaptation function and a similarity function:

```python
reuser = cbrkit.reuse.build(
    adaptation_func=cbrkit.adapt.attribute_value(...),
    similarity_func=cbrkit.sim.attribute_value(...),
)
```

This reuser can then be applied to a retrieval result to adapt cases based on a query:

```python
result = cbrkit.reuse.apply_result(retrieval_result, reuser)
```

As with all CBR phases, the result contains `similarities` (quality scores of adapted cases), `ranking`, and `casebase` (containing the adapted cases).

Multiple reuse pipelines can be combined by passing them as a list or tuple:

```python
reuser1 = cbrkit.reuse.build(...)
reuser2 = cbrkit.reuse.build(...)
result = cbrkit.reuse.apply_result(retrieval_result, (reuser1, reuser2))
```

The result structure follows the same pattern as the retrieval results with `final_step` and `steps` attributes.

## Revise

The revise phase assesses the quality of solutions produced by the reuse phase and optionally repairs them.
The `cbrkit.revise` module provides utility functions for this purpose.
You build a revise pipeline by specifying an assessment function and an optional repair function:

```python
reviser = cbrkit.revise.build(
    assess_func=cbrkit.sim.attribute_value(...),
    repair_func=some_adaptation_func,  # optional
)
```

The reviser can be applied to a reuse result:

```python
result = cbrkit.revise.apply_result(reuse_result, reviser)
```

When a `repair_func` is provided, solutions are repaired before assessment.
The result contains `similarities` with quality assessment scores for each case.

## Retain

The retain phase decides whether and how to integrate new cases into the casebase.
The `cbrkit.retain` module provides utility functions for this purpose.
You build a retain pipeline by specifying an assessment function and a storage function:

```python
retainer = cbrkit.retain.build(
    assess_func=cbrkit.sim.generic.equality(),
    storage_func=cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    ),
)
```

CBRkit provides several built-in storage functions:

- `static`: Generates keys from a fixed reference casebase to avoid collisions.
- `indexable`: Keeps an `IndexableFunc`'s index in sync with the casebase.

You can filter retained cases based on their assessment scores using the `dropout` wrapper:

```python
retainer = cbrkit.retain.dropout(
    retainer_func=cbrkit.retain.build(...),
    min_similarity=0.5,
)
```

The retainer can be applied to a revise result:

```python
result = cbrkit.retain.apply_result(revise_result, retainer)
```

The result contains `similarities` with fitness scores and `casebase` with the updated cases.

## Full CBR Cycle

The `cbrkit.cycle` module orchestrates all four phases (retrieval, reuse, revise, retain) in a single call.
This is useful when you want to run the complete CBR cycle without manually chaining the phases.

```python
result = cbrkit.cycle.apply_query(
    casebase,
    query,
    retrievers=retriever,
    reusers=reuser,
    revisers=reviser,
    retainers=retainer,
)
# Access results from each phase
retrieval_result = result.retrieval
reuse_result = result.reuse
revise_result = result.revise
retain_result = result.retain
```

For multiple queries, use `cbrkit.cycle.apply_queries` or `cbrkit.cycle.apply_batches`.

## System

The `cbrkit.system.System` class provides a high-level interface for composing all CBR phases into a single reusable object.
It is especially useful for integrating CBRkit into applications where the casebase and phase functions are configured once and reused across multiple queries.

```python
system = cbrkit.system.System(
    casebase=casebase,
    retriever_factory=lambda config: retriever,
    reuser_factory=lambda config: reuser,
)

# Run individual phases
retrieval_result = system.retrieve(query)
reuse_result = system.reuse(query)

# Run the full cycle
cycle_result = system.cycle(query)
```

The `System` class supports optional configuration parameters for each phase factory, allowing you to customize the behavior per query.

## Advanced Retrieval

### BM25 Retrieval

CBRkit includes a BM25 retriever for sparse text-based retrieval (requires the `bm25` extra).
The BM25 retriever delegates text tokenization to a `cbrkit.sim.embed.bm25` embedding function:

```python
bm25_func = cbrkit.sim.embed.bm25(language="en")
retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.bm25(conversion_func=bm25_func),
    limit=10,
)
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

### Embedding-Based Retrieval

CBRkit supports embedding-based retrieval through vector similarity search.
The `embed` retriever uses an embedding function with caching and a vector scorer:

```python
embed_func = cbrkit.sim.embed.cache(
    func=cbrkit.sim.embed.sentence_transformers(model="all-MiniLM-L6-v2"),
    path="embeddings.sqlite3",
    table="strf/minilm",
)
retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.embed(conversion_func=embed_func),
    limit=10,
)
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

For persistent storage backends, CBRkit also supports `lancedb`, `chromadb`, and `zvec` retrievers (each requires its respective extra).
These backends manage index persistence and support hybrid search modes.

### Combining Multiple Retrievers

The `combine` function allows merging results from multiple retrievers:

```python
retriever1 = cbrkit.retrieval.build(...)
retriever2 = cbrkit.retrieval.bm25(...)

combined = cbrkit.retrieval.combine(
    retriever_funcs=[retriever1, retriever2],
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)
result = cbrkit.retrieval.apply_query(casebase, query, combined)
```

### Distributed Processing

`build` and `distribute` offer two different levels of parallelism.
`build(sim_func, multiprocessing=True)` parallelizes the **similarity computations** within batches: all (casebase, query) pairs are flattened into individual comparisons and distributed across processes.
`distribute(retriever, multiprocessing=True)` parallelizes across **batches**: each (casebase, query) pair is passed to the wrapped retriever as a separate process.
Use `distribute` when you have many independent queries and want to process them in parallel as separate retrieval tasks:

```python
retriever = cbrkit.retrieval.distribute(
    cbrkit.retrieval.build(...),
    multiprocessing=True,  # or an integer for a specific number of processes
)
```

### Re-ranking

CBRkit supports re-ranking retrieved results using external models.
Re-rankers take the initial retrieval results and reorder them based on a more expensive model.
The following re-rankers are available (each requires its respective extra):

- `cbrkit.retrieval.cohere`: Cohere re-ranking (extra: `cohere`)
- `cbrkit.retrieval.voyageai`: Voyage AI re-ranking (extra: `voyageai`)
- `cbrkit.retrieval.sentence_transformers`: Sentence Transformers cross-encoder re-ranking (extra: `transformers`)

```python
reranker = cbrkit.retrieval.cohere(model="rerank-v3.5")

# Use as a second-stage retriever in a sequential pipeline
retriever = cbrkit.retrieval.build(cbrkit.sim.attribute_value(...))
result = cbrkit.retrieval.apply_query(casebase, query, (retriever, reranker))
```

### Indexed Retrieval

Some retrievers like `bm25`, `embed`, and `lancedb` support **indexed retrieval**, where the casebase is pre-indexed once and then queried without passing the full casebase each time.
This is useful for large casebases or when using external search backends.

To use indexed retrieval, first create a retriever and call its `index()` method:

```python
from frozendict import frozendict

bm25_func = cbrkit.sim.embed.bm25(language="en")
retriever = cbrkit.retrieval.bm25(conversion_func=bm25_func)
retriever.create_index(frozendict(casebase))
```

Then pass an empty casebase (`{}`) to signal that the retriever should use its pre-indexed data:

```python
result = cbrkit.retrieval.apply_query({}, query, retriever)
```

As a convenience, CBRkit provides `apply_query_indexed` and `apply_queries_indexed` which handle the empty casebase automatically:

```python
result = cbrkit.retrieval.apply_query_indexed(query, retriever)
# or for multiple queries:
result = cbrkit.retrieval.apply_queries_indexed(queries, retriever)
```

If a retriever receives an empty casebase but has not been indexed yet, a `ValueError` is raised with a message to call `index()` first.

The `System` class also supports indexed retrieval by defaulting the casebase to an empty dict.
This allows creating a system where all retrievers are pre-indexed and no casebase needs to be provided at query time.

## Evaluation

CBRkit provides evaluation tools through the `cbrkit.eval` module for assessing the quality of retrieval results.
The basic evaluation function `cbrkit.eval.compute` expects the following arguments:

- `qrels`: Ground truth relevance scores for query-case pairs. A higher value means a higher relevance.
- `run`: Retrieval similarity scores for query-case pairs.
- `metrics`: A list of metrics to compute.

You can evaluate retrieval results directly with the functions `cbrkit.eval.retrieval` and `cbrkit.eval.retrieval_step`.

### Custom Metrics

Users can provide custom metric functions that implement the signature defined in the `cbrkit.typing.EvalMetricFunc` protocol:

```python
def custom_metric(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
    # Custom metric logic here
    return 0.0
```

You can then pass your custom metric function to the `compute` function:

```python
results = cbrkit.eval.compute(
    qrels,
    run,
    metrics=["custom_metric@5"],
    metric_funcs={"custom_metric": custom_metric},
)
```

### Built-in Metrics

The module also supports standard Information Retrieval metrics through ranx like `precision`, `recall`, and `f1`.
A complete list is available in the [ranx documentation](https://amenra.github.io/ranx/metrics/).
Additionally, CBRkit provides two custom metrics not available in ranx:

- `correctness`: Measures how well the ranking preserves the relevance ordering (-1 to 1).
- `completeness`: Measures what fraction of relevance pairs are preserved (0 to 1).

All of them can be computed at different cutoff points by appending `@k`, e.g., `precision@5`.
We also offer a function to automatically generate a list of metrics for different cutoff points:

```python
metrics = cbrkit.eval.generate_metrics(["precision", "recall", "f1"], ks=[1, 5, 10])
```

## Synthesis

In the context of CBRkit, synthesis refers to creating new insights from the cases which were retrieved in a previous retrieval step, for example in a RAG context. CBRkit builds a synthesizer using the function `cbrkit.synthesis.build` with a `provider` and a `prompt`. A synthesizer maps a `Result` (obtained in the retrieval step) to an LLM output (can be a string or structurized). An example can be found in [examples/cars_rag.py](https://github.com/wi2trier/cbrkit/blob/main/examples/cars_rag.py).

The following **providers** are available in `cbrkit.synthesis.providers` (each requires its respective extra):

- `openai` / `openai_completions`: OpenAI Completions API (`OPENAI_API_KEY`)
- `openai_responses`: OpenAI Responses API (`OPENAI_API_KEY`)
- `openai_agents`: OpenAI Agents framework (`OPENAI_API_KEY`)
- `anthropic`: Anthropic Claude API (`ANTHROPIC_API_KEY`)
- `cohere`: Cohere API (`CO_API_KEY`)
- `google`: Google Generative AI (`GOOGLE_API_KEY`)
- `ollama`: Ollama (local, no API key needed)
- `pydantic_ai`: Pydantic AI framework
- `instructor`: Instructor for structured output

Providers can be chained using `cbrkit.synthesis.providers.pipe()` and managed as conversations using `cbrkit.synthesis.providers.conversation()`.

The respective provider class in `cbrkit.synthesis.providers` has to be initialized with the model name and a response type (either `str` or a [Pydantic model](https://docs.pydantic.dev/latest/concepts/models/) for structured output). Further model options like `temperature`, `seed`, `max_tokens`, etc. can also be specified here.

A **prompt** produces an LLM input based on the specified `instructions`, an optional `encoder` (which maps a case or query to a string) and optional `metadata`. For a list of the currently included prompts, please refer to the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/synthesis/prompts.html)

If the casebase is small enough, that it fits inside the LLM's context window, you can use CBRKit's synthesis as follows:

```python
import cbrkit

casebase = cbrkit.loaders.LOADER(...)
retriever = cbrkit.retrieval.build(...)
retrieval = cbrkit.retrieval.apply_query(...)
provider = cbrkit.synthesis.providers.PROVIDER(...)
prompt = cbrkit.synthesis.prompts.default(instructions)
synthesizer = cbrkit.synthesis.build(provider, prompt)
response = cbrkit.synthesis.apply_result(retrieval, synthesizer).response
```

### Working with large casebases

Because the built-in `default` and `document_aware` prompt functions include the entire casebase as context, the LLM input can be quite long when working with a large casebase.
Because of this, in this case, we recommend transposing the cases (e.g., truncate every case to a fixed length) and/or apply chunking.

#### Transposing cases

CBRKit's `transpose` prompt allows to transpose cases and queries before they are passed to the main prompt function. This allows shortening entries like so:

```python
from cbrkit.typing import JsonEntry

def encoder(value) -> dict:
    ...
baseprompt = cbrkit.synthesis.prompts.default(instructions, encoder=encoder)
# transform the entries, e.g., by shortening, leaving out irrelevant attributes, etc.
# In this case, the value of every field is truncated to 100 characters
def shorten(entry: dict) -> JsonEntry:
    entry = {k: str(v)[:100] for k, v in entry.items()}
    return cbrkit.dumpers.markdown()(entry)

prompt = cbrkit.synthesis.prompts.transpose(baseprompt, shorten)
synthesizer = cbrkit.synthesis.build(provider, prompt)
...
```

#### Chunking

Instead of using `cbrkit.synthesis.apply_result`, CBRKit also provides the `cbrkit.synthesis.chunks` function to process the synthesis in batches. The partial results can then be aggregated using a `pooling` prompt.

```python
import cbrkit

casebase = cbrkit.loaders.file(...)

queries = [...]
retriever = cbrkit.retrieval.dropout(...)
# run retrieval on each casebase chunk
retrievals = [cbrkit.retrieval.apply_query(casebase, query, retriever) for query in queries]

# batches are tuples of casebase, query, and retrieval similarities
batches = [(casebase, query, retrieval.similarities) for query, retrieval in zip(queries, retrievals)]

# Prompt which should be evaluated on each batch
prompt = cbrkit.synthesis.prompts.default(instructions="...")
provider = cbrkit.synthesis.providers.PROVIDER(...)
synthesizer = cbrkit.synthesis.build(provider, prompt)

# prompt to aggregate the partial results into a final result
pooling_prompt = cbrkit.synthesis.prompts.pooling(instructions="...")
pooling_func = cbrkit.synthesis.pooling(provider, pooling_prompt)
get_result = cbrkit.synthesis.chunks(synthesizer, pooling_func, size=10)
response = get_result(batches)
```

The complete version of this example can be found under `examples/cars_rag_large.py`.

## Tips and Common Patterns

### Parameter Naming Conventions

CBRkit inspects function signatures to determine their behavior:

- **Similarity functions** must use `x` (case) and `y` (query) as parameter names.
- **Adaptation functions** must use `case` and `query` for pair functions, or `casebase` and `query` for map/reduce functions.
- **Batch functions** accept a list of tuples instead of individual pairs: `f([(x1, y1), (x2, y2), ...])`.

### Filtering with `dropout`

The `dropout` wrapper is the standard way to add limits and thresholds to any retriever or retainer.
It supports `limit` (maximum number of results), `min_similarity`, and `max_similarity`:

```python
retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(sim_func),
    limit=10,
    min_similarity=0.3,
)
```

### Composing Multiple Phase Functions

All CBR phases support sequential composition by passing a tuple of phase functions.
Each step receives the output casebase of the previous step, enabling patterns like MAC/FAC:

```python
result = cbrkit.retrieval.apply_query(casebase, query, (cheap_retriever, expensive_retriever))
```

### Using `frozendict` for Immutable Casebases

Several components (e.g., indexed retrieval, retain phase) benefit from immutable casebases.
Use `frozendict` to prevent accidental mutations:

```python
from frozendict import frozendict
casebase = frozendict(cbrkit.loaders.file("cases.json"))
```

### Multiprocessing Support

The `cbrkit.retrieval.build` function supports multiprocessing to parallelize similarity computations within batches:

```python
retriever = cbrkit.retrieval.build(sim_func, multiprocessing=True)
# or with a specific number of processes:
retriever = cbrkit.retrieval.build(sim_func, multiprocessing=4)
```

To parallelize across batches instead, see [Distributed Processing](#distributed-processing).

## Logging

CBRkit integrates with the `logging` module to provide a unified logging interface.
By default, logging is not configured, you can activate by placing the following code in your project's `__init__.py` file:

```python
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("cbrkit.sim.XXX").setLevel(logging.DEBUG) # to see debug messages from function XXX of the sim module
logging.getLogger("cbrkit.retrieval").setLevel(logging.DEBUG) # to see debug messages from the retrieval module
```

## REST API and CLI

In order to use the built-in API and CLI, you need to define a retriever/reuser in a Python module using the function `cbrkit.retrieval.build()` and/or `cbrkit.reuse.build()`.
For example, the file `./retriever_module.py` could contain the following code:

```python
import cbrkit

custom_retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(...),
    ),
    limit=10,
)
```

Our custom retriever can then be specified for the API/CLI using standard Python module syntax: `retriever_module:custom_retriever`.

### CLI

When installing with the `cli` extra, CBRkit provides a command line interface:

```shell
cbrkit --help
```

Please visit the [documentation](https://wi2trier.github.io/cbrkit/cbrkit/cli.html) for more information on how to use the CLI.

### API

When installing with the `api` extra, CBRkit provides a REST API server:

```shell
cbrkit serve --help
```

After starting the server, you can access the API documentation at `http://localhost:8080/docs`.
