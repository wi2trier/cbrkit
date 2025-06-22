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

- `cbrkit.loaders` and `cbrkit.dumpers`: Functions for loading and exporting cases and queries.
- `cbrkit.sim`: Similarity functions for common data types and some utility functions such as `cache`, `combine`, `transpose`, etc.
  - `cbrkit.sim.strings`: String similarity measures (Levenshtein, Jaro, semantic, etc.).
  - `cbrkit.sim.numbers`: Numeric similarity measures (linear, exponential, threshold).
  - `cbrkit.sim.collections`: Similarity measures for collections and sequences (Jaccard, DTW, Smith-Waterman).
  - `cbrkit.sim.embed`: Embedding-based similarity functions with caching support.
  - `cbrkit.sim.graphs`: Graph similarity algorithms including GED, A*, VF2, and more.
  - `cbrkit.sim.taxonomy`: Taxonomy-based similarity functions.
  - `cbrkit.sim.generic`: Generic similarity functions (equality, tables, static).
  - `cbrkit.sim.attribute_value`: Similarity for attribute-value based data.
  - `cbrkit.sim.pooling`: Functions for aggregating multiple similarity values.
  - `cbrkit.sim.aggregator`: Combines multiple local measures into global scores.
- `cbrkit.retrieval`: Functions for defining and applying retrieval pipelines, includes BM25 retrieval, rerankers, etc.
- `cbrkit.adapt`: Adaptation generator functions for adapting cases based on a query.
- `cbrkit.reuse`: Functions for defining and applying reuse pipelines.
- `cbrkit.eval`: Evaluation metrics for retrieval results including precision, recall, and custom metrics.
- `cbrkit.model`: Data models for graphs and results.
- `cbrkit.cycle`: CBR cycle implementation.
- `cbrkit.typing`: Generic type definitions for defining custom functions.
- `cbrkit.synthesis`: Functions for working on a casebase with LLMs to create new insights, e.g., in a RAG context.

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
- `api`: REST API Server
- `cli`: Command Line Interface (CLI)
- `eval`: Evaluation tools for common metrics like `precision` and `recall`
- `graphs`: Graph libraries like `networkx` and `rustworkx`
- `llm`: Large Language Models (LLM) APIs like Ollama and OpenAI
- `nlp`: Standalone NLP tools `levenshtein`, `nltk`, `openai`, and `spacy`
- `timeseries`: Time series similarity measures like `dtw` and `smith_waterman`
- `transformers`: Advanced NLP tools based on `pytorch` and `transformers`

Alternatively, you can also clone this git repository and install CBRKit and its dependencies via uv: `uv sync --all-extras`

## Loading Cases

The first step is to load cases and queries.
We provide predefined functions for the following formats:

- csv
- json
- toml
- xml
- yaml
- py (object inside of a python file).

Loading one of those formats can be done via the `file` function:

```python
import cbrkit
casebase = cbrkit.loaders.file("path/to/cases.[json,toml,yaml,xml,csv]")
```

Additionally, CBRkit also integrates with `polars` and `pandas` for loading data frames.
The following example shows how to load cases and queries from a CSV file using `polars`:

```python
import polars as pl
import cbrkit

df = pl.read_csv("path/to/cases.csv")
casebase = cbrkit.loaders.polars(df)
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
    path="embeddings_cache.npz",
    autodump=True,
    autoload=True
)
cached_sim = cbrkit.sim.embed.build(
    conversion_func=cached_embed_func,
    sim_func=cbrkit.sim.embed.cosine()
)
```

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
graph_sim = cbrkit.sim.graphs.astar(
    node_sim=cbrkit.sim.generic.equality(),
    node_matcher=lambda n1, n2: n1 == n2,
    edge_matcher=lambda e1, e2: e1 == e2
)
```

Available graph algorithms include:
- `astar`: A* search for optimal graph edit distance
- `vf2`: VF2 algorithm for (sub)graph isomorphism
- `lap`: Linear assignment problem solver
- `greedy`: Fast greedy matching
- `brute_force`: Exhaustive search for small graphs
- `dfs`: Depth-first search based matching

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

## Retrieval

The final step is to retrieve cases based on the loaded queries.
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

The reuse phase applies adaptation functions to retrieved cases. The `cbrkit.reuse` module provides utility functions for this purpose. You first build a reuse pipeline by specifying a global adaptation function:

```python
reuser = cbrkit.reuse.build(
    cbrkit.adapt.attribute_value(...),
)
```

This reuser can then be applied to the retrieval result to adapt cases based on a query:

```python
result = cbrkit.reuse.apply_query(retrieval_result, query, reuser)
```

Our result has the following attributes:

- `adaptations`: A dictionary containing the adapted values for each case.
- `ranking`: A list of case indices matching the retrieval result.
- `casebase`: The casebase containing only the adapted cases.

Multiple reuse pipelines can be combined by passing them as a list or tuple:

```python
reuser1 = cbrkit.reuse.build(...)
reuser2 = cbrkit.reuse.build(...)
result = cbrkit.reuse.apply_query(retrieval_result, query, (reuser1, reuser2))
```

The result structure follows the same pattern as the retrieval results with `final_step` and `steps` attributes.

## Advanced Retrieval

### BM25 Retrieval

CBRkit includes a BM25 retriever for text-based retrieval:

```python
retriever = cbrkit.retrieval.bm25(
    key="text_field",  # Field to search in
    limit=10
)
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

### Combining Multiple Retrievers

The `combine` function allows merging results from multiple retrievers:

```python
retriever1 = cbrkit.retrieval.build(...)
retriever2 = cbrkit.retrieval.bm25(...)

combined = cbrkit.retrieval.combine(
    retrievers=[retriever1, retriever2],
    aggregator=cbrkit.sim.aggregator(pooling="mean")
)
result = cbrkit.retrieval.apply_query(casebase, query, combined)
```

### Distributed Processing

For large-scale retrieval, use the `distribute` wrapper:

```python
retriever = cbrkit.retrieval.distribute(
    cbrkit.retrieval.build(...),
    batch_size=1000
)
```

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
metrics = cbrkit.eval.metrics_at_k(["precision", "recall", "f1"], [1, 5, 10])
```

## Synthesis

In the context of CBRkit, synthesis refers to creating new insights from the cases which were retrieved in a previous retrieval step, for example in a RAG context. CBRkit builds a synthesizer using the function `cbrkit.synthesis.build` with a `provider` and a `prompt`. A synthesizer maps a `Result` (obtained in the retrieval step) to an LLM output (can be a string or structurized). An example can be found in [examples/cars_rag.py](https://github.com/wi2trier/cbrkit/blob/main/examples/cars_rag.py).

The following **providers** are currently supported if a valid API key is stored the respective environment variable:

- Anthropic (`ANTHROPIC_API_KEY`)
- Cohere (`CO_API_KEY`)
- Google (`GOOGLE_API_KEY`)
- Ollama
- OpenAI (`OPENAI_API_KEY`)

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
from cbrkit.dumpers import json_markdown

def encoder(value) -> dict:
    ...
baseprompt = cbrkit.synthesis.prompts.default(instructions, encoder=encoder)
# transform the entries, e.g., by shortening, leaving out irrelevant attributes, etc.
# In this case, the value of every field is trunctated to 100 characters
def shorten(entry: dict) -> JsonEntry:
    entry = {k: str(v)[:100] for k,v in entry.items()}
    return json_markdown(entry)

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
