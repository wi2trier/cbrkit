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

- `cbrkit.loaders`: Functions for loading cases and queries.
- `cbrkit.sim`: Similarity generator functions for common data types like strings and numbers.
- `cbrkit.retrieval`: Functions for defining and applying retrieval pipelines.
- `cbrkit.adapt`: Adaptation generator functions for adapting cases based on a query.
- `cbrkit.reuse`: Functions for defining and applying reuse pipelines.
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
- `api`: REST API Server
- `cli`: Command Line Interface (CLI)
- `eval`: Evaluation tools for common metrics like `precision` and `recall`
- `llm`: Large Language Models (LLM) APIs like Ollama and OpenAI
- `nlp`: Standalone NLP tools `levenshtein`, `nltk`, `openai`, and `spacy`
- `timeseries`: Time series similarity measures like `dtw` and `smith_waterman`
- `transformers`: Advanced NLP tools based on `pytorch` and `transformers`

## Loading Cases

The first step is to load cases and queries.
We provide predefined functions for the most common formats like CSV, JSON, and XML.
Additionally, CBRkit also integrates with `polars` and `pandas` for loading data frames.
The following example shows how to load cases and queries from a CSV file using `polars`:

```python
import polars as pl
import cbrkit

df = pl.read_csv("path/to/cases.csv")
casebase = cbrkit.loaders.polars(df)
```

When dealing with formats like json, toml, yaml, or xml, the files can be loaded using

```python
casebase = cbrkit.loaders.file("path/to/cases.<json,toml,yaml,xml,csv>")
```

## Defining Queries

CBRkit expects the type of the queries to match the type of the cases.
You may define a single query directly in Python as follows

```python
query = {"name": "John", "age": 25}
```

If you have a collection of queries, you can load them using the same loader functions as for the cases.

```python
 # for polars
queries = cbrkit.loaders.polars(pl.read_csv("path/to/queries.csv"))
# for json
queries = cbrkit.loaders.json("path/to/queries.json")
```

In case your query collection only contains a single entry, you can use the `singleton` function to extract it.

```python
query = cbrkit.helpers.singleton(queries)
```

## Similarity Measures and Aggregation

The next step is to define similarity measures for the cases and queries.
It is possible to define custom measures, use built-in ones, or combine both.

### Custom Similarity Measures

In CBRkit, a similarity measure is defined as a function that takes two arguments (a case and a query) and returns a similarity score: `sim = f(x, y)`.
It also supports pipeline-based similarity measures that are popular in NLP where a list of tuples is passed to the similarity measure: `sims = f([(x1, y1), (x2, y2), ...])`.
This generic approach allows you to define custom similarity measures for your specific use case.
For instance, the following function not only checks for strict equality, but also for partial matches (e.g., `x = "blue"` and `y = "light blue"`):

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

CBRkit also contains a selection of built-in similarity measures for the most common data types in the module `cbrkit.sim`.
They are provided through **generator functions** that allow you to customize the behavior of the built-in measures.
For example, an spacy-based embedding similarity measure can be obtained as follows:

```python
semantic_similarity = cbrkit.sim.strings.spacy(model="en_core_web_lg")
```

**Please note:** Calling the function `cbrkit.sim.strings.spacy` returns a similarity function itself that has the same signature as the `color_similarity` function defined above.

An overview of all available similarity measures can be found in the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/sim.html).

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
    cbrkit.sim.attribute_value(...),
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
number_adapter = cbrkit.adapt.numbers.aggregate()
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

## Evaluation

CBRkit provides evaluation tools through the `cbrkit.eval` module for assessing the quality of retrieval results.
The basic evaluation function `cbrkit.eval.compute` expects the following arguments:

- `qrels`: Ground truth relevance scores for query-case pairs.
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

After starting the server, you can access the API documentation at `http://localhost:8000/docs`.
