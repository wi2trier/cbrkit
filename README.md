<!-- markdownlint-disable MD033 MD041 -->
<h2><p align="center">CBRkit</p></h2>
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

---

# CBRkit

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

- `nlp`: Standalone NLP tools `levenshtein`, `nltk`, `openai`, and `spacy`
- `transformers`: NLP tools based on `pytorch` and `transformers`
- `cli`: Command Line Interface (CLI)
- `api`: REST API Server
- `all`: All of the above

## Usage

CBRkit allows the definition of similarity metrics through _composition_.
This means that you can easily build even complex similarities by mixing built-in and/or custom measures.
CBRkit also includes predefined aggregation functions.
To get started, we provide a [demo project](https://github.com/wi2trier/cbrkit-demo) that shows how to use the library in a real-world scenario.
The following modules are part of CBRkit:

- `loaders`: Functions for loading cases and queries.
- `sim`: Similarity generator functions for various data types (e.g., strings, numbers).
- `retrieval`: Functions for retrieving cases based on a query.
- `typing`: Generic type definitions for defining custom functions.

CBRkit is fully typed, so IDEs like VSCode and PyCharm can provide autocompletion and type checking.
We will explain all modules and their basic usage in the following sections.

## Loading Cases and Queries

The first step is to load cases and queries.
We provide predefined functions for the most common formats like CSV, JSON, and XML.
Additionally, CBRkit also integrates with `pandas` for loading data frames.
The following example shows how to load cases and queries from a CSV file using `pandas`:

```python
import pandas as pd
import cbrkit

df = pd.read_csv("path/to/cases.csv")
cases = cbrkit.loaders.dataframe(df)
```

When dealing with formats like JSON, the files can be loaded directly:

```python
cases = cbrkit.loaders.json("path/to/cases.json")
```

Queries can either be loaded using the same loader functions.
CBRkit expects the type of the queries to match the type of the cases.

```python
 # for pandas
queries = cbrkit.loaders.dataframe(pd.read_csv("path/to/queries.csv"))
# for json
queries = cbrkit.loaders.json("path/to/queries.json")
```

In case your query collection only contains a single entry, you can use the `singleton` function to extract it.

```python
query = cbrkit.helpers.singleton(queries)
```

Alternatively, you can also create a query directly in Python:

```python
# for pandas
query = pd.Series({"name": "John", "age": 25})
# for json
query = {"name": "John", "age": 25}
```

## Similarity Measures and Aggregation

The next step is to define similarity measures for the cases and queries.
It is possible to define custom measures, use built-in ones, or combine both.

### Custom Similarity Measures

In CBRkit, a similarity measure is defined as a function that takes two arguments (a case and a query) and returns a similarity score: `sim = f(x, y)`.
It also supports pipeline-based similarity measures that are popular in NLP where a list of tuples is passed to the similarity measure: `sims = f([(x1, y1), (x2, y2), ...])`.
This generic approach allows you to define custom similarity measures for your specific use case.
For instance, you may define the following function for comparing colors:

```python
def color_similarity(x: str, y: str) -> float:
    if x == y:
        return 1.0
    elif x in y or y in x:
        return 0.5

    return 0.0
```

In addition to checking for strict equality, our function also checks for partial matches (e.g., `x = "blue"` and `y = "light blue"`).

### Built-in Similarity Measures

CBRkit also contains a selection of built-in similarity measures for the most common data types in the module `cbrkit.sim`.
They are provided through _generator functions_ that allow you to customize the behavior of the built-in measures.
For example, an spacy-based embedding similarity measure can be obtained as follows:

```python
semantic_similarity = cbrkit.sim.strings.spacy(model_name="en_core_web_lg")
```

_Please note:_ Calling the function `cbrkit.sim.strings.spacy` returns a similarity function itself that has the same signature as the `color_similarity` function defined above.

An overview of all available similarity measures can be found in the [module documentation](https://wi2trier.github.io/cbrkit/cbrkit/sim.html).

### Global Similarity and Aggregation

When dealing with cases that are not represented through elementary data types like strings, we need to aggregate individual measures to obtain a global similarity score.
When defining similarity measures from scratch, you may still use the built-in `aggregator` to combine the individual similarity scores:

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
        "color": color_similarity
        ...
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
),
```

The `attribute_value` function lets you define measures for each attribute of the cases/queries as well as the aggregation function.
It also allows to use custom measures like the `color_similarity` function defined above.
