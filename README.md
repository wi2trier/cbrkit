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
- `global_sim`: Similarity generator functions for aggregating the above ones.
- `retrieval`: Functions for retrieving cases based on a query.
- `typing`: Generic type definitions for defining custom functions.

CBRkit is fully typed, so IDEs like VSCode and PyCharm can provide autocompletion and type checking.
We will explain all modules and their basic usage in the following sections.

### Loading Cases and Queries

The first step is to load cases and queries.
We provide predefined functions for the most common formats like CSV, JSON, and XML.
Additionally, `cbrkit` also integrates with `pandas` for loading data frames.
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

### Similarity Measures and Aggregation

In CBRkit, we differentiate between regular similarity measures and global similarity measures.
The former are used to compare individual attributes of cases and queries, while the latter are used to aggregate the results of the former.
If the cases are represented through elementary data types (e.g., each case is a plain-text document), no global similarity measure is needed.
However, if the cases are represented through a combination of elementary data types (e.g., each case is a person with a name and an age), a global similarity measure is needed to aggregate the results of the elementary similarity measures.

In CBRkit, a similarity measure is defined as a function that takes two arguments (a case and a query) and returns a similarity score: `sim = f(case, query)`.
It also supports pipeline-based similarity measures that are popular in NLP where a list of tuples is passed to the similarity measure: `sims = f([(case1, query1), (case2, query2), ...])`.
This generic approach allows you to define custom similarity measures for your specific use case.
To make the process easier, CBRkit provides a set of predefined similarity measures for the most common data types.
They are provided through generator functions that allow you to customize the behavior of the built-in measures.

#### Elementary Similarity Measures

The elementary measures are available in the module `cbrkit.sim`.
