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

> [!caution]
> The project is under active development and does not yet adhere to semantic versioning.
> Breaking changes may occur at any time for versions `0.x.y`.
> Once the project reaches version `1.0`, semantic versioning will be applied.

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

- `nlp`: Natural Language Processing (NLP), including `spacy`, `openai`, and `sentence-transformers`
- `cli`: Command Line Interface (CLI)
- `api`: REST API Server
- `all`: All of the above

## Usage

CBRkit allows the definition of similarity metrics through _composition_.
This means that you can easily build even complex similarities by mixing built-in and/or custom measures.
CBRkit also includes predefined aggregation functions.
A working retrieval example can be found as part of our [testing suite](https://github.com/wi2trier/cbrkit/tree/main/tests/test_retrieve.py).

The following modules are part of CBRkit:

- `sim`: Similarity generator functions for various data types (e.g., strings, numbers).
- `global_sim`: Similarity generator functions for aggregating the above ones.
- `retrieval`: Functions for retrieving cases based on a query.
- `typing`: Generic type definitions for defining custom functions.
