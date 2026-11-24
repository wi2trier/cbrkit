import inspect
from typing import Any

__all__ = [
    "class_docs",
    "describe",
    "parse_docstring",
]


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its description and its argument descriptions.

    Continuation lines of an argument are joined with spaces.

    Examples:
        >>> parse_docstring('''Linear similarity.
        ...
        ... Args:
        ...     max: Distance with a similarity of 0.
        ...     min (float): Distance with a similarity of 1,
        ...         defaults to 0.
        ...
        ... Examples:
        ...     >>> linear(1)
        ... ''')
        ('Linear similarity.', {'max': 'Distance with a similarity of 0.', 'min': 'Distance with a similarity of 1, defaults to 0.'})
    """
    from griffe import Docstring, DocstringSectionParameters, DocstringSectionText

    sections = Docstring(doc).parse("google", warnings=False)
    description = "\n\n".join(
        section.value
        for section in sections
        if isinstance(section, DocstringSectionText)
    )
    args = {
        param.name: " ".join(
            filter(None, map(str.strip, param.description.splitlines()))
        )
        for section in sections
        if isinstance(section, DocstringSectionParameters)
        for param in section.value
    }

    return description, args


def class_docs(cls: type[Any]) -> tuple[str, dict[str, str]]:
    """The description of a class and the descriptions of its arguments from its Google-style docstring.

    Arguments that a base class documents are inherited,
    so the fields of a base class are documented once.

    Examples:
        >>> class Base:
        ...     '''Base.
        ...
        ...     Args:
        ...         a: First.
        ...     '''
        >>> class Child(Base):
        ...     '''Child.
        ...
        ...     Args:
        ...         b: Second.
        ...     '''
        >>> class_docs(Child)
        ('Child.', {'a': 'First.', 'b': 'Second.'})
    """
    args: dict[str, str] = {}

    for base in reversed(cls.__mro__):
        if doc := base.__dict__.get("__doc__"):
            args |= parse_docstring(doc)[1]

    return parse_docstring(inspect.getdoc(cls) or "")[0], args


def describe(schema: dict[str, Any], cls: type[Any]) -> None:
    """Fills the descriptions of a JSON Schema from the Google-style docstrings of a class, see `class_docs`.

    Used as `json_schema_extra` of the classes of a `Family` family.
    """
    description, args = class_docs(cls)

    if description:
        schema["description"] = description

    for name, prop in schema.get("properties", {}).items():
        if name in args:
            prop["description"] = args[name]
