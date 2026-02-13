from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, override

from ..typing import Casebase, IndexableFunc, MapAdaptationFunc

type KeyFunc[K, V] = Callable[[Casebase[K, V]], K]

__all__ = [
    "KeyFunc",
    "indexable",
    "static",
]


@dataclass(slots=True, frozen=True)
class static[K, V](MapAdaptationFunc[K, V]):
    """Storage function that generates keys from a fixed casebase.

    Generates keys via ``key_func`` using the provided ``casebase``
    instead of the pipeline casebase, avoiding key collisions with the
    full collection.
    Each value in the pipeline casebase is added with a new key.
    Unlike ``indexable``, this does not maintain any internal state.

    Args:
        key_func: Callable that generates a new key given a casebase.
        casebase: The full casebase used for key generation.

    Examples:
        >>> func = static(
        ...     key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ...     casebase={0: "a", 1: "b", 2: "c"},
        ... )
        >>> result = func({1: "b"}, "d")
        >>> result == {0: "a", 1: "b", 2: "c", 3: "b"}
        True
    """

    key_func: KeyFunc[K, V]
    casebase: Casebase[K, V] = field(repr=False)

    @override
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        /,
    ) -> Casebase[K, V]:
        combined = dict(self.casebase)

        for value in casebase.values():
            new_key = self.key_func(combined)
            combined[new_key] = value

        return combined


@dataclass(slots=True, frozen=True)
class indexable[K, V](MapAdaptationFunc[K, V]):
    """Storage function that keeps an IndexableFunc's index in sync.

    Adds the pipeline casebase values to the index with new keys
    generated via ``key_func``, then rebuilds the index via
    ``create_index``.

    Keys are generated from the current index contents (or an empty
    dict when no index exists yet) to avoid collisions with existing
    entries.

    Note:
        When combined with ``dropout``, the index is rebuilt before
        dropout filters cases.
        The index will be corrected on the next retain call.

    Args:
        key_func: Callable that generates a new key given a casebase.
        indexable_func: The indexable function whose index will be kept in sync.

    Examples:
        >>> from cbrkit.retain.storage import indexable
        >>> class Store:
        ...     def __init__(self):
        ...         self._data = {}
        ...     @property
        ...     def index(self):
        ...         return self._data
        ...     def create_index(self, data):
        ...         self._data = dict(data)
        ...     def update_index(self, data):
        ...         self._data.update(data)
        ...     def delete_index(self, data):
        ...         for k in data:
        ...             self._data.pop(k, None)
        >>> store = Store()
        >>> func = indexable(
        ...     key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ...     indexable_func=store,
        ... )
        >>> result = func({0: "a", 1: "b"}, "c")
        >>> result == {0: "a", 1: "b"}
        True
        >>> store.index == result
        True

        Pre-populated index routes storage through the full collection:

        >>> store.create_index({0: "a", 1: "b", 2: "c"})
        >>> result = func({1: "b"}, "d")
        >>> result == {0: "a", 1: "b", 2: "c", 3: "b"}
        True
        >>> store.index == result
        True
    """

    key_func: KeyFunc[K, V]
    indexable_func: IndexableFunc[Casebase[K, V], Any]

    @override
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        /,
    ) -> Casebase[K, V]:
        combined = dict(self.indexable_func.index or {})

        for value in casebase.values():
            new_key = self.key_func(combined)
            combined[new_key] = value

        self.indexable_func.create_index(combined)

        return combined
