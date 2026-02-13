from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any, override

from ..typing import Casebase, IndexableFunc, MapAdaptationFunc

type KeyFunc[K] = Callable[[Collection[K]], K]

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
        key_func: Callable that generates a new key given a collection of keys.
        casebase: The full casebase used for key generation.

    Examples:
        >>> func = static(
        ...     key_func=lambda keys: max(keys, default=-1) + 1,
        ...     casebase={0: "a", 1: "b", 2: "c"},
        ... )
        >>> result = func({1: "b"}, "d")
        >>> result == {0: "a", 1: "b", 2: "c", 3: "b"}
        True
    """

    key_func: KeyFunc[K]
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
            new_key = self.key_func(combined.keys())
            combined[new_key] = value

        return combined


@dataclass(slots=True, frozen=True)
class indexable[K, V](MapAdaptationFunc[K, V]):
    """Storage function that keeps an IndexableFunc's index in sync.

    Adds the pipeline casebase values to the index with new keys
    generated via ``key_func``, then updates the index via
    ``update_index``.

    Only the keys are loaded from the index for key generation,
    and only new entries are persisted via ``update_index``,
    avoiding a full database round-trip.

    Note:
        When combined with ``dropout``, the index is updated before
        dropout filters cases.
        The index will be corrected on the next retain call.

    Args:
        key_func: Callable that generates a new key given a collection of keys.
        indexable_func: The indexable function whose index will be kept in sync.

    Examples:
        >>> from cbrkit.retain.storage import indexable
        >>> class Store:
        ...     def __init__(self):
        ...         self._data = {}
        ...     @property
        ...     def index(self):
        ...         return self._data
        ...     @property
        ...     def keys(self):
        ...         return self._data.keys()
        ...     @property
        ...     def values(self):
        ...         return self._data.values()
        ...     def create_index(self, data):
        ...         self._data = dict(data)
        ...     def update_index(self, data):
        ...         self._data.update(data)
        ...     def delete_index(self, data):
        ...         for k in data:
        ...             self._data.pop(k, None)
        >>> store = Store()
        >>> func = indexable(
        ...     key_func=lambda keys: max(keys, default=-1) + 1,
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

    key_func: KeyFunc[K]
    indexable_func: IndexableFunc[Casebase[K, V], Any, Any]

    @override
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        /,
    ) -> Casebase[K, V]:
        existing_keys = set(self.indexable_func.keys)
        new_entries: dict[K, V] = {}

        for value in casebase.values():
            new_key = self.key_func(existing_keys)
            existing_keys.add(new_key)
            new_entries[new_key] = value

        if new_entries:
            self.indexable_func.update_index(new_entries)

        return self.indexable_func.index
