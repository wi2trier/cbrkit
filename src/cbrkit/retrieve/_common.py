"""Shared helpers of the retrievers that score many queries against one casebase at once."""

from collections.abc import Mapping, Sequence

__all__ = ["query_groups"]


def query_groups[K, V](
    batches: Sequence[tuple[Mapping[K, V], V]],
) -> list[tuple[Mapping[K, V], list[V]]]:
    """Groups the queries of batches by casebase, so a retriever scores every group in one call.

    All queries form one group if every batch shares its casebase, otherwise every batch is a group,
    and the flattened results of the groups follow the order of the batches.

    Examples:
        >>> casebase = {"a": 1}
        >>> query_groups([(casebase, 1), (casebase, 2)])
        [({'a': 1}, [1, 2])]
        >>> query_groups([({}, 1), ({}, 2)])
        [({}, [1]), ({}, [2])]
    """
    if batches and all(casebase is batches[0][0] for casebase, _ in batches):
        return [(batches[0][0], [query for _, query in batches])]

    return [(casebase, [query]) for casebase, query in batches]
