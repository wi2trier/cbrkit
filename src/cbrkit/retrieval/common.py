from collections.abc import Sequence

from ..typing import Casebase


def resolve_casebases[K, V](
    batches: Sequence[tuple[Casebase[K, V], V]],
    indexed_casebase: Casebase[K, V] | None,
) -> list[tuple[Casebase[K, V], V]]:
    """Resolve casebases for indexable retrievers.

    Empty casebases are treated as an explicit signal to use indexed retrieval mode.
    In indexed mode, empty casebases are replaced with the previously indexed casebase.
    """
    if indexed_casebase is None:
        if any(len(casebase) == 0 for casebase, _ in batches):
            raise ValueError(
                "Indexed retrieval was requested with an empty casebase, but no index is available. "
                "Call create_index() first."
            )

        return list(batches)

    return [
        (indexed_casebase if len(casebase) == 0 else casebase, query)
        for casebase, query in batches
    ]
