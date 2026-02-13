from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Literal, override

from ..helpers import (
    get_logger,
    get_value,
    mp_starmap,
    optional_dependencies,
    sim_map2ranking,
    unpack_float,
)
from ..sim.aggregator import default_aggregator
from ..typing import (
    AggregatorFunc,
    Casebase,
    ConversionFunc,
    Float,
    IndexableFunc,
    RetrieverFunc,
    SimMap,
    StructuredValue,
)
from .indexable import resolve_casebases

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class dropout[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Filters the retrieved cases based on the similarity values.

    Args:
        retriever_func: The retriever function to be used.
            Typically constructed with the `build` function.
        limit: The maximum number of cases to be returned.
        min_similarity: The minimum similarity value to be considered.
        max_similarity: The maximum similarity value to be considered.

    Returns:
        A retriever function that filters the retrieved cases based on the similarity values.
    """

    retriever_func: RetrieverFunc[K, V, S]
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        return [
            (casebase, self._call_single(sim_map))
            for casebase, sim_map in self.retriever_func(batches)
        ]

    def _call_single(self, similarities: SimMap[K, S]) -> SimMap[K, S]:
        ranking = sim_map2ranking(similarities)

        if self.min_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_float(similarities[key]) >= self.min_similarity
            ]
        if self.max_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_float(similarities[key]) <= self.max_similarity
            ]
        if self.limit is not None:
            ranking = ranking[: self.limit]

        return {key: similarities[key] for key in ranking}


@dataclass(slots=True, frozen=True)
class transpose[K, V1, V2, S: Float](RetrieverFunc[K, V1, S]):
    """Transforms a retriever function from one type to another.

    Useful when the input values need to be converted before retrieval,
    for instance, when the cases are nested and you only need to compare a subset of the values.
    This wrapper is not compatible with indexed retrieval mode (empty casebase inputs).
    If the inner retriever resolves an indexed casebase, values are in the converted type ``V2``,
    so transpose cannot safely reconstruct original ``V1`` values for the returned casebase.
    Use transpose with non-empty casebases at call time.

    Args:
        conversion_func: A function that converts the input values from one type to another.
        retriever_func: The retriever function to be used on the converted values.

    Returns:
        A retriever function that works on the converted values
    """

    retriever_func: RetrieverFunc[K, V2, S]
    conversion_func: ConversionFunc[V1, V2]

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V1], V1]]
    ) -> Sequence[tuple[Casebase[K, V1], SimMap[K, S]]]:
        if any(len(casebase) == 0 for casebase, _ in batches):
            raise ValueError(
                "transpose does not support indexed retrieval mode with empty casebases. "
                "Pass a non-empty casebase to transpose."
            )

        inner_results = self.retriever_func(
            [
                (
                    {
                        key: self.conversion_func(value)
                        for key, value in casebase.items()
                    },
                    self.conversion_func(query),
                )
                for casebase, query in batches
            ]
        )

        return [
            (original_casebase, sim_map)
            for (original_casebase, _), (_, sim_map) in zip(
                batches, inner_results, strict=True
            )
        ]


def transpose_value[K, V, S: Float](
    retriever_func: RetrieverFunc[K, V, S],
) -> RetrieverFunc[K, StructuredValue[V], S]:
    return transpose(retriever_func, get_value)


@dataclass(slots=True, frozen=True)
class combine[K, V, S: Float](RetrieverFunc[K, V, float]):
    """Combines multiple retriever functions into one.

    Args:
        retriever_funcs: A list of retriever functions to be combined.
        aggregator: A function to aggregate the results from the retriever functions.
        strategy: The strategy to combine the results. Either "intersection" or "union".
        default_sim: The default similarity value to use for strategy "union" when a case is not found in one of the retriever results.

    Returns:
        A retriever function that combines the results from multiple retrievers.

    Notes:
        When called with a non-empty input casebase, combine returns that original casebase unchanged.
        When called with an empty casebase (indexed retrieval mode), combine builds the result casebase
        from retriever outputs and only includes keys that appear in the aggregated similarity map.
        If multiple retrievers return different values for the same key, combine keeps the last value
        encountered in retriever order (last one wins).
    """

    retriever_funcs: (
        Sequence[RetrieverFunc[K, V, S]] | Mapping[str, RetrieverFunc[K, V, S]]
    )
    aggregator: AggregatorFunc[str, S | float] = default_aggregator
    strategy: Literal["intersection", "union"] = "union"
    default_sim: float = 0.0

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, float]]]:
        if isinstance(self.retriever_funcs, Sequence):
            func_results = [
                retriever_func(batches) for retriever_func in self.retriever_funcs
            ]
            results: list[tuple[Casebase[K, V], SimMap[K, float]]] = []

            for batch_idx in range(len(batches)):
                combined_sim = self.__call_batch__(
                    [batch_results[batch_idx][1] for batch_results in func_results]
                )
                result_casebase = self._resolve_batch_casebase(
                    batches[batch_idx][0],
                    combined_sim,
                    [batch_results[batch_idx][0] for batch_results in func_results],
                )
                results.append((result_casebase, combined_sim))

            return results

        elif isinstance(self.retriever_funcs, Mapping):
            named_results = {
                func_key: retriever_func(batches)
                for func_key, retriever_func in self.retriever_funcs.items()
            }
            results = []

            for batch_idx in range(len(batches)):
                combined_sim = self.__call_batch__(
                    {
                        func_key: func_results[batch_idx][1]
                        for func_key, func_results in named_results.items()
                    }
                )
                result_casebase = self._resolve_batch_casebase(
                    batches[batch_idx][0],
                    combined_sim,
                    [
                        func_results[batch_idx][0]
                        for func_results in named_results.values()
                    ],
                )
                results.append((result_casebase, combined_sim))

            return results

        raise ValueError(f"Invalid retriever_funcs type: {type(self.retriever_funcs)}")

    def _resolve_batch_casebase(
        self,
        original_casebase: Casebase[K, V],
        combined_sim: SimMap[K, float],
        candidate_casebases: Sequence[Casebase[K, V]],
    ) -> Casebase[K, V]:
        if len(original_casebase) > 0:
            return original_casebase

        return {
            key: value
            for candidate_casebase in candidate_casebases
            for key, value in candidate_casebase.items()
            if key in combined_sim
        }

    def __call_batch__(
        self, results: Sequence[SimMap[K, S]] | Mapping[str, SimMap[K, S]]
    ) -> SimMap[K, float]:
        case_keys: set[K]

        if isinstance(results, Sequence):
            if self.strategy == "intersection":
                case_keys = set().intersection(*(result.keys() for result in results))
            elif self.strategy == "union":
                case_keys = set().union(*(result.keys() for result in results))
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            return {
                case_key: self.aggregator(
                    [result.get(case_key, self.default_sim) for result in results]
                )
                for case_key in case_keys
            }

        elif isinstance(results, Mapping):
            if self.strategy == "intersection":
                case_keys = set().intersection(
                    *(result.keys() for result in results.values())
                )
            elif self.strategy == "union":
                case_keys = set().union(*(result.keys() for result in results.values()))
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            return {
                case_key: self.aggregator(
                    {
                        func_key: result.get(case_key, self.default_sim)
                        for func_key, result in results.items()
                    }
                )
                for case_key in case_keys
            }

        raise ValueError(f"Invalid results type: {type(results)}")


@dataclass(slots=True, frozen=True)
class distribute[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Distributes the retrieval process by passing each batch separately to the retriever function.

    Args:
        retriever_func: The retriever function to be used.
            Typically constructed with the `build` function.
        multiprocessing: Either a boolean to enable multiprocessing with all cores
            or an integer to specify the number of processes to use or a multiprocessing.Pool object.

    Returns:
        A retriever function that distributes the retrieval process.
    """

    retriever_func: RetrieverFunc[K, V, S]
    multiprocessing: Pool | int | bool

    def __call_batch__(
        self, x: Casebase[K, V], y: V
    ) -> tuple[Casebase[K, V], SimMap[K, S]]:
        return self.retriever_func([(x, y)])[0]

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        return mp_starmap(self.__call_batch__, batches, self.multiprocessing, logger)


@dataclass(slots=True)
class stateful[K, V, S: Float](
    RetrieverFunc[K, V, S],
    IndexableFunc[Casebase[K, V], Collection[K], Collection[V]],
):
    """Retriever wrapper that adds indexable support via a reference casebase.

    Wraps a non-indexable retriever and holds a reference to a full
    casebase.  When called with an empty casebase, the stored reference
    is used instead (indexed retrieval mode).  The reference can be
    managed through the ``IndexableFunc`` methods.

    Args:
        retriever_func: The inner retriever function.
        casebase: The initial reference casebase.

    Examples:
        >>> from cbrkit.retrieval import build, stateful
        >>> from cbrkit.sim.generic import equality
        >>> cb = {0: "a", 1: "b", 2: "c"}
        >>> retriever = stateful(
        ...     retriever_func=build(equality()),
        ...     casebase=cb,
        ... )
        >>> results = retriever([(dict(), "a")])
        >>> sorted(results[0][1].keys())
        [0, 1, 2]
    """

    retriever_func: RetrieverFunc[K, V, S]
    casebase: Casebase[K, V] = field(repr=False)

    _casebase: dict[K, V] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._casebase = dict(self.casebase)

    @property
    @override
    def index(self) -> Casebase[K, V]:
        """Return the reference casebase."""
        return self._casebase

    @property
    @override
    def keys(self) -> Collection[K]:
        """Return the reference casebase keys."""
        return list(self._casebase.keys())

    @property
    @override
    def values(self) -> Collection[V]:
        """Return the reference casebase values."""
        return list(self._casebase.values())

    @override
    def create_index(self, data: Casebase[K, V]) -> None:
        """Replace the reference casebase."""
        self._casebase = dict(data)

    @override
    def update_index(self, data: Casebase[K, V]) -> None:
        """Merge entries into the reference casebase."""
        self._casebase.update(data)

    @override
    def delete_index(self, data: Collection[K]) -> None:
        """Remove entries from the reference casebase."""
        for key in data:
            self._casebase.pop(key, None)

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        resolved = resolve_casebases(batches, self._casebase)
        return self.retriever_func(resolved)


with optional_dependencies():
    from chonkie.chunker import BaseChunker

    @dataclass(slots=True, frozen=True)
    class chunk[S: Float](RetrieverFunc[str, str, S]):
        """Chunks string cases using the chonkie library before retrieval.

        This retriever is special in that it returns a different set of cases for each batch
        it processes, as it splits the original string cases into chunks.

        Args:
            retriever_func: The retriever function to be used on the chunked strings.
            chunker: A BaseChunker instance from the chonkie library.

        Returns:
            A retriever function that chunks string cases and retrieves from the chunks.
        """

        retriever_func: RetrieverFunc[str, str, S]
        chunker: BaseChunker

        @override
        def __call__(
            self, batches: Sequence[tuple[Casebase[str, str], str]]
        ) -> Sequence[tuple[Casebase[str, str], SimMap[str, S]]]:
            chunked_batches: list[tuple[Casebase[str, str], str]] = []

            for casebase, query in batches:
                chunked_casebase: dict[str, str] = {}

                for case_key, case_text in casebase.items():
                    chunks = self.chunker.chunk(case_text)

                    for i, chunk in enumerate(chunks):
                        chunk_key = f"{case_key}-chunk{i}"
                        chunked_casebase[chunk_key] = (
                            chunk if isinstance(chunk, str) else chunk.text
                        )

                chunked_batches.append((chunked_casebase, query))

            return self.retriever_func(chunked_batches)
