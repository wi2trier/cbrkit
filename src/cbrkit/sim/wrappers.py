from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, cast, override

from ..helpers import batchify_sim, get_metadata, get_value, getitem_or_getattr
from ..typing import (
    AggregatorFunc,
    AnySimFunc,
    BatchSimFunc,
    ConversionFunc,
    Float,
    HasMetadata,
    JsonDict,
    SimSeq,
    StructuredValue,
)
from .aggregator import default_aggregator
from .generic import static


@dataclass(slots=True)
class transpose[V1, V2, S: Float](BatchSimFunc[V1, S]):
    """Transforms a similarity function from one type to another.

    Args:
        similarity_func: The similarity function to be used on the converted values.
        conversion_func: A function that converts the input values from one type to another.

    Examples:
        >>> from cbrkit.sim.generic import equality
        >>> sim = transpose(
        ...     similarity_func=equality(),
        ...     conversion_func=lambda x: x.lower(),
        ... )
        >>> sim([("A", "a"), ("b", "B")])
        [1.0, 1.0]
    """

    similarity_func: BatchSimFunc[V2, S]
    conversion_func: ConversionFunc[V1, V2]

    def __init__(
        self,
        similarity_func: AnySimFunc[V2, S],
        conversion_func: ConversionFunc[V1, V2],
    ):
        self.similarity_func = batchify_sim(similarity_func)
        self.conversion_func = conversion_func

    @override
    def __call__(self, batches: Sequence[tuple[V1, V1]]) -> Sequence[S]:
        return self.similarity_func(
            [(self.conversion_func(x), self.conversion_func(y)) for x, y in batches]
        )


def transpose_value[V, S: Float](
    func: AnySimFunc[V, S],
) -> BatchSimFunc[StructuredValue[V], S]:
    return transpose(func, get_value)


@dataclass(slots=True)
class combine[V, S: Float](BatchSimFunc[V, float]):
    """Combines multiple similarity functions into one.

    Args:
        sim_funcs: A list of similarity functions to be combined.
        aggregator: A function to aggregate the results from the similarity functions.

    Returns:
        A similarity function that combines the results from multiple similarity functions.
    """

    sim_funcs: InitVar[Sequence[AnySimFunc[V, S]] | Mapping[str, AnySimFunc[V, S]]]
    aggregator: AggregatorFunc[str, S] = default_aggregator
    batch_sim_funcs: Sequence[BatchSimFunc[V, S]] | Mapping[str, BatchSimFunc[V, S]] = (
        field(init=False, repr=False)
    )

    def __post_init__(
        self, sim_funcs: Sequence[AnySimFunc[V, S]] | Mapping[str, AnySimFunc[V, S]]
    ):
        if isinstance(sim_funcs, Sequence):
            self.batch_sim_funcs = [batchify_sim(func) for func in sim_funcs]
        elif isinstance(sim_funcs, Mapping):
            self.batch_sim_funcs = {
                key: batchify_sim(func) for key, func in sim_funcs.items()
            }
        else:
            raise ValueError(f"Invalid sim_funcs type: {type(sim_funcs)}")

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> Sequence[float]:
        if isinstance(self.batch_sim_funcs, Sequence):
            func_results = [func(batches) for func in self.batch_sim_funcs]

            return [
                self.aggregator(
                    [batch_results[batch_idx] for batch_results in func_results]
                )
                for batch_idx in range(len(batches))
            ]

        elif isinstance(self.batch_sim_funcs, Mapping):
            func_results = {
                func_key: func(batches)
                for func_key, func in self.batch_sim_funcs.items()
            }

            return [
                self.aggregator(
                    {
                        func_key: batch_results[batch_idx]
                        for func_key, batch_results in func_results.items()
                    }
                )
                for batch_idx in range(len(batches))
            ]

        raise ValueError(f"Invalid batch_sim_funcs type: {type(self.batch_sim_funcs)}")


@dataclass(slots=True)
class cache[V, U, S: Float](BatchSimFunc[V, S]):
    similarity_func: BatchSimFunc[V, S]
    conversion_func: ConversionFunc[V, U] | None
    store: MutableMapping[tuple[U, U], S] = field(repr=False)

    def __init__(
        self,
        similarity_func: AnySimFunc[V, S],
        conversion_func: ConversionFunc[V, U] | None = None,
    ):
        self.similarity_func = batchify_sim(similarity_func)
        self.conversion_func = conversion_func
        self.store = {}

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[S]:
        transformed_batches = (
            [(self.conversion_func(x), self.conversion_func(y)) for x, y in batches]
            if self.conversion_func is not None
            else cast(list[tuple[U, U]], batches)
        )
        uncached_indexes = [
            idx
            for idx, pair in enumerate(transformed_batches)
            if pair not in self.store
        ]

        uncached_sims = self.similarity_func([batches[idx] for idx in uncached_indexes])
        self.store.update(
            {
                transformed_batches[idx]: sim
                for idx, sim in zip(uncached_indexes, uncached_sims, strict=True)
            }
        )

        return [self.store[pair] for pair in transformed_batches]


@dataclass(slots=True)
class dynamic_table[K, U, V, S: Float](BatchSimFunc[U | V, S], HasMetadata):
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for pairs not in the table
        key_getter: A function that extracts the the key for lookup from the input values

    Examples:
        >>> from cbrkit.helpers import identity
        >>> from cbrkit.sim.generic import static
        >>> sim = dynamic_table(
        ...     {
        ...         ("a", "b"): static(0.5),
        ...         ("b", "c"): static(0.7)
        ...     },
        ...     symmetric=True,
        ...     default=static(0.0),
        ...     key_getter=identity,
        ... )
        >>> sim([("b", "a"), ("a", "c")])
        [0.5, 0.0]
    """

    symmetric: bool
    default: BatchSimFunc[U, S] | None
    key_getter: Callable[[Any], K]
    table: dict[tuple[K, K], BatchSimFunc[V, S]]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "symmetric": self.symmetric,
            "default": get_metadata(self.default),
            "key_getter": get_metadata(self.key_getter),
            "table": [
                {
                    "x": str(k[0]),
                    "y": str(k[1]),
                    "value": get_metadata(v),
                }
                for k, v in self.table.items()
            ],
        }

    def __init__(
        self,
        entries: Mapping[tuple[K, K], AnySimFunc[..., S]]
        | Mapping[K, AnySimFunc[..., S]],
        key_getter: Callable[[Any], K],
        default: AnySimFunc[U, S] | S | None = None,
        symmetric: bool = True,
    ):
        self.symmetric = symmetric
        self.key_getter = key_getter
        self.table = {}

        if isinstance(default, Callable):
            self.default = batchify_sim(default)
        elif default is None:
            self.default = None
        else:
            self.default = batchify_sim(static(default))

        for key, val in entries.items():
            func = batchify_sim(val)

            if isinstance(key, tuple):
                x, y = cast(tuple[K, K], key)
            else:
                x = y = cast(K, key)

            self.table[(x, y)] = func

            if self.symmetric and x != y:
                self.table[(y, x)] = func

    @override
    def __call__(self, batches: Sequence[tuple[U | V, U | V]]) -> SimSeq[S]:
        # then we group the batches by key to avoid redundant computations
        idx_map: defaultdict[tuple[K, K] | None, list[int]] = defaultdict(list)

        for idx, (x, y) in enumerate(batches):
            key = (self.key_getter(x), self.key_getter(y))

            if key in self.table:
                idx_map[key].append(idx)
            else:
                idx_map[None].append(idx)

        # now we compute the similarities
        results: dict[int, S] = {}

        for key, idxs in idx_map.items():
            sim_func = cast(
                BatchSimFunc[U | V, S] | None,
                self.table.get(key) if key is not None else self.default,
            )

            if sim_func is None:
                missing_entries = [batches[idx] for idx in idxs]
                missing_keys = {
                    (self.key_getter(x), self.key_getter(y)) for x, y in missing_entries
                }

                raise ValueError(f"Pairs {missing_keys} not in the table")

            sims = sim_func([batches[idx] for idx in idxs])

            for idx, sim in zip(idxs, sims, strict=True):
                results[idx] = sim

        return [results[idx] for idx in range(len(batches))]


table = dynamic_table


def type_table[U, V, S: Float](
    entries: Mapping[type[V], AnySimFunc[..., S]],
    default: AnySimFunc[U, S] | S | None = None,
) -> BatchSimFunc[U | V, S]:
    return dynamic_table(
        entries=entries,
        key_getter=type,
        default=default,
        symmetric=False,
    )


@dataclass(slots=True, frozen=True)
class attribute_table_key_getter[K]:
    func: Callable[[Any, str], K]
    attribute: str

    def __call__(self, x: Any) -> K:
        return self.func(x, self.attribute)


def attribute_table[K, U, S: Float](
    entries: Mapping[K, AnySimFunc[..., S]],
    attribute: str,
    default: AnySimFunc[U, S] | S | None = None,
    value_getter: Callable[[Any, str], K] = getitem_or_getattr,
) -> BatchSimFunc[Any, S]:
    key_getter = attribute_table_key_getter(value_getter, attribute)

    return dynamic_table(
        entries=entries,
        key_getter=key_getter,
        default=default,
        symmetric=False,
    )
