from collections.abc import Collection, Mapping

import cbrkit
from cbrkit.helpers import unpack_float

SIM_FUNC = cbrkit.sim.attribute_value(
    attributes={
        "price": cbrkit.sim.numbers.linear(max=100000),
        "year": cbrkit.sim.numbers.linear(max=50),
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)


def test_retain_build():
    """Test retainer stores cases and produces real scores."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }
    query = {"price": 10000, "year": 2010}

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    retainer = cbrkit.retain.build(
        assess_func=SIM_FUNC,
        storage_func=storage_func,
    )

    results = retainer([(casebase, query)])

    updated_casebase, sim_map = results[0]
    assert len(updated_casebase) == 4
    assert updated_casebase[2] == casebase[0]
    assert updated_casebase[3] == casebase[1]
    assert all(0.0 <= unpack_float(v) <= 1.0 for v in sim_map.values())


def test_retain_dropout():
    """Test dropout keeps cases above min and rejects above max."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }
    query = {"price": 10000, "year": 2010}

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    inner = cbrkit.retain.build(assess_func=SIM_FUNC, storage_func=storage_func)

    # min_similarity=0.0: all cases kept
    kept = cbrkit.retain.dropout(inner, min_similarity=0.0)([(casebase, query)])
    assert len(kept[0][0]) == 4

    # min_similarity=1.1: impossible threshold, new cases are rejected
    rejected = cbrkit.retain.dropout(inner, min_similarity=1.1)([(casebase, query)])
    assert len(rejected[0][0]) == 2
    assert 2 not in rejected[0][0]
    assert 3 not in rejected[0][0]

    # max_similarity=0.0: all new cases exceed threshold, rejected
    novelty = cbrkit.retain.dropout(inner, max_similarity=0.0)([(casebase, query)])
    assert len(novelty[0][0]) == 2


def test_retain_full_cycle():
    """Test retain phase within a full CBR cycle."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
        2: {"price": 11000, "year": 2010},
    }
    query = {"price": 10000, "year": 2010}

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(SIM_FUNC),
        limit=2,
    )

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=SIM_FUNC,
    )

    reviser = cbrkit.revise.build(assess_func=SIM_FUNC)

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    retainer = cbrkit.retain.build(
        assess_func=SIM_FUNC,
        storage_func=storage_func,
    )

    result = cbrkit.cycle.apply_query(
        casebase,
        query,
        retriever,
        reuser,
        reviser,
        retainer,
    )

    assert len(result.retrieval.casebase) == 2
    assert len(result.retain.casebase) > 0


class FakeIndexableFunc(
    cbrkit.typing.IndexableFunc[Mapping[int, str], Collection[int], Collection[str]],
):
    def __init__(self) -> None:
        self._data: dict[int, str] | None = None

    @property
    def index(self) -> Mapping[int, str]:
        if self._data is None:
            return {}
        return self._data

    @property
    def keys(self) -> Collection[int]:
        if self._data is None:
            return []
        return self._data.keys()

    @property
    def values(self) -> Collection[str]:
        if self._data is None:
            return []
        return list(self._data.values())

    def create_index(self, data: Mapping[int, str]) -> None:
        self._data = dict(data)

    def update_index(self, data: Mapping[int, str]) -> None:
        if self._data is None:
            self.create_index(data)
            return
        self._data.update(data)

    def delete_index(self, data: Collection[int]) -> None:
        if self._data is None:
            return
        for key in data:
            self._data.pop(key, None)


def test_retain_indexable_storage():
    """Test indexable storage stores cases and syncs the index."""
    fake = FakeIndexableFunc()
    retainer = cbrkit.retain.build(
        assess_func=cbrkit.sim.generic.equality(),
        storage_func=cbrkit.retain.indexable(
            key_func=lambda keys: max(keys, default=-1) + 1,
            indexable_func=fake,
        ),
    )

    casebase: dict[int, str] = {0: "a", 1: "b"}
    results = retainer([(casebase, "c")])

    updated_casebase, sim_map = results[0]
    assert updated_casebase == {0: "a", 1: "b"}
    assert fake.index == updated_casebase


def test_retain_indexable_prepopulated():
    """Test indexable with pre-populated index stores into full collection."""
    fake = FakeIndexableFunc()
    fake.create_index({0: "a", 1: "b", 2: "c"})

    retainer = cbrkit.retain.build(
        assess_func=cbrkit.sim.generic.equality(),
        storage_func=cbrkit.retain.indexable(
            key_func=lambda keys: max(keys, default=-1) + 1,
            indexable_func=fake,
        ),
    )

    # Pipeline casebase is a filtered subset
    pipeline_casebase: dict[int, str] = {1: "b"}
    results = retainer([(pipeline_casebase, "d")])

    updated_casebase, sim_map = results[0]
    # Should return the full index with new entry
    assert updated_casebase == {0: "a", 1: "b", 2: "c", 3: "b"}
    assert fake.index == updated_casebase
