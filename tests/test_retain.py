import cbrkit
from cbrkit.helpers import unpack_float

from .conftest import FakeIndexable


def test_retain_build(small_casebase, simple_query, sim_func_simple):
    """Test retainer stores cases and produces real scores."""
    casebase = dict(small_casebase)

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    retainer = cbrkit.retain.build(
        assess_func=sim_func_simple,
        storage_func=storage_func,
    )

    results = retainer([(casebase, simple_query)])

    updated_casebase, sim_map = results[0]
    assert len(updated_casebase) == 4
    assert updated_casebase[2] == casebase[0]
    assert updated_casebase[3] == casebase[1]
    assert all(0.0 <= unpack_float(v) <= 1.0 for v in sim_map.values())


def test_retain_dropout(small_casebase, simple_query, sim_func_simple):
    """Test dropout keeps cases above min and rejects above max."""
    casebase = dict(small_casebase)

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    inner = cbrkit.retain.build(assess_func=sim_func_simple, storage_func=storage_func)

    # min_similarity=0.0: all cases kept
    kept = cbrkit.retain.dropout(inner, min_similarity=0.0)([(casebase, simple_query)])
    assert len(kept[0][0]) == 4

    # min_similarity=1.1: impossible threshold, new cases are rejected
    rejected = cbrkit.retain.dropout(inner, min_similarity=1.1)(
        [(casebase, simple_query)]
    )
    assert len(rejected[0][0]) == 2
    assert 2 not in rejected[0][0]
    assert 3 not in rejected[0][0]

    # max_similarity=0.0: all new cases exceed threshold, rejected
    novelty = cbrkit.retain.dropout(inner, max_similarity=0.0)(
        [(casebase, simple_query)]
    )
    assert len(novelty[0][0]) == 2


def test_retain_full_cycle(medium_casebase, simple_query, sim_func_simple):
    """Test retain phase within a full CBR cycle."""
    casebase = dict(medium_casebase)

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(sim_func_simple),
        limit=2,
    )

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=sim_func_simple,
    )

    reviser = cbrkit.revise.build(assess_func=sim_func_simple)

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    retainer = cbrkit.retain.build(
        assess_func=sim_func_simple,
        storage_func=storage_func,
    )

    result = cbrkit.cycle.apply_query(
        casebase,
        simple_query,
        retriever,
        reuser,
        reviser,
        retainer,
    )

    assert len(result.retrieval.casebase) == 2
    assert len(result.retain.casebase) > 0


def test_retain_indexable_storage():
    """Test indexable storage stores cases and syncs the index."""
    fake = FakeIndexable()
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
    fake = FakeIndexable()
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
