import pickle
from collections.abc import MutableMapping

from cbrkit.helpers import frozendict


def test_frozendict() -> None:
    d = frozendict({"a": 1, "b": 2})

    assert d == {"b": 2, "a": 1}
    assert hash(d) == hash(frozendict(b=2, a=1))
    assert pickle.loads(pickle.dumps(d)) == d
    assert d | {"a": 3} == frozendict(a=3, b=2)
    assert not isinstance(d, MutableMapping)
    assert not hasattr(d, "__setitem__")
