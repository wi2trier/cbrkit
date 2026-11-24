import asyncio
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import cbrkit
from cbrkit.helpers import (
    batched,
    batchify,
    bounded_gather,
    is_batched,
    lazy,
    to_thread,
    unbatchify,
    wait,
)


def test_batch_marking_ignores_signatures() -> None:
    # Counting parameters would take both for batch functions.
    def single(*values: Any) -> int:
        return sum(values)

    @batched
    def batch(batches: Any, scale: int = 1) -> list[int]:
        return [scale * sum(pair) for pair in batches]

    assert batchify(single)([(1, 2), (3, 4)]) == [3, 7]
    assert batchify(batch) is batch
    assert unbatchify(batch)(1, 2) == 3
    assert unbatchify(single) is single


def test_lazy_builds_once_and_pickles_the_factory():
    sim = lazy(
        lambda: cbrkit.sim.AttributeValue(attributes={"a": lambda case, query: 1.0})
    )
    assert is_batched(sim)
    assert sim([({"a": 1}, {"a": 2})])[0].value == 1.0

    restored = pickle.loads(pickle.dumps(lazy(cbrkit.sim.generic.Equality)))
    assert restored("x", "x") == 1.0


def test_lazy_builds_once_across_threads() -> None:
    builds: list[None] = []

    def factory() -> Any:
        builds.append(None)
        time.sleep(0.05)

        return str.upper

    func = lazy(factory)

    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(func, "ab" * 4)) == list("AB" * 4)

    assert len(builds) == 1


def test_bounded_gather_refuses_a_limit_below_one() -> None:
    async def double(item: int) -> int:
        return 2 * item

    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(bounded_gather([1], double, limit=0))


def test_to_thread_never_starves_the_default_executor() -> None:
    async def embed(value: int) -> int:
        return await asyncio.to_thread(lambda: value)

    def measure(value: int) -> int:
        return wait(embed(value))

    async def run() -> list[int]:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(2))

        return await asyncio.wait_for(
            asyncio.gather(*(to_thread(measure, value) for value in range(4))), 10
        )

    assert asyncio.run(run()) == [0, 1, 2, 3]
