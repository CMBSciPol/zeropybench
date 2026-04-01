"""Measure the benchmarker overhead."""

import statistics
import timeit
from typing import Any

from zeropybench import Benchmark


def measure(
    name: str, func, globals: dict[str, Any] | None = None, number: int = 1_000, repeat: int = 7
) -> tuple[float, float]:
    """Elapsed time per call in ns."""
    times = timeit.repeat(func, number=number, repeat=repeat, globals=globals)
    times_ns = [_ / number * 1e9 for _ in times]
    median_ns = statistics.median(times_ns)
    mad = statistics.median(abs(_ - median_ns) for _ in times_ns)
    stdev = 1.4826 * mad
    print(f'     {name}: {median_ns:.3f} ns ± {stdev:.2f}')
    return median_ns, stdev


bench = Benchmark(repeat=20)

measure('timeit=pass', 'pass', repeat=100_000, globals={})
with bench(zeropybench='pass'):
    pass


def func():
    pass


measure('timeit=function frame', 'func()', repeat=100_000, globals={'func': func})
with bench(zeropybench='function frame'):
    func()


def func():
    return None


measure('timeit=function None', func, repeat=100_000, globals={'func': func})
with bench(zeropybench='function None'):
    func()


def func():
    return 1 + 1


measure('timeit=trivial function', 'func()', repeat=100_000, globals={'func': func})
with bench(zeropybench='trivial function'):
    func()

measure('timeit=simple operation', 'sum(range(100))', repeat=10_000, globals={})
with bench(zeropybench='simple operation'):
    sum(range(100))
