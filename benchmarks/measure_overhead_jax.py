"""Measure the benchmarker overhead."""

import statistics
import timeit
from typing import Any

import jax
import jax.numpy as jnp

from zeropybench import Benchmark

VERBOSE = False
REPEAT = 20


def measure(
    name: str,
    code: str,
    *,
    number: int,
    globals: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Elapsed time per call in ns."""

    compilation_and_exec_time = timeit.timeit(code, number=1, globals=globals) * 1e6
    raw_times = timeit.repeat(code, number=number, repeat=REPEAT, globals=globals)
    times = [_ / number * 1e6 for _ in raw_times]
    median_time = statistics.median(times)
    mad = statistics.median(abs(_ - median_time) for _ in times)
    stdev = 1.4826 * mad
    print(
        f'     {name}: {median_time:.3f} us ± {stdev:.2f}, compilation: {compilation_and_exec_time - median_time:8.2f} us'
    )
    return median_time, stdev


# Initialize JAX
a = jnp.ones(10)
b = jnp.ones(10)

bench = Benchmark(repeat=REPEAT, verbose=VERBOSE)


def func(a):
    return a, a


measure(
    'timeit=function',
    'jax.block_until_ready(func(a))',
    number=50_000,
    globals={'func': jax.jit(func), 'a': a, 'jax': jax},
)
with bench(zeropybench='function'):
    func(a)
print()


def func(a, b):
    return a + b


measure(
    'timeit=addition',
    'func(a, b).block_until_ready()',
    number=100_000,
    globals={'func': jax.jit(func), 'a': a, 'b': b},
)
with bench(zeropybench='addition'):
    func(a, b)
print()


def func(a):
    return a + jnp.sum(a) + jnp.sum(jnp.arange(100))


measure(
    'timeit=simple operation',
    'func(a).block_until_ready()',
    number=100_000,
    globals={'func': jax.jit(func), 'a': a},
)
with bench(zeropybench='simple operation'):
    a + jnp.sum(a) + jnp.sum(jnp.arange(100))
print()
