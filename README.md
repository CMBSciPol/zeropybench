<div align="center">
<img src="https://raw.githubusercontent.com/CMBSciPol/zeropybench/main/docs/source/_static/logo.svg" alt="zeropybench" width="100%">
</div>


# Zero-overhead Python and JAX Reliable Benchmarking

[![PyPI version](https://img.shields.io/pypi/v/zeropybench?color=%232EBF4F)](https://pypi.org/project/zeropybench)
[![Python versions](https://img.shields.io/pypi/pyversions/zeropybench.svg?color=%232EBF4F)](https://pypi.org/project/zeropybench)
[![Continuous integration](https://github.com/CMBSciPol/zeropybench/actions/workflows/ci.yml/badge.svg)](https://github.com/CMBSciPol/zeropybench/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/CMBSciPol/zeropybench/graph/badge.svg)](https://codecov.io/gh/CMBSciPol/zeropybench)

<!-- Start common text with source/index.md -->

**ZeroPyBench** is a Python benchmarking library with zero overhead, designed for multidimensional performance analysis.

## Features

- **Context manager API**: Benchmark any code block with `with bench(...): ...`
- **Multidimensional**: Tag benchmarks with arbitrary keyword arguments
- **Zero overhead**: Code is passed directly to `timeit.Timer`, no wrapper function
- **Auto-scaling**: Automatically determines the number of iterations for reliable measurements
- **Multiple exports**: CSV, Parquet, Markdown
- **Plotting**: Built-in visualization with matplotlib

## Quick Example

```python
from zeropybench import Benchmark

bench = Benchmark()

for n in [100, 1000, 10000]:
    data = list(range(n))
    with bench(method='sum', n=n):
        sum(data)
    with bench(method='len', n=n):
        len(data)
```
Output:
```text
method=sum, n=100: 575.124 ns ± 3.35% (median of 7 runs, 500000 loops each)
method=len, n=100: 19.037 ns ± 0.85% (median of 7 runs, 20000000 loops each)
method=sum, n=1000: 2.961 µs ± 36.70% (median of 7 runs, 50000 loops each)
method=len, n=1000: 19.844 ns ± 38.63% (median of 7 runs, 10000000 loops each)
method=sum, n=10000: 50.208 µs ± 9.89% (median of 7 runs, 5000 loops each)
method=len, n=10000: 28.686 ns ± 1.22% (median of 7 runs, 20000000 loops each)
```

```python
print(bench)
```
```text
┌────────┬────────┬────────────────────────────┬───────────┐
│ method ┆ n      ┆ median_execution_time (ns) ┆ ± (%)     │
╞════════╪════════╪════════════════════════════╪═══════════╡
│ sum    ┆ 100    ┆ 575.124442                 ┆ 3.353129  │
│ len    ┆ 100    ┆ 19.036998                  ┆ 0.854601  │
│ sum    ┆ 1_000  ┆ 2_961.25732                ┆ 36.698258 │
│ len    ┆ 1_000  ┆ 19.844193                  ┆ 38.63371  │
│ sum    ┆ 10_000 ┆ 50_207.584997              ┆ 9.894165  │
│ len    ┆ 10_000 ┆ 28.686439                  ┆ 1.22376   │
└────────┴────────┴────────────────────────────┴───────────┘
```

## JAX Support

ZeroPyBench automatically detects JAX arrays and optimizes benchmarking accordingly:

```python
import jax.numpy as jnp
from zeropybench import Benchmark

bench = Benchmark(repeat=20, verbose=True)
x = jnp.ones(1_000_000)
y = jnp.ones(1_000_000)

with bench():
    x + y
```

```
Setup code:
    @jax.jit
    def __bench_func(x, y):
        return x + y
Benchmarked code:
    __bench_func(x, y).block_until_ready()
943.426 µs ± 3.98% (median of 20 runs, 500 loops each)
```
When JAX code is detected, ZeroPyBench:

1. **Wraps the code in a JIT-compiled function** to measure optimized execution
2. **Separates compilation from execution** by reporting `compilation_time` separately
3. **Captures the StableHLO representation** of the compiled function in the `hlo` field
4. **Uses ``block_until_ready``** to ensure accurate timing of asynchronous operations

The benchmark report includes additional fields for JAX:
- `first_execution_time`: Time of the initial (possibly uncompiled) execution
- `compilation_time`: Time to lower and compile the function
- `hlo`: The StableHLO text representation of the compiled computation

```python
report = bench.to_dicts()[0]
print(report['compilation_time'])  # e.g., 12345.67 ns
print(report['hlo'][:100])         # HLO module "jit___bench_func" ...
```

## Installation

```bash
pip install zeropybench
```

## Export and Visualization

```python
# Export results
bench.write_csv('results.csv')
bench.write_parquet('results.parquet')
bench.write_markdown('results.md')

# Plot results
bench.plot()
bench.write_plot('results.pdf')
```

## Configuration

```python
Benchmark(
    repeat=7,                     # Number of measurement repetitions
    min_duration_per_repeat=0.2,  # Minimum duration per repeat (seconds)
    verbose=True,                 # Print the setup and benchmarked code
)
```

<!-- End common text with source/index.md -->

## License

MIT
