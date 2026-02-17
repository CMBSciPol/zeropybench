# zeropybench

A Python benchmarking library with zero overhead, designed for multidimensional performance analysis.

## Installation

```bash
pip install zeropybench
```

## Usage

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
method=sum, n=100: 0.579 us ± 2.38 ns (median ± std. dev. of 7 runs, 500000 loops each)
method=len, n=100: 0.020 us ± 0.45 ns (median ± std. dev. of 7 runs, 20000000 loops each)
method=sum, n=1000: 5.369 us ± 44.70 ns (median ± std. dev. of 7 runs, 50000 loops each)
method=len, n=1000: 0.029 us ± 0.09 ns (median ± std. dev. of 7 runs, 10000000 loops each)
method=sum, n=10000: 53.728 us ± 69.86 ns (median ± std. dev. of 7 runs, 5000 loops each)
method=len, n=10000: 0.029 us ± 0.25 ns (median ± std. dev. of 7 runs, 10000000 loops each)
```

```text
print(bench)
┌────────┬────────┬─────────────────────────────────┐
│ method ┆ n      ┆ execution_times                 │
╞════════╪════════╪═════════════════════════════════╡
│ sum    ┆ 100    ┆ [0.577805, 0.57815, … 0.581231… │
│ len    ┆ 100    ┆ [0.019207, 0.019278, … 0.01958… │
│ sum    ┆ 1_000  ┆ [5.417795, 5.33863, … 5.35146]  │
│ len    ┆ 1_000  ┆ [0.028898, 0.030144, … 0.03007… │
│ sum    ┆ 10_000 ┆ [53.743199, 53.664567, … 53.72… │
│ len    ┆ 10_000 ┆ [0.028857, 0.028911, … 0.02942… │
└────────┴────────┴─────────────────────────────────┘
```

## Features

- **Context manager API**: Benchmark any code block with `with bench(...): ...`
- **Multidimensional**: Tag benchmarks with arbitrary keyword arguments
- **Zero overhead**: Code is passed directly to `timeit.Timer`, no wrapper function
- **Auto-scaling**: Automatically determines the number of iterations for reliable measurements
- **Multiple exports**: CSV, Parquet, Markdown
- **Plotting**: Built-in visualization with matplotlib

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
    repeat=7,                    # Number of measurement repetitions
    min_duration_of_repeat=0.2,  # Minimum duration per repeat (seconds)
    time_units='ns',             # Time units: 'ns', 'us', 'ms', 's'
)
```

## License

MIT
