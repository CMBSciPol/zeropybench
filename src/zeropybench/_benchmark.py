import contextlib
import inspect
import linecache
import sys
import textwrap
import time
import timeit
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from ._io import BenchmarkReader, BenchmarkWriter
from ._jax import CodeASTParser
from ._plot import BenchmarkPlotter
from ._units import get_optimal_time_units, to_units

__all__ = ['Benchmark', 'read_benchmark']

ValidBenchmarkType = bool | int | float | str | list[float] | None


class Benchmark:
    """A class for multidimensional benchmarking of code snippets.

    Example::

        import jax.numpy as jnp

        bench = Benchmark(repeat=10, verbose=True)
        for N in [10_000, 100_000, 1_000_000]:
            x = jnp.ones(N)
            y = jnp.ones(1000)
            with bench(method='broadcast right', N=N):
                x[:, None] + y[None, :]
            with bench(method='broadcast left', N=N):
                x[None, :] + y[:, None]
        print(bench)

    Output::

        ┌───┬─────────────────┬───────────┬────────────────────────────┬──────────┬───────────────────────────┬───────────────────────┐
        │   ┆ method          ┆ N         ┆ median_execution_time (ms) ┆ ± (%)    ┆ first_execution_time (ms) ┆ compilation_time (ms) │
        ╞═══╪═════════════════╪═══════════╪════════════════════════════╪══════════╪═══════════════════════════╪═══════════════════════╡
        │ 0 ┆ broadcast right ┆ 10_000    ┆ 1.621035                   ┆ 7.45449  ┆ 114.662899                ┆ 82.715331             │
        │ 1 ┆ broadcast left  ┆ 10_000    ┆ 1.637076                   ┆ 1.211578 ┆ 76.16242                  ┆ 59.710502             │
        │ 2 ┆ broadcast right ┆ 100_000   ┆ 10.030257                  ┆ 1.190149 ┆ 44.89509                  ┆ 35.997909             │
        │ 3 ┆ broadcast left  ┆ 100_000   ┆ 10.297392                  ┆ 1.467783 ┆ 133.313475                ┆ 54.304368             │
        │ 4 ┆ broadcast right ┆ 1_000_000 ┆ 94.146169                  ┆ 0.159088 ┆ 137.82041                 ┆ 44.644484             │
        │ 5 ┆ broadcast left  ┆ 1_000_000 ┆ 196.27792                  ┆ 0.093503 ┆ 112.162553                ┆ 80.459331             │
        └───┴─────────────────┴───────────┴────────────────────────────┴──────────┴───────────────────────────┴───────────────────────┘

    Then export or plot::

        bench.write_csv('bench.csv')
        bench.plot()

    Attributes:
        repeat (int): The number of times the estimation of the elapsed time will be
            performed. Each repeat will usually execute the benchmarked code many times.
        min_duration_per_repeat (float): The minimum duration of one repeat, in seconds.
            The function will be executed as many times as necessary so that the total
            execution time is greater than this value.
    """

    DEFAULT_REPEAT: ClassVar[int] = 7
    """The default number of measurement repetitions."""

    DEFAULT_MIN_DURATION_PER_REPEAT: ClassVar[float] = 0.2
    """The default minimum duration per repeat in seconds."""

    _report: list[dict[str, ValidBenchmarkType]]
    """Storage for the individual measurements."""

    _cache: dict[str, str]
    """Cache of the content of the file names used to extract the with statement context.
    We don't use the linecache module (except for <...> files) since it's preferable to
    reset the cache each time a Benchmark class is instantiated (otherwise modifications of
    the benchmark may not be reflected)."""

    def __init__(
        self,
        *,
        repeat: int = DEFAULT_REPEAT,
        min_duration_per_repeat: float = DEFAULT_MIN_DURATION_PER_REPEAT,
        verbose: bool = False,
    ) -> None:
        """Returns a Benchmark instance.

        Args:
            repeat: The number of times the estimation of the elapsed time will be performed. Each
                repeat will usually execute the benchmarked code many times.
            min_duration_per_repeat: The minimum duration of one repeat, in seconds. The function
                will be executed as many times as necessary so that the total execution time is
                greater than this value. The execution time for this repeat is the mean value of
                the execution times.
            verbose: If True, print the setup and benchmarked code to stderr. For JAX benchmarks,
                this shows the JIT-compiled function definition and the actual timed call
                (e.g., ``__bench_func(x, y).block_until_ready()``).
        """
        self.repeat = repeat
        self.min_duration_per_repeat = min_duration_per_repeat
        self.verbose = verbose
        self._report = []
        self._cache = {}

    def __repr__(self) -> str:
        with pl.Config(
            thousands_separator='_',
            tbl_cols=-1,
            tbl_rows=-1,
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):
            return str(self._to_display_dataframe())

    @contextlib.contextmanager
    def __call__(self, **keywords: ValidBenchmarkType) -> Iterator[None]:
        start_time = time.perf_counter()
        yield
        first_time = time.perf_counter() - start_time
        code, f_locals, f_globals = self._get_execution_context()
        globals = f_globals | f_locals
        parser = CodeASTParser.from_code(code, globals)

        extra_info: dict[str, ValidBenchmarkType]
        if parser.is_jax_context():
            setup, param_names, globals = parser.transform_jax_code()
            hlo, compilation_time, is_single_array = self._compile_jax(param_names, globals)
            code = f'__bench_func({", ".join(param_names)})'
            if is_single_array:
                code += '.block_until_ready()'
            else:
                code = f'jax.block_until_ready({code})'
            extra_info = {
                'first_execution_time': first_time,
                'compilation_time': compilation_time,
                'hlo': hlo,
            }
            if self.verbose:
                print(f'Setup code:\n{textwrap.indent(setup, "    ")}', file=sys.stderr)

        else:
            extra_info = {}

        if self.verbose:
            print(f'Benchmarked code:\n{textwrap.indent(code, "    ")}', file=sys.stderr)

        if parser.is_jax_context():
            # In JAX context, first_time is not representative because of XLA optimizations.
            # Run the jitted code once to get an accurate estimate for autorange.
            timer = timeit.Timer(code, globals=globals)
            estimated_time = timer.timeit(1)
        else:
            estimated_time = first_time

        execution_times, number = self._run_many_times(code, estimated_time, globals)
        median, rel_stdev = self._get_statistics(execution_times)
        units = get_optimal_time_units([median])
        median_display = to_units(median, units)
        if keywords:
            message = ', '.join(f'{k}={v}' for k, v in keywords.items()) + ': '
        else:
            message = ''
        print(
            f'{message}{median_display:.3f} {units} ± {rel_stdev:.2f}% '
            f'(median of {self.repeat} runs, {number} loops each)'
        )

        record: dict[str, ValidBenchmarkType] = {
            **keywords,
            'median_execution_time': median,
            'execution_times': execution_times,
            **extra_info,
        }
        self._report.append(record)

    def _get_execution_context(self) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Return the code as string, and the locals and globals as dicts."""
        cf = inspect.currentframe()
        assert cf is not None
        cf = cf.f_back
        assert cf is not None
        cf = cf.f_back
        assert cf is not None
        cf = cf.f_back
        assert cf is not None
        filename = cf.f_code.co_filename
        code = self._get_code(filename, cf.f_lineno)
        return code, cf.f_locals, cf.f_globals

    def _get_code(self, filename: str, line_number: int) -> str:
        """Return the content inside the with statement context as text."""
        lines = self._get_lines(filename)
        context = []
        line_with = lines[line_number - 1]
        indent_with = len(line_with) - len(line_with.lstrip())
        for line in lines[line_number:]:
            stripped_line = line.lstrip()
            indent = len(line) - len(stripped_line)
            if stripped_line and indent <= indent_with:
                break
            context.append(line)
        code = textwrap.dedent('\n'.join(context)).strip()
        return code

    def _get_lines(self, filename: str) -> list[str]:
        text = self._cache.get(filename)
        if text is None:
            # Use linecache for special files (<...>, ipykernel temp files, etc.)
            # linecache handles both regular files and IPython/Jupyter execution
            text = ''.join(linecache.getlines(filename))
            if not text and filename == '<string>':
                # Handle python -c: get code from /proc/self/cmdline on Linux
                text = self._get_code_from_cmdline()
            self._cache[filename] = text
        return text.splitlines()

    @staticmethod
    def _get_code_from_cmdline() -> str:
        """Get Python code passed via 'python -c' from /proc/self/cmdline."""
        cmdline_path = Path('/proc/self/cmdline')
        if not cmdline_path.exists():
            raise RuntimeError(
                "Cannot read source code: '/proc/self/cmdline' does not exist. "
                "Benchmarking code passed via 'python -c' is only supported on Linux."
            )
        cmdline = cmdline_path.read_bytes().decode().split('\x00')
        if '-c' not in cmdline:
            raise RuntimeError(
                "Cannot read source code: '-c' flag not found in command line. "
                "This is unexpected when the source file is '<string>'."
            )
        idx = cmdline.index('-c')
        return cmdline[idx + 1]

    def _compile_jax(
        self, param_names: list[str], globals: dict[str, Any]
    ) -> tuple[str | None, float | None, bool]:
        """Compile the JAX function and return HLO, compilation time, and output type info.

        Returns:
            A tuple (hlo, compilation_time, is_single_array) where is_single_array
            indicates whether the output is a single JAX array (vs tuple/pytree).
        """
        bench_func = globals['__bench_func']
        arg_values = [globals[name] for name in param_names]

        try:
            start_time = time.perf_counter()
            lowered = bench_func.lower(*arg_values)
            compiled = lowered.compile()
            compilation_time = time.perf_counter() - start_time
        except Exception as exc:
            print(
                f'Warning: the lowering or compilation of the JAX jitted function failed: {exc}',
                file=sys.stderr,
            )
            compilation_time = None
            hlo = None
            is_single_array = False
        else:
            hlo = compiled.as_text()
            is_single_array = lowered.out_tree.num_leaves == 1
        return hlo, compilation_time, is_single_array

    def _run_many_times(
        self, func: Callable[[], object] | str, first_time: float, globals: dict[str, Any] | None
    ) -> tuple[list[float], int]:
        """Returns execution times in seconds.

        Args:
            func: the function or code snippet to be executed.
            first_time: The execution time in seconds of the code that was run in the
                context manager.
            globals: The combined locals and globals of the code.
        """
        number, time_taken = self._autorange(func, first_time, globals)
        timer = timeit.Timer(func, globals=globals)
        runs = [time_taken / number] + [
            _ / number for _ in timer.repeat(repeat=self.repeat - 1, number=number)
        ]
        return runs, number

    def _autorange(
        self, func: Callable[[], object] | str, first_time: float, globals: dict[str, Any] | None
    ) -> tuple[int, float]:
        """Returns the number of loops so that total time is greater than min_duration_per_repeat.

        Calls the timeit method with increasing numbers from the sequence
        1, 2, 5, 10, 20, 50, ... until the time taken is at least min_duration_per_repeat
        Returns (number, time_taken_in_seconds).

        Adapted from the timeit module.
        """
        if first_time >= self.min_duration_per_repeat:
            return 1, first_time

        timer = timeit.Timer(func, globals=globals)

        i = 1
        while True:
            for j in 1, 2, 5:
                if (i, j) == (1, 1):
                    continue
                number = i * j
                time_taken = timer.timeit(number)
                if time_taken >= self.min_duration_per_repeat:
                    return number, time_taken
            i *= 10

    @staticmethod
    def _get_statistics(execution_times: list[float]) -> tuple[float, float]:
        """Return the median and the relative MAD scaled to estimate standard deviation"""
        df = pl.DataFrame({'values': [execution_times]})
        df = df.select(
            median=pl.col('values').list.median(), mad=Benchmark._get_mad(pl.col('values'))
        )
        median = df['median'].item()
        rel_stdev = 1.4826 * df['mad'].item() / median * 100
        return median, rel_stdev

    @staticmethod
    def _get_mad(column: pl.Expr) -> pl.Expr:
        """Return the Median Absolute Deviation."""
        expr_element = abs(pl.element() - pl.element().median()).median()
        return column.list.eval(expr_element).list.first()

    def to_dataframe(self) -> pl.DataFrame:
        """Returns the benchmark as a Polars dataframe with times in seconds."""
        if not self._report:
            schema = {'median_execution_time': pl.Float64(), 'execution_times': pl.List(pl.Float64)}
            return pl.DataFrame({'median_execution_time': [], 'execution_times': []}, schema=schema)
        if 'hlo' in self._report[0]:
            schema = {
                'compilation_time': pl.Float64(),
                'hlo': pl.String(),
            }
        else:
            schema = {}
        return pl.DataFrame(self._report, schema_overrides=schema)

    def _to_display_dataframe(self) -> pl.DataFrame:
        """Returns the benchmark as a Polars dataframe with times in display units."""
        df = self.to_dataframe()
        excluded_columns = [
            'median_execution_time',
            'execution_times',
            'first_execution_time',
            'compilation_time',
            'mad',
            'hlo',
        ]
        extra_columns = [
            col for col in ('first_execution_time', 'compilation_time') if col in df.columns
        ]

        if not self._report:
            return df.select(
                'median_execution_time',
                pl.lit(None, pl.Float64).alias('± (%)'),
                'execution_times',
            )

        units = get_optimal_time_units(df['median_execution_time'])
        suffix = f' ({units})'
        df = df.with_columns(
            mad=self._get_mad(pl.col('execution_times')),
        )
        df = df.select(
            pl.exclude(excluded_columns),
            to_units(pl.col('median_execution_time').name.suffix(suffix), units),
            (1.4826 * pl.col('mad') / pl.col('median_execution_time') * 100).alias('± (%)'),
            to_units(pl.col(extra_columns).name.suffix(suffix), units),
        ).with_row_index('')
        return df

    def __len__(self) -> int:
        """Returns the number of runs in the benchmark."""
        return len(self._report)

    def __bool__(self) -> bool:
        """Returns True if the benchmark is not empty."""
        return len(self) > 0

    def __getitem__(self, item: int) -> dict[str, ValidBenchmarkType]:
        """Returns the benchmark run with the given index (chronologically)."""
        try:
            return self._report[item]
        except IndexError:
            pass
        message = f'contains only {len(self)} runs' if self else 'is empty'
        raise IndexError(f'Index {item} is out of range. The benchmark report {message}.')

    def to_dicts(self) -> list[dict[str, Any]]:
        """Returns the benchmark as a list of dicts."""
        return deepcopy(self._report)

    def write_csv(self, path: Path | str) -> None:
        """Writes the benchmark report as CSV.

        The file includes a header with metadata comments:
        - ``# repeat = <value>``
        - ``# min_duration_per_repeat = <value>``

        Args:
            path: The path of the CSV file.
        """
        self._create_writer().write_csv(path)

    def write_parquet(self, path: Path | str) -> None:
        """Writes the benchmark report as Parquet.

        The file includes metadata:
        - ``repeat``: The number of measurement repetitions
        - ``min_duration_per_repeat``: The minimum duration per repeat in seconds

        Args:
            path: The path of the Parquet file.
        """
        self._create_writer().write_parquet(path)

    def write_markdown(self, path: Path | str) -> None:
        """Writes the benchmark report as MarkDown table.

        Args:
            path: The path of the MarkDown file.
        """
        self._create_writer().write_markdown(path)

    def _create_writer(self) -> BenchmarkWriter:
        """Create a BenchmarkWriter instance."""
        return BenchmarkWriter(self.to_dataframe(), self.repeat, self.min_duration_per_repeat)

    def plot(
        self,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
        reference: str | None = None,
        **subplots_keywords: Any,
    ) -> None:
        """Plots the benchmark report in a figure.

        Args:
            x: The x-axis of the plots, as a benchmark report column name or expression.
            y: The y-axis of the plots, as a benchmark report column name or expression.
            by: Key to divide into several subplots.
            reference: Legend label of the reference method for speedup comparison.
                When specified, a second column of subplots shows the speedup
                (reference_time / method_time) for each method. Values > 1 mean
                faster than the reference.
        """
        plotter = self._create_plotter(x=x, y=y, by=by, reference=reference)
        plotter.show(**subplots_keywords)

    def write_plot(
        self,
        path: Path | str,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
        reference: str | None = None,
        **subplots_keywords: Any,
    ) -> None:
        """Saves the benchmark plot in a file.

        Args:
            path: The path of the plot file.
            x: The x-axis of the plots, as a benchmark report column name or expression.
            y: The y-axis of the plots, as a benchmark report column name or expression.
            by: Key to divide into several subplots.
            reference: Legend label of the reference method for speedup comparison.
                When specified, a second column of subplots shows the speedup
                (reference_time / method_time) for each method. Values > 1 mean
                faster than the reference.
        """
        plotter = self._create_plotter(x=x, y=y, by=by, reference=reference)
        plotter.save(path, **subplots_keywords)

    def _create_plotter(
        self,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
        reference: str | None = None,
    ) -> BenchmarkPlotter:
        """Create a BenchmarkPlotter instance."""
        return BenchmarkPlotter(self.to_dataframe(), x=x, y=y, by=by, reference=reference)


def read_benchmark(path: Path | str, *, verbose: bool = False) -> Benchmark:
    """Reads a benchmark from a CSV or Parquet file.

    The function automatically detects the file format based on the extension
    and reads the metadata (repeat, min_duration_per_repeat) stored in the file.

    Args:
        path: The path to the CSV or Parquet file.
        verbose: If True, set logging level to INFO. If False, set logging level to WARNING.

    Returns:
        A Benchmark instance with the data and metadata from the file.

    Raises:
        ValueError: If the file extension is not .csv or .parquet.
    """
    reader = BenchmarkReader(Benchmark.DEFAULT_REPEAT, Benchmark.DEFAULT_MIN_DURATION_PER_REPEAT)
    df, repeat, min_duration_per_repeat = reader.read(path)
    bench = Benchmark(
        repeat=repeat, min_duration_per_repeat=min_duration_per_repeat, verbose=verbose
    )
    bench._report = df.to_dicts()
    return bench
