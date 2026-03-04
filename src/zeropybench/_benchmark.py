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
from typing import Any

import polars as pl

from ._jax import CodeASTParser
from ._plot import BenchmarkPlotter

__all__ = ['Benchmark']

from ._units import get_optimal_time_units, to_units

ValidBenchmarkType = bool | int | float | str | list[float]


class Benchmark:
    """A class for multidimensional benchmarking of code snippets.

        Usage:
            >>> import jax.numpy as jnp
            >>> bench = Benchmark(repeat=10)
            >>> for N in [10, 100, 1000, 10000, 100_000, 1_000_000]:
            ...     x = jnp.ones(N)
            ...     y = jnp.ones(1000)
            ...     with bench(method='broadcast right', N=N):
            ...         x[:, None] + y[None, :]
            ...     with bench(method='broadcast left', N=N):
            ...         x[None, :] + y[:, None]
            >>> print(bench)
    ┌─────────────────┬───────────┬────────────────────────────┬───────────┬───────────────────────────┬───────────────────────┐
    │ method          ┆ N         ┆ median_execution_time (µs) ┆ ± (%)     ┆ first_execution_time (µs) ┆ compilation_time (µs) │
    ╞═════════════════╪═══════════╪════════════════════════════╪═══════════╪═══════════════════════════╪═══════════════════════╡
    │ broadcast right ┆ 10        ┆ 42.898731                  ┆ 1.624434  ┆ 57_578.695007             ┆ 35_968.202996         │
    │ broadcast left  ┆ 10        ┆ 42.311095                  ┆ 0.984904  ┆ 42_322.827008             ┆ 26_705.180993         │
    │ broadcast right ┆ 100       ┆ 42.385543                  ┆ 0.367914  ┆ 49_297.293008             ┆ 37_931.365005         │
    │ broadcast left  ┆ 100       ┆ 44.101337                  ┆ 2.147681  ┆ 50_773.183                ┆ 38_241.692004         │
    │ broadcast right ┆ 1_000     ┆ 56.966551                  ┆ 2.445284  ┆ 38_928.845999             ┆ 38_215.008011         │
    │ broadcast left  ┆ 1_000     ┆ 50.634992                  ┆ 9.380772  ┆ 31_585.520002             ┆ 28_907.275002         │
    │ broadcast right ┆ 10_000    ┆ 161.021684                 ┆ 3.255582  ┆ 36_715.780996             ┆ 27_870.487989         │
    │ broadcast left  ┆ 10_000    ┆ 166.723878                 ┆ 4.231237  ┆ 42_878.715001             ┆ 26_596.944008         │
    │ broadcast right ┆ 100_000   ┆ 1_381.86161                ┆ 11.307259 ┆ 49_187.76                 ┆ 41_679.561997         │
    │ broadcast left  ┆ 100_000   ┆ 1_394.936208               ┆ 4.29877   ┆ 44_465.061001             ┆ 27_617.544998         │
    │ broadcast right ┆ 1_000_000 ┆ 10_284.1733                ┆ 0.706548  ┆ 34_732.489992             ┆ 26_034.978            │
    │ broadcast left  ┆ 1_000_000 ┆ 20_405.4006                ┆ 0.286     ┆ 48_092.860001             ┆ 26_902.942001         │
    └─────────────────┴───────────┴────────────────────────────┴───────────┴───────────────────────────┴───────────────────────┘
            >>> bench.write_csv('bench.csv')
            >>> bench.write_markdown('bench.md')
            >>> bench.plot()
            >>> bench.write_plot('bench.pdf')

        Attributes:
            repeat: The number of times the estimation of the elapsed time will be performed. Each
                repeat will usually execute the benchmarked code many times.
            min_duration_of_repeat: The minimum duration of one repeat, in seconds. The function will be
                executed as many times as it is necessary so that the total execution time is greater
                than this value. The execution time for this repeat is the mean value of the execution
                times.
            _report: Storage for the individual measurements.
            _cache: Cache of the content of the file names used to extract the with statement context.
                We don't use the linecache module (except for <...> files) since it's preferable to
                reset the cache each time a Benchmark class is instantiated (otherwise modifications of
                the benchmark may not be reflected).
    """

    def __init__(
        self,
        *,
        repeat: int = 7,
        min_duration_of_repeat: float = 0.2,
    ) -> None:
        """Returns a Benchmark instance.

        Args:
            repeat: The number of times the estimation of the elapsed time will be performed. Each
                repeat will usually execute the benchmarked code many times.
            min_duration_of_repeat: The minimum duration of one repeat, in seconds. The function
                will be executed as many times as necessary so that the total execution time is
                greater than this value. The execution time for this repeat is the mean value of
                the execution times.
        """
        self.repeat = repeat
        self.min_duration_of_repeat = min_duration_of_repeat
        self._report: list[dict[str, ValidBenchmarkType]] = []
        self._cache: dict[str, str] = {}

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
        is_jax = self._is_jax_context(f_locals)
        if is_jax:
            parser = CodeASTParser.from_code(code)
            code, globals = parser.transform_jax_code(f_locals, f_globals)
            hlo, compilation_time = self._compile_jax(globals)
            is_jax_keywords = {
                'first_execution_time': first_time,
                'compilation_time': compilation_time,
                'hlo': hlo,
            }

        else:
            is_jax_keywords = {}
            globals = f_locals | f_globals
        execution_times, number = self._run_many_times(code, first_time, globals)
        median, rel_stdev = self._get_statistics(execution_times)
        units = get_optimal_time_units([median])
        median_display = to_units(median, units)
        message = ', '.join(f'{k}={v}' for k, v in keywords.items() if k != 'first_execution_time')
        print(
            f'{message}: {median_display:.3f} {units} ± {rel_stdev:.2f}% '
            f'(median of {self.repeat} runs, {number} loops each)'
        )

        record: dict[str, ValidBenchmarkType] = {
            **keywords,
            'median_execution_time': median,
            'execution_times': sorted(execution_times),
        }
        record.update(is_jax_keywords)  # type: ignore[arg-type]
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
            self._cache[filename] = text
        return text.splitlines()

    def _is_jax_context(self, locals: dict[str, Any]) -> bool:
        """Returns true if a variable in the with context is a JAX array."""
        jax = sys.modules.get('jax')
        if jax is None:
            return False
        jaxlib = sys.modules.get('jaxlib')
        if jaxlib is None:
            raise ImportError('The library JAX is installed but not jaxlib...')
        return any(isinstance(_, jax.Array | jaxlib._jax.PjitFunction) for _ in locals.values())

    def _compile_jax(self, globals: dict[str, Any]) -> tuple[str, float]:
        """Compile the JAX function and return the HLO and compilation time in seconds."""
        bench_func = globals['__bench_func']
        param_names = list(inspect.signature(bench_func).parameters.keys())
        arg_values = [globals[name] for name in param_names]

        start_time = time.perf_counter()
        lowered = bench_func.lower(*arg_values)
        compiled = lowered.compile()
        compilation_time = time.perf_counter() - start_time

        hlo = compiled.as_text()
        return hlo, compilation_time

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
        """Returns the number of loops so that total time is greater than min_duration_of_repeat.

        Calls the timeit method with increasing numbers from the sequence
        1, 2, 5, 10, 20, 50, ... until the time taken is at least min_duration_of_repeat
        Returns (number, time_taken_in_seconds).

        Adapted from the timeit module.
        """
        if first_time >= self.min_duration_of_repeat:
            return 1, first_time

        timer = timeit.Timer(func, globals=globals)

        i = 1
        while True:
            for j in 1, 2, 5:
                if (i, j) == (1, 1):
                    continue
                number = i * j
                time_taken = timer.timeit(number)
                if time_taken >= self.min_duration_of_repeat:
                    return number, time_taken
            i *= 10

    def _get_statistics(self, execution_times: list[float]) -> tuple[float, float]:
        """Return the median and the relative MAD scaled to estimate standard deviation"""
        df = pl.DataFrame({'values': [execution_times]})
        df = df.select(median=pl.col('values').list.median(), mad=self._get_mad(pl.col('values')))
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
        return pl.DataFrame(self._report)

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
                pl.exclude(excluded_columns),
                pl.lit(None, pl.Float64).alias('± (%)'),
                *extra_columns,
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
        )
        return df

    def to_dicts(self) -> list[dict[str, Any]]:
        """Returns the benchmark as a list of dicts."""
        return deepcopy(self._report)

    def write_csv(self, path: Path | str) -> None:
        """Writes the benchmark report as CSV.

        Args:
            path: The path of the CSV file.
        """
        self.to_dataframe().with_columns(
            execution_times='['
            + pl.col('execution_times').cast(pl.List(pl.String)).list.join(', ')
            + ']'
        ).write_csv(path)

    def write_parquet(self, path: Path | str) -> None:
        """Writes the benchmark report as Parquet.

        Args:
            path: The path of the Parquet file.
        """
        self.to_dataframe().write_parquet(path)

    def write_markdown(self, path: Path | str) -> None:
        """Writes the benchmark report as MarkDown table.

        Args:
            path: The path of the MarkDown file.
        """
        if not isinstance(path, Path):
            path = Path(path)
        with pl.Config(
            tbl_formatting='ASCII_MARKDOWN',
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):
            path.write_text(str(self.to_dataframe()))

    def plot(
        self,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
        **subplots_keywords: Any,
    ) -> None:
        """Plots the benchmark report in a figure.

        Args:
            x: The x-axis of the plots, as a benchmark report column name or expression.
            y: The y-axis of the plots, as a benchmark report column name or expression.
            by: Key to divide into several subplots.
        """
        plotter = self._create_plotter(x=x, y=y, by=by)
        plotter.show(**subplots_keywords)

    def write_plot(
        self,
        path: Path | str,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
        **subplots_keywords: Any,
    ) -> None:
        """Saves the benchmark plot in a file.

        Args:
            path: The path of the plot file.
            x: The x-axis of the plots, as a benchmark report column name or expression.
            y: The y-axis of the plots, as a benchmark report column name or expression.
            by: Key to divide into several subplots.
        """
        plotter = self._create_plotter(x=x, y=y, by=by)
        plotter.save(path, **subplots_keywords)

    def _create_plotter(
        self,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
    ) -> BenchmarkPlotter:
        """Create a BenchmarkPlotter instance."""
        return BenchmarkPlotter(self.to_dataframe(), x=x, y=y, by=by)
