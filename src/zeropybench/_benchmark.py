import contextlib
import functools
import inspect
import linecache
import statistics
import sys
import textwrap
import time
import timeit
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.figure
import matplotlib.pyplot as mp
import polars as pl
import polars.selectors as cs
from matplotlib.axes import Axes

__all__ = ['Benchmark']

ValidBenchmarkType = bool | int | float | str | list[float]
TimeUnitType = Literal['ns', 'us', 'ms', 's']
TIME_UNITS_MULTIPLIER: dict[TimeUnitType, float] = {
    's': 1.0,
    'ms': 1000.0,
    'us': 1000_000.0,
    'ns': 1_000_000_000.0,
}
STDEV_TIME_UNITS: dict[TimeUnitType, TimeUnitType] = {
    's': 's',
    'ms': 'us',
    'us': 'ns',
    'ns': 'ns',
}


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
    ┌─────────────────┬───────────┬──────────────────────┬─────────────────────────────────┐
    │ method          ┆ N         ┆ first_execution_time ┆ execution_times                 │
    ╞═════════════════╪═══════════╪══════════════════════╪═════════════════════════════════╡
    │ broadcast right ┆ 10        ┆ 34.353204            ┆ [0.210183, 0.210074, … 0.20946… │
    │ broadcast left  ┆ 10        ┆ 36.472133            ┆ [0.220572, 0.22118, … 0.221245… │
    │ broadcast right ┆ 100       ┆ 27.134279            ┆ [0.226366, 0.226376, … 0.22619… │
    │ broadcast left  ┆ 100       ┆ 26.604938            ┆ [0.2263, 0.228083, … 0.22751]   │
    │ broadcast right ┆ 1_000     ┆ 19.080388            ┆ [0.336571, 0.367122, … 0.15893… │
    │ broadcast left  ┆ 1_000     ┆ 18.100275            ┆ [0.320024, 0.300872, … 0.32235… │
    │ broadcast right ┆ 10_000    ┆ 33.618927            ┆ [5.599664, 5.555191, … 5.69679… │
    │ broadcast left  ┆ 10_000    ┆ 33.551694            ┆ [5.663586, 5.671626, … 5.76813… │
    │ broadcast right ┆ 100_000   ┆ 79.550476            ┆ [51.543066, 51.413535, … 51.61… │
    │ broadcast left  ┆ 100_000   ┆ 79.723651            ┆ [52.384487, 52.2496, … 52.3993… │
    │ broadcast right ┆ 1_000_000 ┆ 482.147623           ┆ [450.758266, 453.388479, … 452… │
    │ broadcast left  ┆ 1_000_000 ┆ 490.124676           ┆ [467.215606, 462.161977, … 463… │
    └─────────────────┴───────────┴──────────────────────┴─────────────────────────────────┘
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
        time_units: The request units for the execution times ('ns', 'us', 'ms' or 's')
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
        time_units: TimeUnitType = 'ns',
    ) -> None:
        """Returns a Benchmark instance.

        Args:
            repeat: The number of times the estimation of the elapsed time will be performed. Each
                repeat will usually execute the benchmarked code many times.
            min_duration_of_repeat: The minimum duration of one repeat, in seconds. The function
                will be executed as many times as necessary so that the total execution time is
                greater than this value. The execution time for this repeat is the mean value of
                the execution times.
            time_units: The request units for the execution times ('ns', 'us', 'ms' or 's').
        """
        self.repeat = repeat
        self.min_duration_of_repeat = min_duration_of_repeat
        self.time_units = time_units
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
            return str(self.as_dataframe())

    @contextlib.contextmanager
    def __call__(self, **keywords: ValidBenchmarkType) -> Iterator[None]:
        start_time = time.perf_counter()
        yield
        first_time = time.perf_counter() - start_time
        cf = inspect.currentframe()
        assert cf is not None
        cf = cf.f_back
        assert cf is not None
        cf = cf.f_back
        assert cf is not None
        filename = cf.f_code.co_filename
        code = self._get_code(filename, cf.f_lineno)
        # func = self._get_func(code, cf.f_globals, cf.f_locals)
        # if self._is_jax_context(cf.f_locals):
        #    # store the compilation time
        #    keywords = {
        #        **keywords,
        #        'first_execution_time': first_time * TIME_UNITS_MULTIPLIER[self.time_units],
        #    }
        #    first_time = self._run_once(func)
        combined_globals = cf.f_globals | cf.f_locals
        execution_times, number = self._run_many_times(code, first_time, combined_globals)
        min_repeat_time, stdev = self._get_statistics(execution_times)
        stdev_time_units = STDEV_TIME_UNITS[self.time_units]
        stdev *= TIME_UNITS_MULTIPLIER[stdev_time_units] / TIME_UNITS_MULTIPLIER[self.time_units]
        message = ', '.join(f'{k}={v}' for k, v in keywords.items() if k != 'first_execution_time')
        print(
            f'{message}: {min_repeat_time:.3f} {self.time_units} ± {stdev:.2f} {stdev_time_units} '
            f'(median ± std. dev. of {self.repeat} runs, {number} loops each)'
        )

        record: dict[str, ValidBenchmarkType] = {
            **keywords,
            #            'Execution time': min_repeat_time,
            'execution_times': execution_times,
        }
        self._report.append(record)

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
            if filename.startswith('<') and filename.endswith('>'):
                text = ''.join(linecache.getlines(filename))
            else:
                text = Path(filename).read_text()
            self._cache[filename] = text
        return text.splitlines()

    def _get_func(
        self, code: str, f_globals: dict[str, Any], f_locals: dict[str, Any]
    ) -> Callable[[], None]:
        func: Callable[[], None]
        try:
            compiled = compile(code, '<benchmark>', 'eval')
            do_exec = False
        except SyntaxError:
            compiled = compile(code, '<benchmark>', 'exec')
            do_exec = True

        if do_exec:
            func = functools.partial(exec, compiled, f_globals, f_locals)
        elif self._is_jax_context(f_locals):
            import jax

            def func() -> None:
                result = eval(compiled, f_globals, f_locals)
                jax.block_until_ready(result)

        else:
            func = functools.partial(eval, compiled, f_globals, f_locals)
        return func

    def _run_once(self, func: Callable[[], object]) -> float:
        """Executes once a function and returns the elapsed time in the benchmark units."""
        start_time = time.perf_counter()
        func()
        return (time.perf_counter() - start_time) * TIME_UNITS_MULTIPLIER[self.time_units]

    def _run_many_times(
        self, func: Callable[[], object] | str, first_time: float, globals: dict[str, Any] | None
    ) -> tuple[list[float], int]:
        """Returns execution time in the benchmark unit."""
        number, time_taken = self._autorange(func, first_time, globals)
        timer = timeit.Timer(func, globals=globals)
        runs = [time_taken / number] + [
            _ * TIME_UNITS_MULTIPLIER[self.time_units] / number
            for _ in timer.repeat(repeat=self.repeat - 1, number=number)
        ]
        return runs, number

    def _autorange(
        self, func: Callable[[], object] | str, first_time: float, globals: dict[str, Any] | None
    ) -> tuple[int, float]:
        """Returns the number of loops so that total time is greater than min_duration_of_repeat..

        Calls the timeit method with increasing numbers from the sequence
        1, 2, 5, 10, 20, 50, ... until the time taken is at least min_duration_of_repeat
        Returns (number, time_taken).

        Adapted from the timeit module.
        """
        if first_time >= self.min_duration_of_repeat * TIME_UNITS_MULTIPLIER[self.time_units]:
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
                    return number, time_taken * TIME_UNITS_MULTIPLIER[self.time_units]
            i *= 10

    def _get_statistics(self, execution_times: list[float]) -> tuple[float, float]:
        low, median, high = statistics.quantiles(execution_times)
        if self.repeat < 2:
            stdev = 0.0
        else:
            mad = statistics.median(abs(_ - median) for _ in execution_times)
            stdev = 1.4826 * mad
        return median, stdev

    def _is_jax_context(self, f_locals: dict[str, Any]) -> bool:
        """Returns true if a variable in the with context is a JAX array."""
        jax = sys.modules.get('jax')
        if jax is None:
            return False
        return any(isinstance(_, jax.Array) for _ in f_locals.values())

    def as_dataframe(self) -> pl.DataFrame:
        """Returns the benchmark as a Polars dataframe."""
        return pl.DataFrame(self._report)

    def write_csv(self, path: Path | str) -> None:
        """Writes the benchmark report as CSV.

        Args:
            path: The path of the CSV file.
        """
        self.as_dataframe().with_columns(
            execution_times='['
            + pl.col('execution_times').cast(pl.List(pl.String)).list.join(', ')
            + ']'
        ).write_csv(path)

    def write_parquet(self, path: Path | str) -> None:
        """Writes the benchmark report as Parquet.

        Args:
            path: The path of the Parquet file.
        """
        self.as_dataframe().write_parquet(path)

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
            path.write_text(str(self.as_dataframe()))

    def plot(
        self,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr = pl.col('execution_times').list.min().alias('Execution time'),
        by: str | Sequence[str] | None = None,
        **subplots_keywords: Any,
    ) -> None:
        """Plots the benchmark report in a figure.

        Args:
            x: The x-axis of the plots, as a benchmark report column name or expression.
            y: The y-axis of the plots, as a benchmark report column name or expression.
            by: Key to divide into several subplots.
        """
        fig = self._subplots(x=x, y=y, by=by, **subplots_keywords)
        fig.show()

    def write_plot(
        self,
        path: Path | str,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr = pl.col('execution_times').list.min().alias('Execution time'),
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
        fig = self._subplots(x=x, y=y, by=by, **subplots_keywords)
        fig.savefig(path)

    def _subplots(
        self,
        *,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr = pl.col('execution_times').list.min().alias('Execution time'),
        by: str | Sequence[str] | None = None,
        **subplots_keywords: Any,
    ) -> matplotlib.figure.Figure:

        df = self.as_dataframe()
        if x is None:
            x = self._infer_default_x_axis_plot(df)
        elif isinstance(x, str):
            x = pl.col(x)
        if isinstance(y, str):
            y = pl.col(y)
        if by is None:
            by = []
        elif isinstance(by, str):
            by = [by]

        invalid_columns = sorted(set(x.meta.root_names()) - set(df.columns))
        if invalid_columns:
            raise ValueError(
                f'The column {", ".join(invalid_columns)} is not in the benchmark: {df.columns}'
            )
        invalid_columns = sorted(set(y.meta.root_names()) - set(df.columns))
        if invalid_columns:
            raise ValueError(
                f'The column {", ".join(invalid_columns)} is not in the benchmark: {df.columns}'
            )

        excluded_columns = (
            set(x.meta.root_names()) | set(y.meta.root_names()) | {'first_execution_time'}
        )
        excluded_columns |= set(by)
        legend_by = list({col: None for col in df.columns if col not in excluded_columns})

        if by:
            plot_partitions = df.partition_by(by, maintain_order=True, as_dict=True)
        else:
            plot_partitions = {(): df}

        nsubplot = len(plot_partitions)
        fig: matplotlib.figure.Figure
        fig, axs = mp.subplots(nsubplot, **subplots_keywords)
        if nsubplot == 1:
            axs = (axs,)
        for (plot_keys, plot_partition), ax in zip(plot_partitions.items(), axs):
            xlabel = x.meta.output_name()
            ylabel = f'{y.meta.output_name()} [{self.time_units}]'
            self._subplot(ax, plot_partition, x=x, xlabel=xlabel, y=y, ylabel=ylabel, by=legend_by)

        return fig

    def _subplot(
        self,
        ax: Axes,
        df: pl.DataFrame,
        *,
        x: pl.Expr,
        y: pl.Expr,
        xlabel: str,
        ylabel: str,
        by: list[str],
    ) -> None:
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if by:
            legend_partitions = df.partition_by(by, maintain_order=True, as_dict=True)
        else:
            legend_partitions = {(): df}

        for legend_keys, legend_partition in legend_partitions.items():
            x_values = legend_partition.select(x).to_series()
            y_values = legend_partition.select(y).to_series()
            label = ', '.join(f'{k}={v}' for k, v in zip(by, legend_keys))
            ax.loglog(x_values, y_values, marker='.', label=label)
            # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(v)))
            ax.legend()

    @staticmethod
    def _infer_default_x_axis_plot(df: pl.DataFrame) -> pl.Expr:
        """Infers the x-axis.

        1) integer columns
        2) numeric columns
        3) columns with most entries
        4) first benchmark keywords (outer scope)
        """
        numeric_df = df.select(cs.numeric()).select(pl.exclude('first_execution_time'))
        integer_df = numeric_df.select(cs.integer())
        if len(integer_df.columns) == 1:
            return pl.col(integer_df.columns[0])
        if len(integer_df.columns) > 1:
            candidate_df = integer_df
        else:
            candidate_df = numeric_df

        if len(candidate_df.columns) == 0:
            raise ValueError('No numerical axis can be inferred from the benchmark.')

        n_uniques = next(candidate_df.select(pl.all().n_unique()).iter_rows(named=True))
        sorted_n_uniques = sorted(n_uniques.items(), key=lambda v: v[1])
        max_entries = sorted_n_uniques[-1][1]
        first_column_with_max_entries = [_[0] for _ in sorted_n_uniques if _[1] == max_entries][0]
        return pl.col(first_column_with_max_entries)
