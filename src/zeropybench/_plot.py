import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.figure
import matplotlib.pyplot as mp
import matplotlib.ticker as ticker
import polars as pl
import polars.selectors as cs
from matplotlib.axes import Axes

from ._units import TimeUnitType, get_optimal_time_units, to_units

__all__ = ['BenchmarkPlotter']

DEFAULT_Y = pl.col('median_execution_time').alias('Execution time')

# Columns that should be excluded from legend partitioning
EXCLUDED_COLUMNS = frozenset(
    {'first_execution_time', 'compilation_time', 'median_execution_time', 'hlo', 'execution_times'}
)


class BenchmarkPlotter:
    """Handles plotting of benchmark results.

    Args:
        df: The benchmark data in seconds as a Polars DataFrame.
        display_time_units: The time units to be used for display (automatically inferred by default).
        x: The x-axis, as a column name or Polars expression.
        y: The y-axis, as a column name or Polars expression.
        by: Key(s) to divide into several subplot rows.
        reference: Legend label of the reference method for speedup comparison.
            When specified, a second column of subplots shows the speedup
            (reference_time / method_time) for each method. Values > 1 mean
            faster than the reference.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        display_time_units: TimeUnitType | Literal['us'] | None = None,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        by: str | Sequence[str] | None = None,
        reference: str | None = None,
    ) -> None:
        if display_time_units is None:
            display_time_units = get_optimal_time_units(df['median_execution_time'])
        elif display_time_units == 'us':
            display_time_units = 'µs'
        if (dtype := df.schema.get('execution_times')) is not None and dtype == pl.String:
            # transform string "[1.12, 0.42]" into list of floats [1.12, 0.42]
            df = df.with_columns(
                pl.col('execution_times')
                .str.strip_prefix('[')
                .str.strip_suffix(']')
                .str.split(', ')
                .cast(pl.List(pl.Float64))
            )
        df = df.with_columns(to_units(pl.col('^.*_times?$'), display_time_units))

        if y is None:
            y = DEFAULT_Y
        self.df = df
        self.display_time_units = display_time_units
        self.x = self._normalize_x(x, df)
        self.y = self._normalize_expr(y)
        self.by = self._normalize_by(by)
        self.reference = reference

        self._validate_columns()

    def _normalize_x(self, x: str | pl.Expr | None, df: pl.DataFrame) -> pl.Expr:
        """Normalize x to a Polars expression, inferring if None."""
        if x is None:
            x = self._infer_x_axis(df)
        if isinstance(x, str):
            return pl.col(x)
        return x

    @staticmethod
    def _normalize_expr(expr: str | pl.Expr) -> pl.Expr:
        """Normalize a string or expression to a Polars expression."""
        if isinstance(expr, str):
            return pl.col(expr)
        return expr

    @staticmethod
    def _normalize_by(by: str | Sequence[str] | None) -> list[str]:
        """Normalize by to a list of strings."""
        if by is None:
            return []
        if isinstance(by, str):
            return [by]
        return list(by)

    def _validate_columns(self) -> None:
        """Validate that x and y columns exist in the DataFrame."""
        invalid_columns = sorted(set(self.x.meta.root_names()) - set(self.df.columns))
        if invalid_columns:
            raise ValueError(
                f'The column {", ".join(invalid_columns)} is not in the benchmark: '
                f'{self.df.columns}'
            )
        invalid_columns = sorted(set(self.y.meta.root_names()) - set(self.df.columns))
        if invalid_columns:
            raise ValueError(
                f'The column {", ".join(invalid_columns)} is not in the benchmark: '
                f'{self.df.columns}'
            )

    def _compute_legend_by(self) -> list[str]:
        """Compute which columns to use for legend partitioning."""
        excluded = (
            set(self.x.meta.root_names())
            | set(self.y.meta.root_names())
            | EXCLUDED_COLUMNS
            | set(self.by)
        )
        return [col for col in self.df.columns if col not in excluded]

    def create_figure(self, **subplots_keywords: Any) -> matplotlib.figure.Figure:
        """Create and return the matplotlib figure.

        Args:
            **subplots_keywords: Additional keyword arguments passed to plt.subplots().

        Returns:
            The matplotlib Figure containing the benchmark plots.
        """
        legend_by = self._compute_legend_by()

        if self.by:
            plot_partitions = self.df.partition_by(self.by, maintain_order=True, as_dict=True)
        else:
            plot_partitions = {(): self.df}

        nsubplot = len(plot_partitions)
        ncols = 2 if self.reference else 1
        fig: matplotlib.figure.Figure
        fig, axs = mp.subplots(nsubplot, ncols, **subplots_keywords)
        if nsubplot == 1 and ncols == 1:
            axs = ((axs,),)
        elif nsubplot == 1:
            axs = (axs,)
        elif ncols == 1:
            axs = tuple((ax,) for ax in axs)
        if nsubplot > 1 or ncols > 1:
            fig.subplots_adjust(hspace=0, wspace=0.05 if ncols > 1 else None)

        xlabel = self.x.meta.output_name()
        ylabel_base = f'{self.y.meta.output_name()} [{self.display_time_units}]'
        ratio_ylabel_base = f'Speedup vs {self.reference}'

        for i, ((plot_keys, plot_partition), ax_row) in enumerate(
            zip(plot_partitions.items(), axs)
        ):
            show_xlabel = i == nsubplot - 1
            if plot_keys:
                by_label = ', '.join(f'{k}: {v}' for k, v in zip(self.by, plot_keys))
                ylabel = f'{by_label}\n{ylabel_base}'
                ratio_ylabel = f'{by_label}\n{ratio_ylabel_base}'
            else:
                ylabel = ylabel_base
                ratio_ylabel = ratio_ylabel_base

            ax_left = ax_row[0]
            colors = self._draw_subplot(
                ax_left,
                plot_partition,
                xlabel=xlabel if show_xlabel else '',
                ylabel=ylabel,
                legend_by=legend_by,
            )
            ax_left.xaxis.set_tick_params(which='both', direction='in', labelbottom=show_xlabel)

            if self.reference:
                ax_right = ax_row[1]
                self._draw_speedup_subplot(
                    ax_right,
                    plot_partition,
                    xlabel=xlabel if show_xlabel else '',
                    ylabel=ratio_ylabel,
                    legend_by=legend_by,
                    colors=colors,
                )
                ax_right.xaxis.set_tick_params(
                    which='both', direction='in', labelbottom=show_xlabel
                )

        return fig

    def _draw_subplot(
        self,
        ax: Axes,
        df: pl.DataFrame,
        *,
        xlabel: str,
        ylabel: str,
        legend_by: list[str],
    ) -> dict[tuple[Any, ...], Any]:
        """Draw a single subplot and return the colors used for each legend key."""
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if legend_by:
            legend_partitions = df.partition_by(legend_by, maintain_order=True, as_dict=True)
        else:
            legend_partitions = {(): df}

        colors: dict[tuple[Any, ...], Any] = {}
        for legend_keys, legend_partition in legend_partitions.items():
            x_values = legend_partition.select(self.x).to_series()
            y_values = legend_partition.select(self.y).to_series()
            label = ', '.join(f'{k}={v}' for k, v in zip(legend_by, legend_keys))
            (line,) = ax.loglog(x_values, y_values, marker='.', label=label if label else None)
            colors[legend_keys] = line.get_color()

        ax.xaxis.set_major_formatter(_format_x_tick)
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(_format_y_tick)

        if legend_by:
            ax.legend()

        return colors

    def _draw_speedup_subplot(
        self,
        ax: Axes,
        df: pl.DataFrame,
        *,
        xlabel: str,
        ylabel: str,
        legend_by: list[str],
        colors: dict[tuple[Any, ...], Any],
    ) -> None:
        """Draw a ratio subplot comparing all legend items to the reference."""
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if legend_by:
            legend_partitions = df.partition_by(legend_by, maintain_order=True, as_dict=True)
        else:
            assert False, 'unreachable'

        # Find the reference partition
        reference_key = None
        for legend_keys in legend_partitions:
            label = ', '.join(f'{k}={v}' for k, v in zip(legend_by, legend_keys))
            if label == self.reference or (
                len(legend_keys) == 1 and legend_keys[0] == self.reference
            ):
                reference_key = legend_keys
                break

        if reference_key is None:
            raise ValueError(f'Reference "{self.reference}" not found in legend keys.')

        reference_partition = legend_partitions[reference_key]
        ref_x = reference_partition.select(self.x).to_series()
        ref_y = reference_partition.select(self.y).to_series()
        # Create a mapping from x to y for the reference
        ref_mapping = dict(zip(ref_x, ref_y))

        for legend_keys, legend_partition in legend_partitions.items():
            x_values = legend_partition.select(self.x).to_series()
            y_values = legend_partition.select(self.y).to_series()
            # Compute speedup (reference / method), so > 1 means faster than reference
            ratios = [ref_mapping.get(x, float('nan')) / y for x, y in zip(x_values, y_values)]
            label = ', '.join(f'{k}={v}' for k, v in zip(legend_by, legend_keys))
            ax.semilogx(
                x_values,
                ratios,
                marker='.',
                label=label if label else None,
                color=colors[legend_keys],
            )

        ax.xaxis.set_major_formatter(_format_x_tick)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8)

        # Move y-axis to the right
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    def show(self, **subplots_keywords: Any) -> None:
        """Display the plot interactively.

        Args:
            **subplots_keywords: Additional keyword arguments passed to plt.subplots().
        """
        fig = self.create_figure(**subplots_keywords)
        fig.show()

    def save(self, path: Path | str, **subplots_keywords: Any) -> None:
        """Save the plot to a file.

        Args:
            path: The path to save the plot to.
            **subplots_keywords: Additional keyword arguments passed to plt.subplots().
        """
        fig = self.create_figure(**subplots_keywords)
        fig.savefig(path)

    @staticmethod
    def _infer_x_axis(df: pl.DataFrame) -> str:
        """Infer the default x-axis from the DataFrame.

        Priority:
        1) Single integer column
        2) Integer column with the greatest number of unique values
        3) Numeric column with the greatest number of unique values

        Raises:
            ValueError: If no numerical column can be inferred.
        """
        numeric_df = df.select(cs.numeric()).select(
            pl.exclude('median_execution_time', 'first_execution_time', 'compilation_time')
        )
        integer_df = numeric_df.select(cs.integer())

        if len(integer_df.columns) == 1:
            return integer_df.columns[0]

        if len(integer_df.columns) > 1:
            candidate_df = integer_df
        else:
            candidate_df = numeric_df

        if len(candidate_df.columns) == 0:
            raise ValueError('No numerical axis can be inferred from the benchmark.')

        n_uniques = next(candidate_df.select(pl.all().n_unique()).iter_rows(named=True))
        first_column_with_max_entries, _ = max(n_uniques.items(), key=lambda x: x[1])
        return first_column_with_max_entries


def _format_x_tick(x: float, pos: float) -> str:
    """Format X tick values: use decimal for 0.001-1000, power of 10 otherwise."""
    if 0.001 <= x <= 1000:
        return f'{x:g}'

    exp = int(math.log10(x))
    return f'$10^{{{exp}}}$'


def _format_y_tick(x: float, pos: float) -> str:
    """Format Y tick values: use decimal for 0.001-1000, scientific notation otherwise."""
    if 0.001 <= x <= 1000:
        return f'{x:g}'
    mantissa, exp = f'{x:.2e}'.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    return f'{mantissa}e{exp}'
