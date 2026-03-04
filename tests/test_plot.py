"""Tests for BenchmarkPlotter."""

from pathlib import Path

import matplotlib
import polars as pl
import pytest
from polars.testing import assert_frame_equal

matplotlib.use('Agg')  # Non-interactive backend

from zeropybench import BenchmarkPlotter


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Simple benchmark DataFrame with one integer column."""
    return pl.DataFrame(
        {
            'n': [10, 100, 1000],
            'execution_times': [[1.0, 1.1, 1.2], [10.0, 10.1, 10.2], [100.0, 100.1, 100.2]],
            'median_execution_time': [1.1, 10.1, 100.1],
        }
    )


@pytest.fixture
def multidimensional_df() -> pl.DataFrame:
    """Benchmark DataFrame with multiple dimensions."""
    return pl.DataFrame(
        {
            'n': [10, 10, 100, 100],
            'method': ['sum', 'len', 'sum', 'len'],
            'execution_times': [[1.0, 1.1], [0.5, 0.6], [10.0, 10.1], [0.5, 0.6]],
            'median_execution_time': [1.05, 0.55, 10.05, 0.55],
        }
    )


@pytest.fixture
def multiple_integer_columns_df() -> pl.DataFrame:
    """Benchmark DataFrame with multiple integer columns."""
    return pl.DataFrame(
        {
            'n': [10, 10, 100, 100],
            'm': [5, 10, 5, 10],
            'execution_times': [[1.0, 1.1], [2.0, 2.1], [10.0, 10.1], [20.0, 20.1]],
            'median_execution_time': [1.05, 2.05, 10.05, 20.05],
        }
    )


@pytest.fixture
def float_column_df() -> pl.DataFrame:
    """Benchmark DataFrame with float column instead of integer."""
    return pl.DataFrame(
        {
            'x': [1.0, 2.0, 3.0],
            'execution_times': [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
            'median_execution_time': [1.05, 2.05, 3.05],
        }
    )


@pytest.fixture
def no_numeric_user_df() -> pl.DataFrame:
    """Benchmark DataFrame with no numeric user columns (only internal ones)."""
    return pl.DataFrame(
        {
            'name': ['test1', 'test2'],
            'execution_times': [[1.0, 1.1], [2.0, 2.1]],
            # first_execution_time is excluded from x-axis inference
            'first_execution_time': [1.05, 2.05],
        }
    )


def test_benchmark_plotter_conversion_from_csv() -> None:
    df = pl.DataFrame({'median_execution_time': [1.0], 'execution_times': ['[0.9, 1.1]']})
    plotter = BenchmarkPlotter(df)

    expected_df = pl.DataFrame({'median_execution_time': [1.0], 'execution_times': [[0.9, 1.1]]})
    assert_frame_equal(plotter.df, expected_df)


def test_benchmark_plotter_display_time_units() -> None:
    df = pl.DataFrame({'median_execution_time': [1.0]})
    plotter = BenchmarkPlotter(df, display_time_units='ns')

    assert plotter.display_time_units == 'ns'


def test_benchmark_plotter_display_time_units_from_us() -> None:
    df = pl.DataFrame({'median_execution_time': [1.0]})
    plotter = BenchmarkPlotter(df, display_time_units='us')

    assert plotter.display_time_units == 'µs'


def test_create_figure(simple_df: pl.DataFrame, tmp_path: Path):
    """Test creating a figure."""
    plotter = BenchmarkPlotter(simple_df)
    fig = plotter.create_figure()
    assert fig is not None

    path = tmp_path / 'results.png'
    fig.savefig(path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_save(simple_df: pl.DataFrame, tmp_path: Path):
    """Test saving a plot to file."""
    plotter = BenchmarkPlotter(simple_df)
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_x_as_string(simple_df: pl.DataFrame, tmp_path: Path):
    """Test specifying x axis as string."""
    plotter = BenchmarkPlotter(simple_df, x='n')
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_x_as_expression(simple_df: pl.DataFrame, tmp_path: Path):
    """Test specifying x axis as Polars expression."""
    plotter = BenchmarkPlotter(simple_df, x=pl.col('n'))
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_y_as_string(simple_df: pl.DataFrame, tmp_path: Path):
    """Test specifying y axis as string."""
    plotter = BenchmarkPlotter(simple_df, y='median_execution_time')
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_y_as_expression(multidimensional_df: pl.DataFrame, tmp_path: Path):
    """Test specifying y axis as Polars expression."""
    plotter = BenchmarkPlotter(multidimensional_df, y=pl.col('median_execution_time'))
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_by_as_string(multidimensional_df: pl.DataFrame, tmp_path: Path):
    """Test specifying by parameter as string."""
    plotter = BenchmarkPlotter(multidimensional_df, by='method')
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_by_as_list(multidimensional_df: pl.DataFrame, tmp_path: Path):
    """Test specifying by parameter as list."""
    plotter = BenchmarkPlotter(multidimensional_df, by=['method'])
    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_invalid_x_column(simple_df: pl.DataFrame):
    """Test error for invalid x column."""
    with pytest.raises(ValueError, match='not in the benchmark'):
        BenchmarkPlotter(simple_df, x='nonexistent')


def test_invalid_y_column(simple_df: pl.DataFrame):
    """Test error for invalid y column."""
    with pytest.raises(ValueError, match='not in the benchmark'):
        BenchmarkPlotter(simple_df, y='nonexistent')


def test_infer_x_axis_single_integer_column(simple_df: pl.DataFrame, tmp_path: Path):
    """Test x-axis inference with single integer column."""
    plotter = BenchmarkPlotter(simple_df)
    # Should infer 'n' as x-axis
    assert plotter.x.meta.output_name() == 'n'

    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_infer_x_axis_multiple_integer_columns(
    multiple_integer_columns_df: pl.DataFrame, tmp_path: Path
):
    """Test x-axis inference with multiple integer columns."""
    plotter = BenchmarkPlotter(multiple_integer_columns_df)
    # Should infer one of the integer columns
    assert plotter.x.meta.output_name() in ['n', 'm']

    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_infer_x_axis_float_column(float_column_df: pl.DataFrame, tmp_path: Path):
    """Test x-axis inference with float column."""
    plotter = BenchmarkPlotter(float_column_df)
    # Should infer 'x' as x-axis (float column)
    assert plotter.x.meta.output_name() == 'x'

    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_infer_x_axis_no_numeric_columns(no_numeric_user_df: pl.DataFrame):
    """Test error when no numeric columns available for x-axis."""
    with pytest.raises(ValueError, match='No numerical axis can be inferred'):
        BenchmarkPlotter(no_numeric_user_df)


def test_legend_partitioning(multidimensional_df: pl.DataFrame, tmp_path: Path):
    """Test that legend partitioning works correctly."""
    plotter = BenchmarkPlotter(multidimensional_df)
    # Without 'by', 'method' should be used for legend
    legend_by = plotter._compute_legend_by()
    assert 'method' in legend_by

    path = tmp_path / 'results.png'
    plotter.save(path)
    assert path.exists()


def test_time_units_in_ylabel(simple_df: pl.DataFrame):
    """Test that time units appear in y-axis label."""
    for unit in ['ns', 'µs', 'ms', 's']:
        plotter = BenchmarkPlotter(simple_df, unit)  # type: ignore[arg-type]
        fig = plotter.create_figure()
        ax = fig.axes[0]
        assert unit in ax.get_ylabel()


def test_subplots_keywords(simple_df: pl.DataFrame, tmp_path: Path):
    """Test passing additional keywords to subplots."""
    plotter = BenchmarkPlotter(simple_df)
    fig = plotter.create_figure(figsize=(10, 8))
    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 8


def test_show(simple_df: pl.DataFrame, monkeypatch):
    """Test show method calls fig.show()."""
    plotter = BenchmarkPlotter(simple_df)

    show_called = []

    def mock_show(self):
        show_called.append(True)

    monkeypatch.setattr('matplotlib.figure.Figure.show', mock_show)
    plotter.show()

    assert len(show_called) == 1
