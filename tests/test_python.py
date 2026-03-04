"""Tests for zeropybench with pure Python code."""

from pathlib import Path

import polars as pl
from pytest_mock import MockerFixture

from zeropybench import Benchmark


def test_basic_benchmark():
    """Test basic benchmark functionality."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='sum'):
        sum(range(100))

    df = bench.to_dataframe()
    assert len(df) == 1
    assert df['name'][0] == 'sum'
    assert 'execution_times' in df.columns
    assert len(df['execution_times'][0]) == 3


def test_multiple_benchmarks():
    """Test multiple benchmarks with different parameters."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(n=10):
        sum(range(10))

    with bench(n=100):
        sum(range(100))

    df = bench.to_dataframe()
    assert len(df) == 2
    assert df['n'].to_list() == [10, 100]


def test_multidimensional_keywords():
    """Test benchmarks with multiple keyword arguments."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(method='sum', size=100, variant='a'):
        sum(range(100))

    df = bench.to_dataframe()
    assert df['method'][0] == 'sum'
    assert df['size'][0] == 100
    assert df['variant'][0] == 'a'


def test_time_units():
    """Test different time units."""
    for unit in ['ns', 'us', 'ms', 's']:
        bench = Benchmark(repeat=3, min_duration_of_repeat=0.01, time_units=unit)

        with bench(name='test'):
            sum(range(100))

        assert bench.time_units == unit


def test_multiline_code():
    """Test benchmark with multiple statements."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='multiline'):
        x = list(range(100))
        y = sum(x)
        z = y * 2  # noqa: F841

    d = bench.to_dicts()
    assert len(d) == 1


def test_execution_times_are_positive():
    """Test that all execution times are positive."""
    bench = Benchmark(repeat=5, min_duration_of_repeat=0.01)

    with bench(name='test'):
        sum(range(1000))

    d = bench.to_dicts()
    times = d[0]['execution_times']
    assert all(t > 0 for t in times)


def test_empty_benchmark():
    """Test empty benchmark returns empty dataframe."""
    bench = Benchmark()
    df = bench.to_dicts()
    assert len(df) == 0


def test_local_variables():
    """Test benchmark with local variables."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    data = list(range(100))
    multiplier = 2

    with bench(name='local_sum'):
        sum(data) * multiplier

    d = bench.to_dicts()
    assert len(d) == 1
    assert d[0]['name'] == 'local_sum'


def test_local_variables_in_loop():
    """Test benchmark with local variables inside a loop."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    for n in [10, 100]:
        with bench(n=n):
            sum(range(n))

    df = bench.to_dataframe()
    assert len(df) == 2
    assert df['n'].to_list() == [10, 100]


def test_repr():
    """Test __repr__ returns a string table."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='test'):
        sum(range(100))

    repr_str = repr(bench)
    assert isinstance(repr_str, str)
    assert 'test' in repr_str
    assert 'execution_times' in repr_str


def test_write_csv(tmp_path: Path):
    """Test writing benchmark to CSV."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.csv'
    bench.write_csv(path)
    assert path.exists()
    content = path.read_text()
    assert 'test' in content
    assert 'execution_times' in content


def test_write_parquet(tmp_path: Path):
    """Test writing benchmark to Parquet."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.parquet'
    bench.write_parquet(path)
    assert path.exists()
    df = pl.read_parquet(path)
    assert len(df) == 1
    assert df['name'][0] == 'test'


def test_write_markdown(tmp_path: Path):
    """Test writing benchmark to Markdown."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.md'
    bench.write_markdown(path)
    assert path.exists()
    content = path.read_text()
    assert 'test' in content
    assert '|' in content  # Markdown table separator


def test_write_markdown_with_string_path(tmp_path: Path):
    """Test writing benchmark to Markdown with string path."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='test'):
        sum(range(100))

    path = str(tmp_path / 'results.md')  # String path, not Path object
    bench.write_markdown(path)
    assert Path(path).exists()


def test_repeat_two():
    """Test benchmark with repeat=2 (minimum for statistics.quantiles)."""
    bench = Benchmark(repeat=2, min_duration_of_repeat=0.01)

    with bench(name='test'):
        sum(range(100))

    d = bench.to_dicts()
    assert len(d[0]['execution_times']) == 2


def test_plot(mocker: MockerFixture):
    """Test plot method calls plotter.show()."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(n=10):
        sum(range(10))

    with bench(n=100):
        sum(range(100))

    mock_show = mocker.patch('matplotlib.figure.Figure.show')
    bench.plot()

    mock_show.assert_called_once()
