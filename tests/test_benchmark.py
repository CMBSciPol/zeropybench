"""Tests for zeropybench with pure Python code."""

from pathlib import Path

import polars as pl
import pytest
from pytest_mock import MockerFixture

from zeropybench import Benchmark, read_benchmark


def test_basic_benchmark():
    """Test basic benchmark functionality."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(name='sum'):
        sum(range(100))

    df = bench.to_dataframe()
    assert len(df) == 1
    assert df['name'][0] == 'sum'
    assert 'execution_times' in df.columns
    assert len(df['execution_times'][0]) == 3


def test_multiple_benchmarks():
    """Test multiple benchmarks with different parameters."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(n=10):
        sum(range(10))

    with bench(n=100):
        sum(range(100))

    df = bench.to_dataframe()
    assert len(df) == 2
    assert df['n'].to_list() == [10, 100]


def test_multidimensional_keywords():
    """Test benchmarks with multiple keyword arguments."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(method='sum', size=100, variant='a'):
        sum(range(100))

    df = bench.to_dataframe()
    assert df['method'][0] == 'sum'
    assert df['size'][0] == 100
    assert df['variant'][0] == 'a'


def test_multiline_code():
    """Test benchmark with multiple statements."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(name='multiline'):
        x = list(range(100))
        y = sum(x)
        z = y * 2  # noqa: F841

    d = bench.to_dicts()
    assert len(d) == 1


def test_execution_times_are_positive():
    """Test that all execution times are positive."""
    bench = Benchmark(repeat=5, min_duration_per_repeat=0.01)

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


def test_empty_benchmark_display_dataframe():
    """Test _to_display_dataframe returns empty dataframe with correct columns."""
    bench = Benchmark()
    df = bench._to_display_dataframe()
    assert len(df) == 0
    assert '± (%)' in df.columns


def test_local_variables():
    """Test benchmark with local variables."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    data = list(range(100))
    multiplier = 2

    with bench(name='local_sum'):
        sum(data) * multiplier

    d = bench.to_dicts()
    assert len(d) == 1
    assert d[0]['name'] == 'local_sum'


def test_local_variables_in_loop():
    """Test benchmark with local variables inside a loop."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    for n in [10, 100]:
        with bench(n=n):
            sum(range(n))

    df = bench.to_dataframe()
    assert len(df) == 2
    assert df['n'].to_list() == [10, 100]


def test_repr():
    """Test __repr__ returns a string table."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(name='test'):
        sum(range(100))

    repr_str = repr(bench)
    assert isinstance(repr_str, str)
    assert 'test' in repr_str
    assert 'median_execution_time' in repr_str
    assert '± (%)' in repr_str


def test_write_csv(tmp_path: Path):
    """Test writing benchmark to CSV with metadata header."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.05)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.csv'
    bench.write_csv(path)
    assert path.exists()
    content = path.read_text()
    assert content.startswith('# repeat = 3\n# min_duration_per_repeat = 0.05\n')
    assert 'test' in content
    assert 'execution_times' in content


def test_write_parquet(tmp_path: Path):
    """Test writing benchmark to Parquet with metadata."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.05)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.parquet'
    bench.write_parquet(path)
    assert path.exists()
    df = pl.read_parquet(path)
    assert len(df) == 1
    assert df['name'][0] == 'test'

    # Check metadata
    metadata = pl.read_parquet_metadata(path)
    assert metadata['repeat'] == '3'
    assert metadata['min_duration_per_repeat'] == '0.05'


def test_write_markdown(tmp_path: Path):
    """Test writing benchmark to Markdown."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

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
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(name='test'):
        sum(range(100))

    path = str(tmp_path / 'results.md')  # String path, not Path object
    bench.write_markdown(path)
    assert Path(path).exists()


def test_repeat_two():
    """Test benchmark with repeat=2 (minimum for statistics.quantiles)."""
    bench = Benchmark(repeat=2, min_duration_per_repeat=0.01)

    with bench(name='test'):
        sum(range(100))

    d = bench.to_dicts()
    assert len(d[0]['execution_times']) == 2


def test_plot(mocker: MockerFixture):
    """Test plot method calls plotter.show()."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    with bench(n=10):
        sum(range(10))

    with bench(n=100):
        sum(range(100))

    mock_show = mocker.patch('matplotlib.figure.Figure.show')
    bench.plot()

    mock_show.assert_called_once()


def test_get_code_from_cmdline_no_c_flag():
    """Test _get_code_from_cmdline raises error when -c flag is not present."""
    import pytest

    # This test only works on Linux
    cmdline_path = Path('/proc/self/cmdline')
    if not cmdline_path.exists():
        return

    # When running via pytest, there's no -c flag
    with pytest.raises(RuntimeError, match='-c.*flag not found'):
        Benchmark._get_code_from_cmdline()


def test_get_code_from_cmdline_not_linux(mocker: MockerFixture):
    """Test _get_code_from_cmdline raises error on non-Linux systems."""
    import pytest

    mocker.patch('zeropybench._benchmark.Path.exists', return_value=False)

    with pytest.raises(RuntimeError, match='only supported on Linux'):
        Benchmark._get_code_from_cmdline()


def test_get_code_from_cmdline_extracts_code(mocker: MockerFixture):
    """Test _get_code_from_cmdline extracts code after -c flag."""
    code = 'print("hello")'
    cmdline = f'python\x00-c\x00{code}\x00'.encode()

    mocker.patch('zeropybench._benchmark.Path.exists', return_value=True)
    mocker.patch('zeropybench._benchmark.Path.read_bytes', return_value=cmdline)

    result = Benchmark._get_code_from_cmdline()
    assert result == code


def test_get_lines_calls_get_code_from_cmdline(mocker: MockerFixture):
    """Test _get_lines calls _get_code_from_cmdline for '<string>' filename."""
    code = 'x = 1\ny = 2'
    mock_cmdline = mocker.patch.object(Benchmark, '_get_code_from_cmdline', return_value=code)
    mocker.patch('zeropybench._benchmark.linecache.getlines', return_value=[])

    bench = Benchmark()
    lines = bench._get_lines('<string>')

    mock_cmdline.assert_called_once()
    assert lines == ['x = 1', 'y = 2']


def test_read_benchmark_csv(tmp_path: Path):
    """Test reading benchmark from CSV with metadata."""
    bench = Benchmark(repeat=5, min_duration_per_repeat=0.03)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.csv'
    bench.write_csv(path)

    loaded = read_benchmark(path)
    assert loaded.repeat == 5
    assert loaded.min_duration_per_repeat == 0.03
    assert loaded.to_dataframe()['name'][0] == 'test'
    assert loaded.to_dataframe()['n'][0] == 100
    assert len(loaded.to_dataframe()['execution_times'][0]) == 5


def test_read_benchmark_parquet(tmp_path: Path):
    """Test reading benchmark from Parquet with metadata."""
    bench = Benchmark(repeat=4, min_duration_per_repeat=0.02)

    with bench(name='test', n=200):
        sum(range(200))

    path = tmp_path / 'results.parquet'
    bench.write_parquet(path)

    loaded = read_benchmark(path)
    assert loaded.repeat == 4
    assert loaded.min_duration_per_repeat == 0.02
    assert loaded.to_dataframe()['name'][0] == 'test'
    assert loaded.to_dataframe()['n'][0] == 200
    assert len(loaded.to_dataframe()['execution_times'][0]) == 4


def test_read_benchmark_unsupported_extension(tmp_path: Path):
    """Test read_benchmark raises error for unsupported file extensions."""
    path = tmp_path / 'results.json'
    path.write_text('{}')

    with pytest.raises(ValueError, match='Unsupported file extension'):
        read_benchmark(path)
