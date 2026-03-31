"""Tests for zeropybench I/O functionality."""

from pathlib import Path

import polars as pl
import pytest

from zeropybench import Benchmark, read_benchmark


def test_write_csv(tmp_path: Path):
    """Test writing benchmark to CSV with metadata header."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.05)

    with bench(name='test', n=100):
        sum(range(100))

    path = tmp_path / 'results.csv'
    bench.write_csv(path)
    assert path.exists()
    content = path.read_text()
    assert '# repeat = 3' in content
    assert '# min_duration_per_repeat = 0.05' in content
    assert 'test' in content
    assert 'execution_times' in content
    # The file should be valid CSV (no comment lines before headers)
    df = pl.read_csv(path)
    assert len(df) == 1


def test_write_csv_with_string_path(tmp_path: Path):
    """Test writing benchmark to CSV with string path."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.05)

    with bench(name='test'):
        sum(range(100))

    path = str(tmp_path / 'results.csv')  # String path, not Path object
    bench.write_csv(path)
    assert Path(path).exists()


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


def test_read_benchmark_csv_with_string_path(tmp_path: Path):
    """Test reading benchmark from CSV with string path."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.05)

    with bench(name='test'):
        sum(range(100))

    path = tmp_path / 'results.csv'
    bench.write_csv(path)

    loaded = read_benchmark(str(path))  # String path, not Path object
    assert loaded.repeat == 3


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


def test_read_csv_with_leading_comment_lines(tmp_path: Path):
    """Test that leading comment lines are skipped when reading a CSV."""
    path = tmp_path / 'results.csv'
    path.write_text('# some comment\nname,median_execution_time\ntest,0.1\n')
    loaded = read_benchmark(path)
    assert loaded.to_dataframe()['name'][0] == 'test'


def test_read_benchmark_unsupported_extension(tmp_path: Path):
    """Test read_benchmark raises error for unsupported file extensions."""
    path = tmp_path / 'results.json'
    path.write_text('{}')

    with pytest.raises(ValueError, match='Unsupported file extension'):
        read_benchmark(path)
