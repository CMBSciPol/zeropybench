"""Tests for zeropybench with pure Python code."""

from zeropybench import Benchmark


def test_basic_benchmark():
    """Test basic benchmark functionality."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(name='sum'):
        sum(range(100))

    df = bench.as_dataframe()
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

    df = bench.as_dataframe()
    assert len(df) == 2
    assert df['n'].to_list() == [10, 100]


def test_multidimensional_keywords():
    """Test benchmarks with multiple keyword arguments."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    with bench(method='sum', size=100, variant='a'):
        sum(range(100))

    df = bench.as_dataframe()
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

    df = bench.as_dataframe()
    assert len(df) == 1


def test_execution_times_are_positive():
    """Test that all execution times are positive."""
    bench = Benchmark(repeat=5, min_duration_of_repeat=0.01)

    with bench(name='test'):
        sum(range(1000))

    df = bench.as_dataframe()
    times = df['execution_times'][0]
    assert all(t > 0 for t in times)


def test_empty_benchmark():
    """Test empty benchmark returns empty dataframe."""
    bench = Benchmark()
    df = bench.as_dataframe()
    assert len(df) == 0


def test_local_variables():
    """Test benchmark with local variables."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    data = list(range(100))
    multiplier = 2

    with bench(name='local_sum'):
        sum(data) * multiplier

    df = bench.as_dataframe()
    assert len(df) == 1
    assert df['name'][0] == 'local_sum'


def test_local_variables_in_loop():
    """Test benchmark with local variables inside a loop."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    for n in [10, 100]:
        with bench(n=n):
            sum(range(n))

    df = bench.as_dataframe()
    assert len(df) == 2
    assert df['n'].to_list() == [10, 100]
