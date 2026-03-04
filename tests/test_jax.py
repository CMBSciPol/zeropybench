from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from zeropybench import Benchmark
from zeropybench._jax import CodeASTParser


@pytest.mark.parametrize(
    'code',
    [
        'x[:, None] + y',
        'z = x[:, None] + y',
        'z: jax.Array = x[:, None] + y',
    ],
)
def test_expr(code: str) -> None:
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    globals_ = {'x': x, 'y': y}
    parser = CodeASTParser.from_code(code)
    code, globals_ = parser.transform_jax_code({}, globals_)
    assert code == '__rv = __bench_func(x, y)\njax.block_until_ready(__rv)'
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    assert_array_equal(__bench_func(x, y), jnp.array([[4, 5], [5, 6]]))


@pytest.mark.parametrize(
    'code',
    [
        'func(x, y)',
        'z = func(x, y)',
        'z: jax.Array = func(x, y)',
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_func(code: str, do_jit: bool) -> None:
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])

    def func(a, b):
        return a[:, None] + b

    if do_jit:
        func = jax.jit(func)

    globals_ = {'x': x, 'y': y, 'func': func}

    parser = CodeASTParser.from_code(code)
    code, globals_ = parser.transform_jax_code({}, globals_)
    assert code == '__rv = __bench_func(x, y)\njax.block_until_ready(__rv)'
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    if do_jit:
        # Already jitted function is reused directly
        assert __bench_func is func

    assert_array_equal(__bench_func(x, y), jnp.array([[4, 5], [5, 6]]))


@pytest.mark.parametrize(
    'code',
    [
        'func(x+1, y)',
        'z = func(x+1, y)',
        'z: jax.Array = func(x+1, y)',
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_func_with_argument_as_expr(code: str, do_jit: bool) -> None:
    x = jnp.array([0, 1])
    y = jnp.array([3, 4])

    def func(a, b):
        return a[:, None] + b

    if do_jit:
        func = jax.jit(func)

    globals_ = {'x': x, 'y': y, 'func': func}

    parser = CodeASTParser.from_code(code)
    code, globals_ = parser.transform_jax_code({}, globals_)
    assert code == '__rv = __bench_func(x, y)\njax.block_until_ready(__rv)'
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')

    assert_array_equal(__bench_func(x, y), jnp.array([[4, 5], [5, 6]]))


def test_compound_statement_if() -> None:
    """Test compound statement: if/else."""
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    code = """\
if True:
    z = x + y
else:
    z = x - y
"""
    globals_ = {'x': x, 'y': y}
    parser = CodeASTParser.from_code(code)
    new_code, globals_ = parser.transform_jax_code({}, globals_)
    assert new_code == '__rv = __bench_func(x, y)\njax.block_until_ready(__rv)'
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    # Single assigned variable: returns value directly (not tuple)
    assert_array_equal(__bench_func(x, y), jnp.array([4, 6]))


def test_compound_statement_for() -> None:
    """Test compound statement: for loop."""
    x = jnp.array([1, 2])
    code = """\
result = x
for _ in range(3):
    result = result + x
"""
    globals_ = {'x': x, 'range': range}
    parser = CodeASTParser.from_code(code)
    new_code, globals_ = parser.transform_jax_code({}, globals_)
    assert new_code == '__rv = __bench_func(x)\njax.block_until_ready(__rv)'
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    # Single assigned variable: returns value directly (not tuple)
    assert_array_equal(__bench_func(x), jnp.array([4, 8]))


def test_multiple_statements() -> None:
    """Test multiple statements."""
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    code = 'a = x + y\nb = a * 2'
    globals_ = {'x': x, 'y': y}
    parser = CodeASTParser.from_code(code)
    new_code, globals_ = parser.transform_jax_code({}, globals_)
    assert new_code == '__rv = __bench_func(x, y)\njax.block_until_ready(__rv)'
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    result = __bench_func(x, y)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert_array_equal(result[0], jnp.array([4, 6]))  # a
    assert_array_equal(result[1], jnp.array([8, 12]))  # b


def test_benchmark_jax_context():
    """Test full JAX benchmark context with HLO and compilation time."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])

    with bench(method='add'):
        x + y

    report = bench.to_dicts()[0]

    # Check JAX-specific fields are present
    assert 'first_execution_time' in report
    assert 'compilation_time' in report
    assert 'hlo' in report

    # Check types
    assert isinstance(report['first_execution_time'], float)
    assert isinstance(report['compilation_time'], float)
    assert isinstance(report['hlo'], str)

    # HLO should contain module info
    assert 'HloModule' in report['hlo'] or 'module' in report['hlo'].lower()


def test_benchmark_jax_jitted_function():
    """Test JAX benchmark with already jitted function."""
    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    @jax.jit
    def add_arrays(x, y):
        return x + y

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])

    with bench(method='jitted_add'):
        add_arrays(x, y)

    report = bench.to_dicts()[0]
    assert 'hlo' in report
    assert report['method'] == 'jitted_add'


def test_benchmark_jax_plot(tmp_path):
    """Test JAX benchmark plotting (excludes JAX-specific columns from legend)."""
    import matplotlib

    matplotlib.use('Agg')

    bench = Benchmark(repeat=3, min_duration_of_repeat=0.01)

    for n in [10, 100]:
        x = jnp.ones(n)
        y = jnp.ones(n)
        with bench(n=n):
            x + y

    path = tmp_path / 'results.png'
    bench.write_plot(path)
    assert path.exists()
