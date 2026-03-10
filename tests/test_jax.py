from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest_mock import MockerFixture

from zeropybench import Benchmark
from zeropybench._jax import CodeASTParser


@pytest.mark.parametrize(
    'var_names, expected',
    [
        (['jax_array'], True),  # jax.Array
        (['jitted_func'], True),  # jitted function
        (['jax_array', 'numpy_array'], True),  # mixed, has jax.Array
        (['jitted_func', 'numpy_array'], True),  # mixed, has jitted function
        (['numpy_array'], False),  # numpy array only
        (['python_list'], False),  # python list only
        (['regular_func'], False),  # non-jitted function
        (['numpy_array', 'python_list'], False),  # no jax
        ([], False),  # no variables
    ],
)
def test_is_jax_context(var_names: list[str], expected: bool) -> None:
    """Test is_jax_context detection of JAX arrays and jitted functions."""
    # Define all possible variables
    all_vars = {
        'jax_array': jnp.array([1, 2, 3]),
        'numpy_array': np.array([1, 2, 3]),
        'python_list': [1, 2, 3],
        'jitted_func': jax.jit(lambda x: x + 1),
        'regular_func': lambda x: x + 1,
    }

    # Select variables for this test
    globals_ = {name: all_vars[name] for name in var_names}

    # Build code that uses these variables
    code = ' + '.join(var_names) if var_names else '1 + 1'

    parser = CodeASTParser.from_code(code, globals_)
    assert parser.is_jax_context() == expected


@pytest.mark.parametrize(
    'pytree, expected',
    [
        # Standard pytrees containing jax.Array
        ((jnp.array([1]), jnp.array([2])), True),  # tuple of jax.Array
        ([jnp.array([1]), jnp.array([2])], True),  # list of jax.Array
        ({'a': jnp.array([1]), 'b': jnp.array([2])}, True),  # dict of jax.Array
        # Mixed pytrees
        ((jnp.array([1]), np.array([2])), True),  # tuple with jax and numpy
        ({'jax': jnp.array([1]), 'numpy': np.array([2])}, True),  # dict with jax and numpy
        # No jax.Array
        ((np.array([1]), np.array([2])), False),  # tuple of numpy arrays
        ([1, 2, 3], False),  # plain list
        ({'a': 1, 'b': 2}, False),  # dict of ints
        ((), False),  # empty tuple
        ([], False),  # empty list
        ({}, False),  # empty dict
    ],
    ids=[
        'tuple_of_jax',
        'list_of_jax',
        'dict_of_jax',
        'tuple_mixed',
        'dict_mixed',
        'tuple_numpy_only',
        'plain_list',
        'dict_ints',
        'empty_tuple',
        'empty_list',
        'empty_dict',
    ],
)
def test_is_jax_context_pytrees(pytree: object, expected: bool) -> None:
    """Test is_jax_context detection of JAX arrays in pytrees."""
    globals_ = {'pytree': pytree}
    parser = CodeASTParser.from_code('pytree', globals_)
    assert parser.is_jax_context() == expected


def test_is_jax_context_registered_dataclass() -> None:
    """Test is_jax_context with dataclass registered via @jax.tree_util.register_dataclass."""
    from dataclasses import dataclass

    @jax.tree_util.register_dataclass
    @dataclass
    class State:
        position: jax.Array
        velocity: jax.Array

    state = State(position=jnp.array([1.0, 2.0]), velocity=jnp.array([0.1, 0.2]))
    globals_ = {'state': state}
    parser = CodeASTParser.from_code('state', globals_)
    assert parser.is_jax_context() is True


def test_is_jax_context_dataclass_without_jax_arrays() -> None:
    """Test is_jax_context with dataclass that doesn't contain JAX arrays."""
    from dataclasses import dataclass

    @jax.tree_util.register_dataclass
    @dataclass
    class Config:
        learning_rate: float
        batch_size: int

    config = Config(learning_rate=0.01, batch_size=32)
    globals_ = {'config': config}
    parser = CodeASTParser.from_code('config', globals_)
    assert parser.is_jax_context() is False


def test_is_jax_context_no_jax(mocker: MockerFixture) -> None:
    """Test is_jax_context returns False when JAX is not imported."""
    mocker.patch.dict('sys.modules', {'jax': None})

    globals_ = {'x': [1, 2, 3]}
    parser = CodeASTParser.from_code('x', globals_)
    assert parser.is_jax_context() is False


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
    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    assert '@jax.jit' in setup_code
    assert 'def __bench_func(x, y):' in setup_code
    assert args == ['x', 'y']
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    assert_array_equal(__bench_func(x, y), jnp.array([[4, 5], [5, 6]]))


def test_subscript_call() -> None:
    """Test that subscript call funcs[0](x) is wrapped (not a simple call)."""
    x = jnp.array([1, 2])

    def add_one(a):
        return a + 1

    funcs = [add_one]
    globals_ = {'x': x, 'funcs': funcs}
    parser = CodeASTParser.from_code('funcs[0](x)', globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    # Not a simple call, so it's wrapped in a jitted function
    assert setup_code == '@jax.jit\ndef __bench_func(funcs, x):\n    return funcs[0](x)'


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

    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    if do_jit:
        assert setup_code == '__bench_func = func'
    else:
        assert setup_code == '__bench_func = jax.jit(func)'
    assert args == ['x', 'y']
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
        'op.add(x, y)',
        'z = op.add(x, y)',
        'z: jax.Array = op.add(x, y)',
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_method_call(code: str, do_jit: bool) -> None:
    """Test method call obj.method(x, y)."""
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])

    class Op:
        def add(self, a, b):
            return a + b

    op = Op()
    if do_jit:
        op.add = jax.jit(op.add)

    globals_ = {'x': x, 'y': y, 'op': op}
    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()

    if do_jit:
        assert setup_code == '__bench_func = op.add'
    else:
        assert setup_code == '__bench_func = jax.jit(op.add)'
    assert args == ['x', 'y']
    __bench_func = globals_.get('__bench_func')
    assert isinstance(__bench_func, Callable)
    assert hasattr(__bench_func, 'lower')
    if do_jit:
        # Already jitted method is reused directly
        assert __bench_func is op.add

    assert_array_equal(__bench_func(x, y), jnp.array([4, 6]))


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

    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    assert '@jax.jit' in setup_code
    assert 'def __bench_func(x, y):' in setup_code
    assert args == ['x', 'y']
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
    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    assert '@jax.jit' in setup_code
    assert 'def __bench_func(x, y):' in setup_code
    assert args == ['x', 'y']
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
    globals_ = {'x': x}
    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    assert '@jax.jit' in setup_code
    assert 'def __bench_func(x):' in setup_code
    assert args == ['x']
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
    parser = CodeASTParser.from_code(code, globals_)
    setup_code, args, globals_ = parser.transform_jax_code()
    assert '@jax.jit' in setup_code
    assert 'def __bench_func(x, y):' in setup_code
    assert args == ['x', 'y']
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
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

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


def test_benchmark_jax_jitted_function(capsys: pytest.CaptureFixture):
    """Test JAX benchmark with already jitted function."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01, verbose=True)

    @jax.jit
    def add_arrays(x, y):
        return x + y

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])

    with bench():
        add_arrays(x, y)
    captured = capsys.readouterr().err
    assert (
        captured
        == """\
Setup code:
    __bench_func = add_arrays
Benchmarked code:
    __bench_func(x, y).block_until_ready()
"""
    )


def test_benchmark_jax_multiple_outputs(capsys: pytest.CaptureFixture):
    """Test JAX benchmark with multiple outputs (uses jax.block_until_ready)."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01, verbose=True)

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])

    with bench():
        a = x + y
        b = a * 2  # noqa: F841

    captured = capsys.readouterr().err
    assert (
        captured
        == """\
Setup code:
    @jax.jit
    def __bench_func(x, y):
        a = x + y
        b = a * 2
        return (a, b)
Benchmarked code:
    jax.block_until_ready(__bench_func(x, y))
"""
    )


def test_benchmark_jax_verbose(capsys: pytest.CaptureFixture) -> None:
    """Test JAX benchmark verbose mode prints setup and benchmarked code."""
    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01, verbose=True)

    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])

    with bench(method='add'):
        x + y

    captured = capsys.readouterr().err
    assert 'Setup code:' in captured
    assert '__bench_func' in captured
    assert 'Benchmarked code:' in captured


def test_benchmark_jax_plot(tmp_path: Path):
    """Test JAX benchmark plotting (excludes JAX-specific columns from legend)."""
    import matplotlib

    matplotlib.use('Agg')

    bench = Benchmark(repeat=3, min_duration_per_repeat=0.01)

    for n in [10, 100]:
        x = jnp.ones(n)
        y = jnp.ones(n)
        with bench(n=n):
            x + y

    path = tmp_path / 'results.png'
    bench.write_plot(path)
    assert path.exists()
