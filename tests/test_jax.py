from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

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
