import polars as pl
import pytest

from zeropybench._units import get_optimal_time_units, to_units


@pytest.mark.parametrize(
    'values, expected_units',
    [
        ([10.0, 1.4], 's'),
        ([10.0e-3, 1.4e-3], 'ms'),
        ([10e-6, 1.4e-6], 'µs'),
        ([10e-9, 1.4e-9], 'ns'),
    ],
)
def test_get_optimal_time_units(values: list[float], expected_units: str) -> None:
    actual_units = get_optimal_time_units(values)
    assert actual_units == expected_units


def test_get_optimal_time_units_empty() -> None:
    assert get_optimal_time_units([]) == 's'


@pytest.mark.parametrize(
    'value, units, expected',
    [
        (1.0, 's', 1.0),
        (1.0, 'ms', 1000.0),
        (1.0, 'µs', 1_000_000.0),
        (1.0, 'ns', 1_000_000_000.0),
        (0.001, 'ms', 1.0),
        (1e-6, 'µs', 1.0),
    ],
)
def test_to_units_float(value: float, units: str, expected: float) -> None:
    result = to_units(value, units)  # type: ignore[arg-type]
    assert result == expected


def test_to_units_polars_expr() -> None:
    df = pl.DataFrame({'time': [1.0, 2.0, 3.0]})
    result = df.select(to_units(pl.col('time'), 'ms'))
    expected = pl.DataFrame({'time': [1000.0, 2000.0, 3000.0]})
    assert result.equals(expected)
