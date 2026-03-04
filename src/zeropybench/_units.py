from collections.abc import Iterable
from typing import Literal, overload

import polars as pl

TimeUnitType = Literal['ns', 'µs', 'ms', 's']
TIME_UNITS_MULTIPLIER: dict[TimeUnitType, float] = {
    's': 1.0,
    'ms': 1000.0,
    'µs': 1000_000.0,
    'ns': 1_000_000_000.0,
}


def get_optimal_time_units(values_in_s: Iterable[float]) -> TimeUnitType:
    """Return the time units that minimize character count for 3 significant figures.

    If the input is empty, 's' (seconds) is returned.

    Args:
        values_in_s: The values in seconds used to infer the optimal time units.
    """

    def char_count(units: TimeUnitType) -> int:
        total = 0
        for value_in_s in values_in_s:
            value = to_units(value_in_s, units)
            total += len(f'{value:.3g}')
        return total

    return min(TIME_UNITS_MULTIPLIER.keys(), key=char_count)


@overload
def to_units(value: float, units: TimeUnitType) -> float: ...


@overload
def to_units(value: pl.Expr, units: TimeUnitType) -> pl.Expr: ...


def to_units(value: float | pl.Expr, units: TimeUnitType) -> float | pl.Expr:
    multiplier = TIME_UNITS_MULTIPLIER[units]
    return value * multiplier
