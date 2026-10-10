"""Where a float meets a declared `int`: a whole number crosses, anything else is refused, never truncated."""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from decikit.exceptions import NotIntegralError


def refuse(subject: str, value: object) -> NotIntegralError:
    return NotIntegralError(f"{subject} is declared int but got {value!r}: a float must be a whole number "
                            "(7.0), and is never truncated; round it in a step, or declare the value float")


def int64_array(values: list, subject: Callable[[int], str]) -> np.ndarray:
    """`values` as int64, refusing a float that is not a whole number; `subject(i)` names value `i` in the message.

    Example::

        int64_array([7, 8.0], lambda i: "input 'n'")      # array([7, 8])
        int64_array([7.5], lambda i: "input 'n'")         # NotIntegralError
    """
    try:
        out = np.array(values, np.int64)
    except (ValueError, OverflowError):
        # NaN and infinity stop numpy itself; a plain overflow is not ours to rename.
        for i, x in enumerate(values):
            if isinstance(x, float) and not x.is_integer():
                raise refuse(subject(i), x) from None
        raise
    if out.tolist() != values:
        for i, x in enumerate(values):
            if isinstance(x, (float, np.floating)) and x != out[i]:
                raise refuse(subject(i), x)
    return out


def check_series(series: pl.Series) -> None:
    """Refuse a float column's NaN, infinity or fractional value, before it is cast to an integer.

    Example::

        check_series(pl.Series("n", [7.0, 7.5]))   # NotIntegralError: input 'n' (row 1) ...
    """
    bad = series.is_not_null() & ~(series.is_finite() & (series == series.floor()))
    if bad.any():
        row = bad.arg_true()[0]
        raise refuse(f"input '{series.name}'" + (f" (row {row})" if len(series) > 1 else ""), series[row])
