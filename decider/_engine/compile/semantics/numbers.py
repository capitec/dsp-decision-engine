# Float arithmetic in kernels with CPython's answers; a CPU-target overload
# outranks numba's generic builtin.
#
# `round(x, ndigits)`: numba rounds the scaled float (`2.675 * 100` is `267.5`,
# so `2.68`); CPython rounds the exact value of the double (`2.67499999...`, so
# `2.67`), half to even only on exact ties (`0.125` gives `0.12`).
#
# `x ** n` (float x, int n): numba multiplies by squaring, which drifts a few
# ulps from the libm `pow` CPython calls, e.g. in `(1 + r) ** 12`.
from __future__ import annotations

import operator

from numba.core import types
from numba.extending import overload, register_jitable

from decider.ops import decimal
from decider.ops.decimal import Rounding


@register_jitable
def cpython_round(x: float, ndigits: int) -> float:
    return decimal.round(x, ndigits, Rounding.HALF_EVEN)


@overload(round, target="cpu")
def _round(x, ndigits):
    if isinstance(x, types.Float) and isinstance(ndigits, types.Integer):
        return lambda x, ndigits: cpython_round(x, ndigits)


def _float_pow(x, n):
    if isinstance(x, types.Float) and isinstance(n, types.Integer):
        # ponytail: LLVM still folds a literal `x ** 2` into `x * x`, which misses libm's pow by an ulp now and then.
        return lambda x, n: x ** float(n)


overload(operator.pow, target="cpu")(_float_pow)
overload(operator.ipow, target="cpu")(_float_pow)
