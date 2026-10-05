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

import math
import operator

import numpy as np
from numba.core import types
from numba.extending import overload, register_jitable

_SPLIT = 134217729.0  # 2**27 + 1
# Past 2**53 a double is an even integer, so the scaled value is already whole.
_WHOLE = 2.0 ** 53
# A table, not `10.0 ** n`: pow() would dominate the cost of a call.
_POW10 = tuple(float(10 ** k) for k in range(23))


@register_jitable
def _two_product(a: float, b: float) -> tuple[float, float]:
    # Dekker: p + e == a * b exactly, without an fma.
    p = a * b
    c = _SPLIT * a
    ah = c - (c - a)
    al = a - ah
    c = _SPLIT * b
    bh = c - (c - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@register_jitable
def _half_even(q: float, above: float) -> float:
    # The integer nearest q + above, ties to even, where |above| is at most half an ulp of q.
    f = np.floor(q)
    d = q - f
    if d > 0.5 or d == 0.5 and (above > 0 or above == 0 and f % 2 != 0):
        f += 1.0
    elif d == 0 and abs(above) == 0.5 and f % 2 != 0:
        # q is whole and its ulp is 1: q + above is a tie with the neighbour on that side.
        f += math.copysign(1.0, above)
    return f


@register_jitable
def cpython_round(x: float, ndigits: int) -> float:
    if not math.isfinite(x):
        return x
    if 0 <= ndigits <= 22:
        s = _POW10[ndigits]
        p, e = _two_product(x, s)
        if abs(p) >= _WHOLE:
            return x
        return math.copysign(_half_even(p, e) / s, x)
    if -22 <= ndigits < 0:
        s = _POW10[-ndigits]
        q = x / s
        if abs(q) >= _WHOLE:
            return x
        p, e = _two_product(q, s)
        # x - q*s, exactly, so the true quotient is q + that / s.
        return math.copysign(_half_even(q, ((x - p) - e) / s) * s, x)
    # ponytail: beyond 22 digits 10**n isn't exact, so this scales and rounds like numba; exact needs big integers.
    if ndigits > 0:
        s = 10.0 ** ndigits
        y = x * s
        return x if abs(y) >= _WHOLE else math.copysign(np.rint(y) / s, x)
    s = 10.0 ** -ndigits
    return math.copysign(np.rint(x / s) * s if s < math.inf else 0.0, x)


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
