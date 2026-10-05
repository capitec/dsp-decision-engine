from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
from numba.extending import register_jitable

__all__ = ["Rounding", "round"]


class Rounding(IntEnum):
    """How `round` treats a value halfway between two results.

    `HALF_UP` goes away from zero, `HALF_EVEN` to the even neighbour, and
    `DOWN` truncates toward zero (every value, not only ties).
    """

    HALF_UP = 0
    HALF_EVEN = 1
    DOWN = 2


_SPLIT = 134217729.0  # 2**27 + 1
# Past 2**53 a double is an even integer, so the scaled value is already whole.
_WHOLE = 2.0 ** 53
# A table, not `10.0 ** n`: pow() would dominate the cost of a call.
_POW10 = tuple(float(10 ** k) for k in range(23))


@register_jitable
def _two_product(a, b):
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
def _pick(q, above, mode):
    # The whole number that q + above rounds to under `mode`, for 0 <= q < 2**53 and |above| at most half an ulp of q.
    f = np.floor(q)
    d = q - f
    if mode == Rounding.DOWN:
        if d == 0 and above < 0:
            f -= 1.0
    elif mode == Rounding.HALF_UP:
        # q whole with ulp 1: q + above is a tie with the neighbour on that side.
        if d > 0.5 or d == 0.5 and above >= 0 or d == 0 and above == 0.5:
            f += 1.0
    elif d > 0.5 or d == 0.5 and (above > 0 or above == 0 and f % 2 != 0):
        f += 1.0
    elif d == 0 and abs(above) == 0.5 and f % 2 != 0:
        f += 1.0 if above > 0 else -1.0
    return f


@register_jitable
def _past_whole(p, e, s, mode):
    # ax * s == p + e exactly with p an even integer beyond 2**53 and e a whole-number-sized residual:
    # round e on its own (p is even, so the parity of p + floor(e) is floor(e)'s), then divide.
    fe = np.floor(e)
    d = e - fe
    g = fe
    if mode == Rounding.HALF_UP:
        if d >= 0.5:
            g += 1.0
    elif mode == Rounding.HALF_EVEN:
        if d > 0.5 or d == 0.5 and fe % 2 != 0:
            g += 1.0
    q = p / s
    a, b = _two_product(q, s)
    # ponytail: q + t is two roundings, off by an ulp only on a halfway case; exact needs big integers.
    return q + (((p - a) - b) + g) / s


@register_jitable
def _scaled(y, mode):
    # The float-arithmetic fallback, for scales 10**n can't represent exactly.
    if mode == Rounding.DOWN:
        return np.floor(y)
    if mode == Rounding.HALF_UP:
        return np.floor(y + 0.5)
    return np.rint(y)


@register_jitable
def round(x, digits=0, mode=Rounding.HALF_UP):
    """Round a float to `digits` decimal places under an explicit rounding mode.

    It rounds the exact value of the double, so `round(2.675, 2)` is `2.67`
    under every mode (the double is 2.67499999...). Negative `digits` round to
    tens, hundreds and so on. NaN and infinities come back unchanged, and the
    sign of zero is kept. The mode is an ordinary int, so one compiled step
    serves every mode.

    Example::

        from decider.ops import decimal
        from decider.ops.decimal import Rounding

        decimal.round(2.5)                              # 3.0
        decimal.round(2.5, mode=Rounding.HALF_EVEN)     # 2.0
        decimal.round(-2.99, 1, Rounding.DOWN)          # -2.9
    """
    if not math.isfinite(x):
        return x
    ax = abs(x)
    if 0 <= digits <= 22:
        s = _POW10[digits]
        if ax >= _WHOLE:
            return x
        p, e = _two_product(ax, s)
        if p >= _WHOLE:
            return math.copysign(_past_whole(p, e, s, mode), x)
        return math.copysign(_pick(p, e, mode) / s, x)
    if -22 <= digits < 0:
        s = _POW10[-digits]
        q = ax / s
        if q >= _WHOLE:
            return x
        p, e = _two_product(q, s)
        # ax - q*s, exactly, so the true quotient is q + that / s.
        return math.copysign(_pick(q, ((ax - p) - e) / s, mode) * s, x)
    # ponytail: beyond 22 digits 10**n isn't exact, so this scales and rounds in floats; exact needs big integers.
    if digits > 0:
        s = 10.0 ** digits
        y = ax * s
        return x if y >= _WHOLE else math.copysign(_scaled(y, mode) / s, x)
    s = 10.0 ** -digits
    return math.copysign(_scaled(ax / s, mode) * s if s < math.inf else 0.0, x)
