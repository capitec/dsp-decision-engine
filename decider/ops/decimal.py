from __future__ import annotations

import builtins
import math
import operator
from decimal import Decimal as _Std
from enum import IntEnum
from fractions import Fraction
from typing import Annotated

import numpy as np
from numba.extending import register_jitable

from decider._annotations import DecimalSpec

__all__ = ["Currency", "Decimal", "Rounding", "round", "round_float"]


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
def round_float(x, digits=0, mode=Rounding.HALF_UP):
    """Round a float to `digits` decimal places under an explicit rounding mode.

    It rounds the exact value of the double, so `round(2.675, 2)` is `2.67`
    under every mode (the double is 2.67499999...). Negative `digits` round to
    tens, hundreds and so on. NaN and infinities come back unchanged, and the
    sign of zero is kept. The mode is an ordinary int, so one compiled step
    serves every mode.

    Example::

        from decider.ops import decimal
        from decider.ops.decimal import Rounding

        decimal.round_float(2.5)                            # 3.0
        decimal.round_float(2.5, mode=Rounding.HALF_EVEN)   # 2.0
        decimal.round_float(-2.99, 1, Rounding.DOWN)        # -2.9
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


_INT64 = 1 << 63


def _fit(raw: int) -> int:
    if not -_INT64 <= raw < _INT64:
        raise OverflowError(f"{raw} does not fit a Decimal's int64; a compiled step wraps instead")
    return raw


def _div(n: int, d: int) -> int:
    """`n / d` rounded half-even, exactly."""
    q, r = divmod(n, d)
    if abs(2 * r) > abs(d) or abs(2 * r) == abs(d) and q % 2:
        q += 1
    return q


def _co(o):
    """`o` as `(raw, scale)` when it is a Decimal or an int (whole units), else None."""
    if isinstance(o, Decimal):
        return o.raw, o.scale
    if isinstance(o, int):
        return o, 0


_ARITHMETIC = {
    "add": lambda a, sa, b, sb, s: a * 10 ** (s - sa) + b * 10 ** (s - sb),
    "sub": lambda a, sa, b, sb, s: a * 10 ** (s - sa) - b * 10 ** (s - sb),
    "mul": lambda a, sa, b, sb, s: _div(a * b, 10 ** min(sa, sb)),
    "truediv": lambda a, sa, b, sb, s: _div(a * 10 ** (s + sb - sa), b),
}


def _arithmetic(name: str, reflected: bool):
    f = _ARITHMETIC[name]

    def op(self, other):
        c = _co(other)
        if c is None:
            return NotImplemented
        (a, sa), (b, sb) = ((c, (self.raw, self.scale)) if reflected else ((self.raw, self.scale), c))
        if name == "truediv" and b == 0:
            raise ZeroDivisionError("Decimal division by zero")
        units = self.units if not isinstance(other, Decimal) or other.units == self.units else None
        return Decimal.from_raw(_fit(f(a, sa, b, sb, max(sa, sb))), max(sa, sb), units)

    return op


_mul, _truediv = _arithmetic("mul", False), _arithmetic("truediv", False)


def _compare(name: str):
    test = getattr(operator, name)

    def op(self, other):
        c = _co(other)
        if c is not None:
            s = max(self.scale, c[1])
            return test(self.raw * 10 ** (s - self.scale), c[0] * 10 ** (s - c[1]))
        if isinstance(other, float):
            return test(float(self), other)
        if isinstance(other, _Std):
            # decimal.Decimal, as a frame column's value arrives: exact, and False for NaN.
            return other.is_finite() and test(Fraction(self.raw, 10 ** self.scale), Fraction(other)) or name == "ne" and not other.is_finite()
        return NotImplemented

    return op


class Decimal:
    """A fixed-point number: an int64 `raw` count of 10**-`scale` units.

    `Decimal(2.675, 2)`, `Decimal("2.675", 2)` and `Decimal(3)` build one. A
    float is rounded to the scale from its exact binary value (2.67 here), a str
    from its digits (2.68 under half-even). `scale` defaults to 0 for an int and
    to the digits after the point for a str; a float needs it. `units` is a
    display string and never takes part in a comparison or in a compiled step.

    `+ - * /` between Decimals give the larger scale and round half-even once per
    operation. An int counts as whole units, so `Decimal(2.5, 1) + 1` is 3.5, and
    a float multiplier or divisor gives a Decimal at the same scale. A result
    outside int64 raises `OverflowError` here and wraps in a compiled step (a float
    product outside int64 has no defined value there).

    Example::

        a = Decimal("2.50"); b = Decimal("0.05")
        a * b            # Decimal('0.12'), 0.125 rounded half-even
        a + 1            # Decimal('3.50')
        a < 2.6          # True
    """

    __slots__ = ("raw", "scale", "units")

    def __init__(self, value=0, scale: int | None = None, *, units: str | None = None) -> None:
        if isinstance(value, str):
            try:
                value = _Std(value)
            except ArithmeticError:
                raise ValueError(f"not a number: {value!r}") from None
        default = 0
        if isinstance(value, _Std):
            if not value.is_finite():
                raise ValueError(f"{value} is not a finite number")
            default, exact = max(0, -value.as_tuple().exponent), Fraction(value)
        elif isinstance(value, Decimal):
            default, exact = value.scale, Fraction(value.raw, 10 ** value.scale)
            units = value.units if units is None else units
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(f"{value} is not a finite number")
                default = None
            exact = Fraction(value)
        else:
            raise TypeError(f"Decimal takes an int, a float, a str or a Decimal, not {type(value).__name__}")
        if scale is None:
            if default is None:
                raise ValueError("a float needs a scale: Decimal(2.65, scale=2)")
            scale = default
        if not isinstance(scale, int) or not 0 <= scale <= 18:
            raise ValueError(f"scale must be an int from 0 to 18, not {scale!r}")
        object.__setattr__(self, "raw", _fit(builtins.round(exact * 10 ** scale)))
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "units", units)

    def __class_getitem__(cls, key):
        spec = _spec(cls, key)
        return Annotated[cls, spec]

    @classmethod
    def from_raw(cls, raw: int, scale: int, units: str | None = None) -> Decimal:
        """The Decimal whose int64 is `raw` at `scale`: `Decimal.from_raw(265, 2)` is 2.65."""
        self = object.__new__(cls)
        object.__setattr__(self, "raw", _fit(raw))
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "units", units)
        return self

    def __setattr__(self, name, value):
        raise AttributeError("a Decimal is immutable")

    def __reduce__(self):
        return type(self).from_raw, (self.raw, self.scale, self.units)

    def __str__(self) -> str:
        q, r = divmod(abs(self.raw), 10 ** self.scale)
        number = ("-" if self.raw < 0 else "") + (f"{q}.{r:0{self.scale}d}" if self.scale else str(q))
        return f"{self.units} {number}" if self.units else number

    def __repr__(self) -> str:
        return str(self) if self.units else f"{type(self).__name__}('{self}')"

    def __float__(self) -> float:
        return float(self.raw) / float(10 ** self.scale)

    def __int__(self) -> int:
        q = abs(self.raw) // 10 ** self.scale
        return q if self.raw >= 0 else -q

    def __bool__(self) -> bool:
        return self.raw != 0

    def __hash__(self) -> int:
        return hash(Fraction(self.raw, 10 ** self.scale))

    def __neg__(self) -> Decimal:
        return type(self).from_raw(-self.raw, self.scale, self.units)

    def __abs__(self) -> Decimal:
        return type(self).from_raw(abs(self.raw), self.scale, self.units)

    def __mul__(self, other):
        if isinstance(other, float):
            return Decimal.from_raw(builtins.round(float(self.raw) * other), self.scale, self.units)
        return _mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, float):
            return Decimal.from_raw(builtins.round(float(self.raw) / other), self.scale, self.units)
        return _truediv(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, float):
            raise TypeError("float / Decimal is not defined: use float(d)")
        return NotImplemented

    def __round__(self, *_):
        raise TypeError("use decider.ops.decimal.round(d, digits, mode) to round a Decimal")


for _name in ("add", "sub"):
    setattr(Decimal, f"__{_name}__", _arithmetic(_name, False))
    setattr(Decimal, f"__r{_name}__", _arithmetic(_name, True))
for _name in ("eq", "ne", "lt", "le", "gt", "ge"):
    setattr(Decimal, f"__{_name}__", _compare(_name))


def _spec(cls: type, key) -> DecimalSpec:
    if issubclass(cls, Currency):
        if not isinstance(key, str):
            raise TypeError('write Currency["R"] to set display units, or plain Currency for none')
        return DecimalSpec(19, 4, key)
    p, s = key if isinstance(key, tuple) and len(key) == 2 else (None, None)
    if not (isinstance(p, int) and isinstance(s, int)):
        raise TypeError("write Decimal[precision, scale], for example Decimal[18, 2]")
    if not 1 <= p <= 19:
        raise ValueError(f"Decimal[{p}, {s}]: a Decimal is an int64, so its precision is at most 19; "
                         "cast a wider column in the frame tier or keep it as a float")
    if not 0 <= s <= min(p, 18):
        raise ValueError(f"Decimal[{p}, {s}]: the scale must be from 0 to {min(p, 18)}")
    return DecimalSpec(p, s)


class Currency(Decimal):
    """A Decimal with four decimal places (up to 922,337,203,685,477.5807).

    `Currency(2.65)` rounds a float to the scale. `Currency(major, minor)` is
    exact: `minor` counts 0.0001 units, so `Currency(2, 6500)` is 2.65 and
    `Currency(2, 6)` is 2.0006. The sign of `major` applies to the whole value,
    and `minor` stays between 0 and 9999, so write -0.0005 as `-Currency(0, 5)`.
    `units` is a display string shown before the amount; `Currency["R"]` in an
    annotation sets it for a step's input or output. Negation and `abs` stay a
    Currency; the other arithmetic returns a plain Decimal.

    Example::

        Currency(2, 6500, units="R")    # R 2.6500
        Currency(-2, 5)                 # Currency('-2.0005')
    """

    __slots__ = ()

    def __init__(self, major=0, minor: int | None = None, *, units: str | None = None) -> None:
        if minor is None:
            super().__init__(major, 4, units=units)
            return
        if not all(isinstance(v, int) for v in (major, minor)):
            raise TypeError("Currency(major, minor) takes two ints")
        if not 0 <= minor < 10 ** 4:
            raise ValueError(f"minor must be from 0 to 9999 (units of 0.0001), not {minor}")
        super().__init__(0, 4, units=units)
        object.__setattr__(self, "raw", _fit(major * 10 ** 4 + (minor if major >= 0 else -minor)))


def _round_decimal(x: Decimal, digits: int, mode: int) -> Decimal:
    k = x.scale - digits
    if k <= 0:
        return x
    q, r = divmod(abs(x.raw), p := 10 ** k)
    if mode != Rounding.DOWN and (2 * r > p or 2 * r == p and (mode == Rounding.HALF_UP or q % 2)):
        q += 1
    return Decimal.from_raw(q * p * (1 if x.raw >= 0 else -1), x.scale, x.units)


def round(x, digits=0, mode=Rounding.HALF_UP):
    """Round a float or a Decimal to `digits` decimal places under an explicit rounding mode.

    A Decimal keeps its scale: `digits` below the scale zeroes the lower places,
    so `round(Decimal("2.675"), 2)` is 2.680. A float is rounded as in
    `round_float`. A Decimal result outside int64 raises `OverflowError` here and
    wraps in a compiled step.

    Example::

        from decider.ops import decimal
        from decider.ops.decimal import Decimal, Rounding

        decimal.round(Decimal("2.675"), 2)                          # Decimal('2.680')
        decimal.round(Decimal("2.675"), 2, Rounding.HALF_EVEN)      # Decimal('2.680')
        decimal.round(Decimal("2.5"), 0, Rounding.HALF_EVEN)        # Decimal('2.0')
    """
    return _round_decimal(x, digits, Rounding(mode)) if isinstance(x, Decimal) else round_float(x, digits, mode)
