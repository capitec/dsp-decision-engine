# Operators on `DecimalType` (an int64 count of 10**-scale units) that answer as a Python
# `decikit.ops.decimal.Decimal` does, except that int64 overflow wraps. `+ - * /` between
# decimals give the larger scale and round half-even once; an int is whole units. Every overload
# is keyed on `DecimalType`, so plain ints and floats are untouched. The 128-bit work is
# hand-rolled in uint64 because LLVM's i128 division calls `__udivti3`, which aborts the process.
from __future__ import annotations

import functools
import operator
from functools import lru_cache

import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload, overload_attribute, register_jitable

from decikit._engine.compile.decimal_type import DecimalType
from decikit.ops import decimal as ops
from decikit.ops.decimal import Currency, Decimal, Rounding, round_float

_U = np.uint64
_M32, _S32, _ZERO, _ONE = _U(0xFFFFFFFF), _U(32), _U(0), _U(1)
_POW10 = np.array([10 ** k for k in range(20)], dtype=np.uint64)


@intrinsic
def raw(typingctx, x):
    """A decimal or an int as its int64 (an int is whole units, so it is its own raw); an optional must not be None."""
    if isinstance(_strip(x), (DecimalType, types.Integer)):
        def codegen(context, builder, sig, args):
            v, t = args[0], sig.args[0]
            if isinstance(t, types.Optional):
                v, t = context.cast(builder, v, t, t.type), t.type
            return v if isinstance(t, DecimalType) else context.cast(builder, v, t, types.int64)

        return types.int64(x), codegen


@lru_cache(maxsize=None)
def wrapper(scale: int):
    """An intrinsic that types an int64 as a decimal of `scale`."""
    @intrinsic
    def wrap(typingctx, i):
        if isinstance(i, types.Integer):
            return DecimalType(scale)(i), lambda context, builder, sig, args: context.cast(builder, args[0], sig.args[0], types.int64)

    return wrap


@register_jitable
def _mag(x):
    u = _U(x)
    return (~u) + _ONE if x < 0 else u


@register_jitable
def _signed(q, negative):
    r = np.int64(q)
    return -r if negative else r


@register_jitable
def _mul64(a, b):
    a0 = a & _M32
    a1 = a >> _S32
    b0 = b & _M32
    b1 = b >> _S32
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    mid = (p00 >> _S32) + (p01 & _M32) + (p10 & _M32)
    return a1 * b1 + (p01 >> _S32) + (p10 >> _S32) + (mid >> _S32), (p00 & _M32) | (mid << _S32)


@register_jitable
def _nlz(x):
    n = _ZERO
    if x <= _U(0x00000000FFFFFFFF):
        n += _U(32)
        x <<= _U(32)
    if x <= _U(0x0000FFFFFFFFFFFF):
        n += _U(16)
        x <<= _U(16)
    if x <= _U(0x00FFFFFFFFFFFFFF):
        n += _U(8)
        x <<= _U(8)
    if x <= _U(0x0FFFFFFFFFFFFFFF):
        n += _U(4)
        x <<= _U(4)
    if x <= _U(0x3FFFFFFFFFFFFFFF):
        n += _U(2)
        x <<= _U(2)
    if x <= _U(0x7FFFFFFFFFFFFFFF):
        n += _ONE
    return n


@register_jitable
def _divlu(u1, u0, v):
    # (u1:u0) / v for u1 < v, as (quotient, remainder): Knuth's algorithm D in two 32-bit digits.
    if u1 == _ZERO:
        q = u0 // v
        return q, u0 - q * v
    b = _ONE << _S32
    s = _nlz(v)
    v = v << s
    vn1 = v >> _S32
    vn0 = v & _M32
    un32 = (u1 << s) | ((u0 >> (_U(64) - s)) if s > _ZERO else _ZERO)
    un10 = u0 << s
    un1 = un10 >> _S32
    un0 = un10 & _M32
    q1 = un32 // vn1
    rhat = un32 - q1 * vn1
    while q1 >= b or q1 * vn0 > b * rhat + un1:
        q1 -= _ONE
        rhat += vn1
        if rhat >= b:
            break
    un21 = un32 * b + un1 - q1 * v
    q0 = un21 // vn1
    rhat = un21 - q0 * vn1
    while q0 >= b or q0 * vn0 > b * rhat + un0:
        q0 -= _ONE
        rhat += vn1
        if rhat >= b:
            break
    return q1 * b + q0, (un21 * b + un0 - q0 * v) >> s


@register_jitable
def _round_up(q, r, v):
    # Half-even: whether the quotient q with remainder r of a division by v goes up by one.
    rest = v - r
    return r > rest or (r == rest and (q & _ONE) == _ONE)


@register_jitable
def _mul_scaled(x, y, k):
    # x * y / 10**k, half-even; the low 64 bits of the exact result.
    hi, lo = _mul64(_mag(x), _mag(y))
    v = _POW10[k]
    q, r = _divlu(hi % v, lo, v)
    if _round_up(q, r, v):
        q += _ONE
    return _signed(q, (x < 0) != (y < 0))


@register_jitable
def _div_scaled(x, y, e):
    # x * 10**e / y, half-even, a chunk of at most 19 digits at a time so no step needs more than 128 bits.
    if y == 0:
        raise ZeroDivisionError("Decimal division by zero")
    n = _mag(x)
    v = _mag(y)
    q = n // v
    r = n - q * v
    while e > 0:
        c = min(e, 19)
        hi, lo = _mul64(r, _POW10[c])
        qc, r = _divlu(hi, lo, v)
        q = q * _POW10[c] + qc
        e -= c
    if _round_up(q, r, v):
        q += _ONE
    return _signed(q, (x < 0) != (y < 0))


@register_jitable
def _div_pow10(x, k):
    # x / 10**k, half-even.
    d = _POW10[k]
    m = _mag(x)
    q = m // d
    if _round_up(q, m - q * d, d):
        q += _ONE
    return _signed(q, x < 0)


@register_jitable
def _compare(a, fa, b, fb):
    # -1, 0 or 1 for a * fa against b * fb, exactly.
    na = a < 0
    if na != (b < 0):
        return -1 if na else 1
    ha, la = _mul64(_mag(a), fa)
    hb, lb = _mul64(_mag(b), fb)
    if ha != hb:
        c = -1 if ha < hb else 1
    elif la != lb:
        c = -1 if la < lb else 1
    else:
        return 0
    return -c if na else c


@register_jitable
def _round_raw(x, k, mode):
    # x rounded to a multiple of 10**k (k > 0) under `mode`.
    if k > 19:
        return np.int64(0)
    d = _POW10[k]
    m = _mag(x)
    q = m // d
    r = m - q * d
    if mode != Rounding.DOWN and (r > d - r or r == d - r and (mode == Rounding.HALF_UP or (q & _ONE) == _ONE)):
        q += _ONE
    return _signed(q * d, x < 0)


def _strip(t):
    # The `x + y` of an `x: Currency | None` reads x as a Currency, as numba does an optional float.
    return t.type if isinstance(t, types.Optional) else t


def _unwrapping(f):
    @functools.wraps(f)
    def typed(*args):
        return f(*map(_strip, args))

    return typed


def _scale(t) -> int:
    return t.scale if isinstance(t, DecimalType) else 0


def _number(t) -> bool:
    return isinstance(t, (DecimalType, types.Integer))


def _either(a, b) -> bool:
    return isinstance(a, DecimalType) or isinstance(b, DecimalType)


def _undefined(symbol: str, a, b, advice: str = "") -> TypingError:
    return TypingError(f"{a} {symbol} {b} is not defined for a Decimal{': ' + advice if advice else ''}")


def _wide(a, b) -> int:
    return max(_scale(a), _scale(b))


def _additive(symbol, op):
    def overloaded(a, b):
        if not _either(a, b):
            return None
        if not (_number(a) and _number(b)):
            raise _undefined(symbol, a, b, "an int counts as whole units, a float is not allowed")
        s = _wide(a, b)
        ka, kb, w = 10 ** (s - _scale(a)), 10 ** (s - _scale(b)), wrapper(s)
        return lambda a, b: w(op(raw(a) * ka, raw(b) * kb))

    return overloaded


def _multiplicative(symbol, divide: bool):
    def overloaded(a, b):
        if not _either(a, b):
            return None
        if divide and not isinstance(a, DecimalType):
            raise _undefined(symbol, a, b, "use float(d) for a ratio")
        d = a if isinstance(a, DecimalType) else b
        w = wrapper(d.scale)
        if isinstance(a, types.Float) or isinstance(b, types.Float):
            if divide:
                return lambda a, b: w(np.int64(np.rint(float(raw(a)) / b)))
            return (lambda a, b: w(np.int64(np.rint(float(raw(a)) * b)))) if isinstance(b, types.Float) else (
                lambda a, b: w(np.int64(np.rint(float(raw(b)) * a))))
        if not (_number(a) and _number(b)):
            raise _undefined(symbol, a, b)
        s = _wide(a, b)
        w = wrapper(s)
        if divide:
            e = s + _scale(b) - _scale(a)
            return lambda a, b: w(_div_scaled(raw(a), raw(b), e))
        k = min(_scale(a), _scale(b))
        if k == 0:
            return lambda a, b: w(raw(a) * raw(b))
        return lambda a, b: w(_mul_scaled(raw(a), raw(b), k))

    return overloaded


def _comparison(op):
    def overloaded(a, b):
        if not _either(a, b):
            return None
        if isinstance(a, types.Float) or isinstance(b, types.Float):
            s = _scale(a) or _scale(b)
            f = float(10 ** s)
            if isinstance(a, types.Float):
                return lambda a, b: op(a, float(raw(b)) / f)
            return lambda a, b: op(float(raw(a)) / f, b)
        if not (_number(a) and _number(b)):
            raise _undefined(op.__name__, a, b)
        s = _wide(a, b)
        fa, fb = _U(10 ** (s - _scale(a))), _U(10 ** (s - _scale(b)))
        if fa == fb == 1:
            return lambda a, b: op(raw(a), raw(b))
        return lambda a, b: op(_compare(raw(a), fa, raw(b), fb), 0)

    return overloaded


for _op, _symbol in ((operator.add, "+"), (operator.sub, "-")):
    _f = _additive(_symbol, _op)
    overload(_op, target="cpu")(_unwrapping(_f))
    overload(getattr(operator, f"i{_op.__name__}"), target="cpu")(_unwrapping(_f))
for _op, _symbol, _divide in ((operator.mul, "*", False), (operator.truediv, "/", True)):
    _f = _multiplicative(_symbol, _divide)
    overload(_op, target="cpu")(_unwrapping(_f))
    overload(getattr(operator, f"i{_op.__name__}"), target="cpu")(_unwrapping(_f))
for _op in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
    overload(_op, target="cpu")(_unwrapping(_comparison(_op)))


def _unsupported(symbol):
    def overloaded(a, b):
        if _either(a, b):
            raise _undefined(symbol, a, b, "round with decimal.round, divide with /")

    return overloaded


for _op, _symbol in ((operator.floordiv, "//"), (operator.mod, "%"), (operator.pow, "**")):
    overload(_op, target="cpu")(_unsupported(_symbol))
    overload(getattr(operator, f"i{_op.__name__}"), target="cpu")(_unsupported(_symbol))


@overload(operator.neg, target="cpu")
@_unwrapping
def _neg(a):
    if isinstance(a, DecimalType):
        w = wrapper(a.scale)
        return lambda a: w(-raw(a))


@overload(abs, target="cpu")
@_unwrapping
def _abs(a):
    if isinstance(a, DecimalType):
        w = wrapper(a.scale)
        return lambda a: w(-raw(a) if raw(a) < 0 else raw(a))


def _extreme(better):
    def overloaded(a, b):
        if isinstance(a, DecimalType) and isinstance(b, DecimalType):
            s = _wide(a, b)
            fa, fb, w = _U(10 ** (s - a.scale)), _U(10 ** (s - b.scale)), wrapper(s)
            ka, kb = 10 ** (s - a.scale), 10 ** (s - b.scale)
            return lambda a, b: w(raw(b) * kb if better(_compare(raw(b), fb, raw(a), fa), 0) else raw(a) * ka)

    return overloaded


overload(max, target="cpu")(_unwrapping(_extreme(operator.gt)))
overload(min, target="cpu")(_unwrapping(_extreme(operator.lt)))


@overload(float, target="cpu")
@_unwrapping
def _float(a):
    if isinstance(a, DecimalType):
        f = float(10 ** a.scale)
        return lambda a: float(raw(a)) / f


@overload(int, target="cpu")
@_unwrapping
def _int(a):
    if isinstance(a, DecimalType):
        p = _U(10 ** a.scale)
        return lambda a: _signed(_mag(raw(a)) // p, raw(a) < 0)


@overload(bool, target="cpu")
@_unwrapping
def _bool(a):
    if isinstance(a, DecimalType):
        return lambda a: raw(a) != 0


@overload(Decimal, target="cpu", prefer_literal=True)
def _new(value=0, scale=0):
    if not isinstance(value, types.Integer):
        raise TypingError(f"Decimal({value}) in a step takes an int; build it from a float or str outside the step")
    if isinstance(scale, types.IntegerLiteral):
        s = scale.literal_value
    elif isinstance(scale, int):
        s = scale   # the default, which numba hands over as the value
    else:
        raise TypingError("the scale of Decimal(n, scale) in a step must be a literal")
    if not 0 <= s <= 18:
        raise TypingError(f"scale must be from 0 to 18, not {s}")
    w, p = wrapper(s), 10 ** s
    return lambda value=0, scale=0: w(raw(value) * p)


@overload_attribute(DecimalType, "raw", target="cpu")
def _raw_attribute(a):
    return lambda a: raw(a)


@overload(ops.round, target="cpu")
def _round(x, digits=0, mode=Rounding.HALF_UP):
    if not isinstance(x, DecimalType):
        return lambda x, digits=0, mode=Rounding.HALF_UP: round_float(x, digits, mode)
    s, w = x.scale, wrapper(x.scale)

    def impl(x, digits=0, mode=Rounding.HALF_UP):
        k = s - digits
        if k <= 0:
            return x
        return w(_round_raw(raw(x), k, mode))

    return impl


def store_as(t, scale: int):
    """A function from a step's returned value of type `t` to the int64 stored for a Decimal output at `scale`.

    A decimal of another scale is rescaled (coarser rounds half-even), an int is whole units, and an optional
    stays optional. `None` for a step that only raises.
    """
    if isinstance(t, types.NoneType):
        return None
    if isinstance(t, types.Optional):
        inner = register_jitable(store_as(t.type, scale))

        def optional(x):
            if x is None:
                return None
            return inner(x)

        return optional
    if isinstance(t, DecimalType) and t.scale > scale:
        k = t.scale - scale
        return lambda x: _div_pow10(raw(x), k)
    if isinstance(t, (DecimalType, types.Integer)):
        m = 10 ** (scale - _scale(t))
        return lambda x: raw(x) * m
    raise TypingError(f"a Decimal output can't be stored from a {t}: return a Decimal or an int")


@overload(Currency, target="cpu")
def _currency(major=0, minor=None):
    if not isinstance(major, types.Integer):
        raise TypingError("Currency(...) in a step takes ints, as Currency(major, minor); build one from a float outside the step")
    w = wrapper(4)
    if minor is None or isinstance(minor, (types.Omitted, types.NoneType)):
        return lambda major=0, minor=None: w(raw(major) * 10000)
    if not isinstance(minor, types.Integer):
        raise TypingError("Currency(major, minor) in a step takes two ints")

    def impl(major=0, minor=None):
        if minor < 0 or minor >= 10000:
            raise ValueError("minor must be from 0 to 9999 (units of 0.0001)")
        return w(raw(major) * 10000 + (raw(minor) if major >= 0 else -raw(minor)))

    return impl
