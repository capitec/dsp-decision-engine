# A `Raw[bytes]` value in a kernel is a `(address, byte length)` pair into Arrow
# memory. `SpanType` gives that pair its own numba type, so `==`, `len` and the
# rest below mean string things on a span and nothing else: overloading them on
# the bare `UniTuple(int64, 2)` would capture every unrelated pair of ints in a
# step, answering False for `(1, 2) == (1, 2)`. The type deliberately isn't a
# tuple either, because numba rewrites `len(t)` on a tuple *argument* to the
# constant 2 before type inference runs, where no overload can reach it.
from __future__ import annotations

import operator

import numpy as np
from numba.core import types
from numba.core.datamodel import models
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload, overload_method, register_jitable, register_model

from decider.engine.boundary._arrow.intrinsics import load_u8

_PAIR = types.UniTuple(types.int64, 2)


class SpanType(types.Type):
    """A `Raw[bytes]` value in compiled code: UTF-8 bytes at an address, byte length -1 for a null."""

    # What UniTupleModel reads, so a span is the same `[2 x i64]` value the kernel builds.
    dtype = types.int64
    count = 2

    def __init__(self) -> None:
        super().__init__("span")

    def __len__(self) -> int:
        return 2


register_model(SpanType)(models.UniTupleModel)

SPAN = SpanType()

_ADVICE = ("compare a span against a str literal, a module-level str constant or a param(), "
           "not against a string built at run time")


@intrinsic
def _pair(typingctx, t):
    """A span as a plain `(address, byte length)` tuple; the same value under another type."""
    if not isinstance(t, SpanType):
        return None
    return _PAIR(t), (lambda context, builder, sig, args: args[0])


def _comparable(t: object) -> bool:
    return t == _PAIR or isinstance(t, (SpanType, types.StringLiteral, types.UnicodeType))


def _constant(op: str, t: object) -> np.ndarray | None:
    """The UTF-8 bytes of a compile-time str, or `None` when the other side is a span."""
    if isinstance(t, types.StringLiteral):
        # Closed over by the implementation below, so numba lowers it as one constant array.
        return np.frombuffer(t.literal_value.encode(), np.uint8)
    if isinstance(t, types.UnicodeType):
        raise TypingError(f"'{op}' on a span: {_ADVICE}")
    return None


@register_jitable
def _same(a: int, b: int, n: int) -> bool:
    for k in range(n):
        if load_u8(a + k) != load_u8(b + k):
            return False
    return True


@register_jitable
def _same_const(a: int, lit: np.ndarray) -> bool:
    for k in range(len(lit)):
        if load_u8(a + k) != lit[k]:
            return False
    return True


@overload(operator.getitem, target="cpu")
def _span_getitem(s, k):
    if isinstance(s, SpanType) and isinstance(k, types.Integer):
        return lambda s, k: _pair(s)[k]


@overload(operator.eq, target="cpu", prefer_literal=True)
def _span_eq(a, b):
    if isinstance(b, SpanType) and not isinstance(a, SpanType):
        a, b = b, a
    if not isinstance(a, SpanType) or not _comparable(b):
        return None
    lit = _constant("==", b)
    if lit is None:
        # A null equals nothing, not even another null, as a NaN feature does.
        return lambda a, b: a[1] >= 0 and a[1] == b[1] and _same(a[0], b[0], a[1])
    m = len(lit)
    return lambda a, b: a[1] == m and _same_const(a[0], lit)


@overload(operator.ne, target="cpu", prefer_literal=True)
def _span_ne(a, b):
    # Only where `==` has an answer, so an unrelated type still reports its own error.
    if _span_eq(a, b) is not None:
        return lambda a, b: not (a == b)


@overload(len, target="cpu")
def _span_len(x):
    if not isinstance(x, SpanType):
        return None

    # CPython counts code points, so count the bytes that are not UTF-8 continuations.
    def impl(x):
        n = x[1]
        if n < 0:
            return -1
        points = 0
        for k in range(n):
            if (load_u8(x[0] + k) & 0xC0) != 0x80:
                points += 1
        return points

    return impl


@overload_method(SpanType, "startswith", target="cpu", prefer_literal=True)
def _span_startswith(a, b):
    if not _comparable(b):
        return None
    lit = _constant("startswith", b)
    if lit is None:
        return lambda a, b: b[1] >= 0 and a[1] >= b[1] and _same(a[0], b[0], b[1])
    m = len(lit)
    return lambda a, b: a[1] >= m and _same_const(a[0], lit)


@overload_method(SpanType, "endswith", target="cpu", prefer_literal=True)
def _span_endswith(a, b):
    if not _comparable(b):
        return None
    lit = _constant("endswith", b)
    if lit is None:
        return lambda a, b: b[1] >= 0 and a[1] >= b[1] and _same(a[0] + a[1] - b[1], b[0], b[1])
    m = len(lit)
    return lambda a, b: a[1] >= m and _same_const(a[0] + a[1] - m, lit)


@overload(operator.contains, target="cpu", prefer_literal=True)
def _span_contains(a, b):
    if not isinstance(a, SpanType) or not _comparable(b):
        return None
    lit = _constant("in", b)
    if lit is None:
        def impl(a, b):
            if b[1] < 0 or a[1] < b[1]:
                return False
            for k in range(a[1] - b[1] + 1):
                if _same(a[0] + k, b[0], b[1]):
                    return True
            return False

        return impl
    m = len(lit)

    def const_impl(a, b):
        if a[1] < m:
            return False
        for k in range(a[1] - m + 1):
            if _same_const(a[0] + k, lit):
                return True
        return False

    return const_impl
