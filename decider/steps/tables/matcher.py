"""The one compiled decision-table matcher: every table's conditions and rows are data this kernel reads."""
from __future__ import annotations

from numba import njit, types
from numba.extending import overload

from decider.engine.boundary._arrow.intrinsics import load_f64, load_i64
from decider.steps.tables.encode import (
    BETWEEN,
    ELSE,
    EQ_B,
    EQ_F,
    EQ_S,
    HI_OP,
    IN_F,
    IN_S,
    KIND,
    LEAF,
    LO_OP,
    MATCH,
    THEN,
    VAR,
    WIDTH,
)
from decider.steps.trees.walker import EXACT, _compare, _matches, split


@njit(inline="always")
def _i(base, k):
    return load_i64(base + 8 * k)


@njit(inline="always")
def _f(base, k):
    return load_f64(base + 8 * k)


@njit
def _holds(r, f, b, s, rows, code, n):
    # Program row `code` tested against table row `r`; a null input never matches.
    ints, floats, chars = rows[0], rows[1], rows[2]
    kind, v, leaf = _i(code, KIND), _i(code, VAR), _i(code, LEAF)
    i0, f0 = _i(ints, 2 * leaf), _i(ints, 2 * leaf + 1)
    if kind == BETWEEN:
        x = f[v]
        if x is None:
            return False
        if _i(ints, i0 + r) and not _compare(_i(code, LO_OP), x, _f(floats, f0 + r)):
            return False
        return not _i(ints, i0 + n + r) or _compare(_i(code, HI_OP), x, _f(floats, f0 + n + r))
    if kind == EQ_F or kind == IN_F:
        y = f[v]
        if y is None:
            return False
        if kind == EQ_F:
            return _i(ints, i0 + r) != 0 and y == _f(floats, f0 + r)
        for j in range(_i(ints, i0 + r), _i(ints, i0 + r + 1)):
            if y == _f(floats, j):
                return True
        return False
    if kind == EQ_B:
        z = b[v]
        if z is None or not _i(ints, i0 + r):
            return False
        return (1.0 if z else 0.0) == _f(floats, f0 + r)
    if kind == EQ_S or kind == IN_S:
        addr, m = s[v]
        if m < 0:
            return False
        if kind == EQ_S:
            lo = _i(ints, i0 + r)
            return _i(ints, i0 + n + 1 + r) != 0 and _matches(addr, m, chars + lo, _i(ints, i0 + r + 1) - lo, EXACT)
        for j in range(_i(ints, i0 + r), _i(ints, i0 + r + 1)):
            lo = _i(ints, j)
            if _matches(addr, m, chars + lo, _i(ints, j + 1) - lo, EXACT):
                return True
        return False
    w = b[v]
    if w is None:
        return False
    return w


@njit
def _scan(f, b, s, rows, prog, entry):
    n = rows[3]
    for r in range(n):
        pc = entry
        while pc >= 0:
            code = prog + 8 * WIDTH * pc
            pc = _i(code, THEN) if _holds(r, f, b, s, rows, code, n) else _i(code, ELSE)
        if pc == MATCH:
            return r
    return -1


def table(params, consts):
    """The table's `Rows`: its param when its rows come from the params document, else its last const."""


@overload(table, inline="always")
def _table(params, consts):
    if len(params.types) == 0:
        return lambda params, consts: consts[3]
    return lambda params, consts: params[0]


def pick(outs, rows, r):
    """Each output's value at matched row `r`, or its default when `r` is -1."""


# Inlined by LLVM rather than numba, like the tree walker's `pick`: numba
# inlining re-types the recursion at every level.
@overload(pick, jit_options={"forceinline": True})
def _pick(outs, rows, r):
    if len(outs.types) == 0:
        return lambda outs, rows, r: ()
    spec = outs.types[0].types
    read = _read_f if isinstance(spec[1], types.Float) else _read_b if isinstance(spec[1], types.Boolean) else _read_i

    # An output is `(header slot, default[, default valid])`; the header
    # gives where its validity and values start.
    def impl(outs, rows, r):
        o = outs[0]
        v = o[1] if r < 0 else read(rows, o[0], r)
        return (v,) + pick(outs[1:], rows, r)

    def nullable(outs, rows, r):
        o = outs[0]
        if r < 0:
            v = o[1] if o[2] else None
        else:
            v = read(rows, o[0], r) if _i(rows[0], _i(rows[0], 2 * o[0]) + r) else None
        return (v,) + pick(outs[1:], rows, r)

    return nullable if len(spec) == 3 else impl


@njit(inline="always")
def _read_f(rows, slot, r):
    return _f(rows[1], _i(rows[0], 2 * slot + 1) + r)


@njit(inline="always")
def _read_i(rows, slot, r):
    return _i(rows[0], _i(rows[0], 2 * slot) + rows[3] + r)


@njit(inline="always")
def _read_b(rows, slot, r):
    return _i(rows[0], _i(rows[0], 2 * slot) + rows[3] + r) != 0


@njit(cache=True)
def match(row, params, consts):
    """The outputs of the first table row `row` matches: `row` laid out floats, bools, spans; `consts` from `encode`.

    `consts` is `(program address, entry, outputs[, Rows])`; the rows come
    from `params` when they are a table param. Only numbers cross this call.
    """
    f, _, b, s = split(row)
    return pick(consts[2], table(params, consts), _scan(f, b, s, table(params, consts), consts[0], consts[1]))
