"""The one compiled tree walker: every tree is data (`encode.Program`) this kernel walks."""
from __future__ import annotations

from numba import njit, types
from numba.extending import overload

from decider.engine.boundary._arrow.intrinsics import load_f64, load_i64, load_u8
from decider.steps.expr.postfix import evaluate
from decider.steps.trees.ops import EQ, GE, GT, LE, LT

# Program row kinds, and the columns of a program row.
LEAF, CMP_F, CMP_I, CMP_B, CMP_E, MATCH = range(6)
KIND, FEAT, OP, THR, THEN, ELSE, UNKNOWN = range(7)
WIDTH = 7
# A null int feature arrives as this; a null float as NaN; a null bool as None.
NULL_INT = -(2**63)
# How a MATCH row compares a string with its patterns.
EXACT, PREFIX, SUFFIX, CONTAINS = range(4)
# Where each part of the program starts: the fields of `Program.layout`.
PROG, THR_I, ROOTS, N_ROOTS, EXPR_CODE, EXPR_STARTS, PAT_STARTS, PAT_GROUPS, THR_F, EXPR_CONSTS, EXPR_DEPTH = range(11)


def _kind(t) -> int:
    if isinstance(t, types.Optional):
        t = t.type
    if isinstance(t, types.Boolean):
        return 2
    if isinstance(t, types.Integer):
        return 1
    return 0 if isinstance(t, types.Float) else 3


def split(values):
    """`(floats, ints, bools, spans)` of a tuple laid out in that order; a span is `(address, byte length)`."""


@overload(split, inline="always")
def _split(values):
    kinds = [_kind(v) for v in values.types]
    a, b, c, n = kinds.count(0), kinds.count(0) + kinds.count(1), len(kinds) - kinds.count(3), len(kinds)

    # An empty group becomes a one-item tuple no program row indexes, so it
    # still types as a homogeneous tuple.
    def impl(values):
        return ((values[:a] if a else (0.0,)), (values[a:b] if b > a else (0,)),
                (values[b:c] if c > b else (False,)), (values[c:] if n > c else ((0, -1),)))

    return impl


@njit(inline="always")
def _i(base, k):
    return load_i64(base + 8 * k)


@njit(inline="always")
def _f(base, k):
    return load_f64(base + 8 * k)


@njit(inline="always")
def _compare(op, a, b):
    if op == LT:
        return a < b
    if op == LE:
        return a <= b
    if op == EQ:
        return a == b
    if op == GT:
        return a > b
    if op == GE:
        return a >= b
    return a != b


@njit
def _same(a, b, n):
    for k in range(n):
        if load_u8(a + k) != load_u8(b + k):
            return False
    return True


@njit
def _matches(s, n, p, m, mode):
    # UTF-8 bytes against pattern bytes: an empty pattern is a prefix, suffix and part of every string.
    if mode == EXACT:
        return n == m and _same(s, p, m)
    if mode == PREFIX:
        return n >= m and _same(s, p, m)
    if mode == SUFFIX:
        return n >= m and _same(s + n - m, p, m)
    for k in range(n - m + 1):
        if _same(s + k, p, m):
            return True
    return False


@njit
def _match(op, span, g, ps, ints, chars, lay):
    addr, n = span
    if g < 0:
        p, m = ps[-g - 1]
        return _matches(addr, n, p, m, op)
    starts = lay[PAT_STARTS]
    for k in range(_i(ints, lay[PAT_GROUPS] + g), _i(ints, lay[PAT_GROUPS] + g + 1)):
        lo = _i(ints, starts + k)
        if _matches(addr, n, chars + lo, _i(ints, starts + k + 1) - lo, op):
            return True
    return False


@njit(inline="always")
def _walk(pc, ctx):
    f, i, b, s, pf, pi, ps, ints, floats, chars, lay = ctx
    while True:
        row = lay[PROG] + WIDTH * pc
        k = _i(ints, row + KIND)
        if k == LEAF:
            return _i(ints, row + FEAT)
        j, op, t = _i(ints, row + FEAT), _i(ints, row + OP), _i(ints, row + THR)
        # A null feature takes the row's UNKNOWN target, which the encoder points at
        # the branch that null takes; a negative threshold slot is a param: -1 the first of its kind.
        if k == CMP_I:
            xi = i[j]
            if xi == NULL_INT:
                pc = _i(ints, row + UNKNOWN)
                continue
            r = _compare(op, xi, _i(ints, lay[THR_I] + t) if t >= 0 else pi[-t - 1])
        elif k == CMP_B:
            y = b[j]
            if y is None:
                pc = _i(ints, row + UNKNOWN)
                continue
            r = _compare(op, y, _i(ints, lay[THR_I] + t) != 0)
        elif k == MATCH:
            if s[j][1] < 0:
                pc = _i(ints, row + UNKNOWN)
                continue
            r = _match(op, s[j], t, ps, ints, chars, lay)
        else:
            if k == CMP_F:
                x = f[j]
            else:
                start = _i(ints, lay[EXPR_STARTS] + j)
                x = evaluate(ints + 8 * lay[EXPR_CODE], start, _i(ints, lay[EXPR_STARTS] + j + 1),
                             floats + 8 * lay[EXPR_CONSTS], f, lay[EXPR_DEPTH])
            # A null float, and so a computed feature over one, is NaN.
            if x != x:
                pc = _i(ints, row + UNKNOWN)
                continue
            r = _compare(op, x, _f(floats, lay[THR_F] + t) if t >= 0 else pf[-t - 1])
        pc = _i(ints, row + THEN) if r else _i(ints, row + ELSE)


# Real calls, not numba inlining: inlining the walk at each output or call
# site multiplies compile time, and LLVM inlines what pays anyway.
@njit
def _leaf(rule, ctx):
    ints, lay = ctx[7], ctx[10]
    if rule >= 0:
        return _walk(_i(ints, lay[ROOTS] + rule), ctx)
    for k in range(lay[N_ROOTS]):
        leaf = _walk(_i(ints, lay[ROOTS] + k), ctx)
        if leaf != -1:
            return leaf
    return -1


def pick(outs, ctx, rule, leaf):
    """Each output's value at the leaf its rule reached; consecutive outputs of one rule walk it once."""


# Inlined by LLVM, not numba: numba inlining this recursion re-types every
# level at every level, which grows compile time exponentially with outputs.
@overload(pick, jit_options={"forceinline": True})
def _pick(outs, ctx, rule, leaf):
    if len(outs.types) == 0:
        return lambda outs, ctx, rule, leaf: ()
    spec = outs.types[0].types
    read = _read_f if isinstance(spec[3], types.Float) else _read_b if isinstance(spec[3], types.Boolean) else _read_i

    # An output is `(rule, offset, rows, zero[, valid offset])`: `rows` values
    # from `offset`, the last one the default row's, in the int or float array.
    def impl(outs, ctx, rule, leaf):
        o = outs[0]
        if o[0] != rule:
            rule, leaf = o[0], _leaf(o[0], ctx)
        k = leaf if leaf >= 0 else o[2] - 1
        return (read(ctx, o[1] + k),) + pick(outs[1:], ctx, rule, leaf)

    def nullable(outs, ctx, rule, leaf):
        o = outs[0]
        if o[0] != rule:
            rule, leaf = o[0], _leaf(o[0], ctx)
        k = leaf if leaf >= 0 else o[2] - 1
        return (read(ctx, o[1] + k) if _i(ctx[7], o[4] + k) else None,) + pick(outs[1:], ctx, rule, leaf)

    return nullable if len(spec) == 5 else impl


@njit(inline="always")
def _read_f(ctx, k):
    return _f(ctx[8], k)


@njit(inline="always")
def _read_i(ctx, k):
    return _i(ctx[7], k)


@njit(inline="always")
def _read_b(ctx, k):
    return _i(ctx[7], k) != 0


@njit(cache=True)
def walk(row, params, consts):
    """Walk a tree for one row: `row` and `params` laid out floats, ints, bools, spans; `consts` a `Program`'s.

    `consts` is `(ints, floats, chars, layout, outputs)`: the addresses of the
    program's int64, float64 and pattern-byte arrays, where each part starts,
    and one spec per output. Only numbers cross this call, so nothing is
    reference-counted per row. Returns one value per output.
    """
    f, i, b, s = split(row)
    pf, pi, _, ps = split(params)
    ints, floats, chars, lay, outs = consts
    ctx = (f, i, b, s, pf, pi, ps, ints, floats, chars, lay)
    rule = outs[0][0]
    return pick(outs, ctx, rule, _leaf(rule, ctx))
