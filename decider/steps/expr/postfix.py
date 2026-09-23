"""Expressions as data: postfix opcode rows evaluated by one generic njit kernel."""
from __future__ import annotations

import ast
import typing as t

import numpy as np
from numba import njit

from decider.engine.boundary._arrow.intrinsics import load_f64, load_i64
from decider.steps.expr import Expr

PUSH, NAME, ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, NEG, NOT = range(11)
LT, LE, GT, GE, EQ, NE, AND, OR, MIN, MAX, ABS = range(11, 22)

_BINOPS = {ast.Add: ADD, ast.Sub: SUB, ast.Mult: MUL, ast.Div: DIV, ast.FloorDiv: FLOORDIV, ast.Mod: MOD,
           ast.Pow: POW}
_CMPOPS = {ast.Lt: LT, ast.LtE: LE, ast.Gt: GT, ast.GtE: GE, ast.Eq: EQ, ast.NotEq: NE}
_CALLS = {"min": MIN, "max": MAX}


def to_postfix(e: Expr, slot: t.Callable[[str], int], consts: list[float]) -> tuple[list[tuple[int, int]], int]:
    """`e` as postfix `(opcode, argument)` rows and the stack depth they need.

    `slot(name)` is where a name's value sits in the float array the kernel
    is given; literals are appended to `consts` and pushed by position.

    Example::

        consts = []
        rows, depth = to_postfix(parse("x - 2"), {"x": 0}.__getitem__, consts)
        code, lits = np.array(rows), np.array(consts)
        evaluate(code.ctypes.data, 0, len(rows), lits.ctypes.data, (5.0,), depth)   # 3.0
    """
    rows: list[tuple[int, int]] = []
    depth = [0, 0]

    def push(op: int, arg: int = 0, pops: int = 0) -> None:
        rows.append((op, arg))
        depth[0] += 1 - pops
        depth[1] = max(depth[1], depth[0])

    def fold(values: t.Sequence[ast.AST], op: int) -> None:
        emit(values[0])
        for v in values[1:]:
            emit(v)
            push(op, pops=2)

    def emit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            push(NAME, slot(node.id))
        elif isinstance(node, ast.Constant):
            consts.append(float(node.value))
            push(PUSH, len(consts) - 1)
        elif isinstance(node, ast.BinOp):
            fold((node.left, node.right), _BINOPS[type(node.op)])
        elif isinstance(node, ast.UnaryOp):
            emit(node.operand)
            push(NEG if isinstance(node.op, ast.USub) else NOT, pops=1)
        elif isinstance(node, ast.Compare):
            fold((node.left, node.comparators[0]), _CMPOPS[type(node.ops[0])])
        elif isinstance(node, ast.BoolOp):
            # ponytail: both sides are always evaluated; a guard like `x == 0 or 1 / x > 2`
            # raises here where Python short-circuits. Add jump opcodes if a tree needs it.
            fold(node.values, AND if isinstance(node.op, ast.And) else OR)
        elif node.func.id == "abs":
            emit(node.args[0])
            push(ABS, pops=1)
        else:
            fold(node.args, _CALLS[node.func.id])

    emit(e.node)
    return rows, depth[1]


@njit(inline="always")
def _apply(op, a, b):
    if op == ADD:
        return a + b
    if op == SUB:
        return a - b
    if op == MUL:
        return a * b
    if op == DIV:
        return a / b
    if op == FLOORDIV:
        return a // b
    if op == MOD:
        return a % b
    if op == POW:
        return a ** b
    if op == LT:
        return 1.0 if a < b else 0.0
    if op == LE:
        return 1.0 if a <= b else 0.0
    if op == GT:
        return 1.0 if a > b else 0.0
    if op == GE:
        return 1.0 if a >= b else 0.0
    if op == EQ:
        return 1.0 if a == b else 0.0
    if op == NE:
        return 1.0 if a != b else 0.0
    if op == AND:
        return 1.0 if a != 0.0 and b != 0.0 else 0.0
    if op == OR:
        return 1.0 if a != 0.0 or b != 0.0 else 0.0
    if op == MIN:
        return min(a, b)
    return max(a, b)


@njit
def evaluate(code, start, end, consts, names, depth):
    """The value of postfix rows `start` to `end` at address `code` (two int64 per row).

    Names are read from the tuple `names`, literals from the float64 array at
    address `consts`; addresses, so no array is reference-counted per row.
    """
    # ponytail: one small allocation per evaluation; a fixed-size stack if computed features get hot.
    stack = np.empty(depth)
    top = 0
    for pc in range(start, end):
        op = load_i64(code + 16 * pc)
        arg = load_i64(code + 16 * pc + 8)
        if op == PUSH:
            stack[top] = load_f64(consts + 8 * arg)
            top += 1
        elif op == NAME:
            stack[top] = names[arg]
            top += 1
        elif op == NEG:
            stack[top - 1] = -stack[top - 1]
        elif op == NOT:
            stack[top - 1] = 1.0 if stack[top - 1] == 0.0 else 0.0
        elif op == ABS:
            stack[top - 1] = abs(stack[top - 1])
        else:
            top -= 1
            stack[top - 1] = _apply(op, stack[top - 1], stack[top])
    return stack[0]
