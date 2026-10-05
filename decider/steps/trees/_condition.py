"""`expr_condition`: a boolean expression compiled into a one-output decision tree."""
from __future__ import annotations

import ast
import typing as t

from decider.exceptions import ExprError
from decider.steps import expr

if t.TYPE_CHECKING:
    from decider.steps.trees import TreeConfig

_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
_SYMBOL = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}


def _literal_number(node: ast.AST) -> t.Optional[t.Union[int, float]]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_number(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _feature(node: ast.AST) -> t.Any:
    if isinstance(node, ast.Name):
        return node.id
    text = ast.unparse(node)
    expr.parse(text)  # raises ExprError for anything not admitted; we don't duplicate its checks
    return {"type": "computed", "expression": text}


def _compare(node: ast.Compare, negated: bool) -> dict:
    op_type = type(node.ops[0])
    if op_type not in _SYMBOL:
        raise ExprError(f"comparison {op_type.__name__!r} is not admitted (only < <= > >= == != are)")
    sym = _SYMBOL[op_type]
    left, right = node.left, node.comparators[0]
    left_str = isinstance(left, ast.Constant) and isinstance(left.value, str)
    right_str = isinstance(right, ast.Constant) and isinstance(right.value, str)
    if left_str or right_str:
        if op_type not in (ast.Eq, ast.NotEq):
            raise ExprError(f"a string literal only compares with == or !=, not {sym!r}")
        name_side, str_side = (right, left) if left_str else (left, right)
        if not isinstance(name_side, ast.Name):
            raise ExprError(f"a string literal only compares against a plain name, not {ast.unparse(name_side)!r}")
        condition = {"op": "string_match", "feature": name_side.id, "patterns": [str_side.value]}
        # A NOT doesn't flip a string match's null result as it does a numeric test's, so under
        # negation a null must match for the whole expression to stay false ("!=" negates too).
        if negated ^ (op_type is ast.NotEq):
            condition["null_handling"] = "match"
        return {"type": "composite", "op": "not", "conditions": [condition]} if op_type is ast.NotEq else condition
    left_num, right_num = _literal_number(left), _literal_number(right)
    if left_num is not None and right_num is None:
        return {"op": _FLIP[sym], "feature": _feature(right), "threshold": left_num}
    if right_num is not None and left_num is None:
        return {"op": sym, "feature": _feature(left), "threshold": right_num}
    # Neither side is a bare number literal: compare their difference to 0.
    # ponytail: float subtraction, so exact equality of large ints can be off by rounding;
    # upgrade to a two-feature int comparison in the tree encoder if that ever matters.
    difference = f"({ast.unparse(left)}) - ({ast.unparse(right)})"
    expr.parse(difference)
    return {"op": sym, "feature": {"type": "computed", "expression": difference}, "threshold": 0}


def _condition(node: ast.AST, negated: bool = False) -> dict:
    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        return {"type": "composite", "op": op, "conditions": [_condition(v, negated) for v in node.values]}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return {"type": "composite", "op": "not", "conditions": [_condition(node.operand, not negated)]}
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExprError("a chained comparison (e.g. 'a < b < c') is not admitted; write 'a < b and b < c'")
        return _compare(node, negated)
    if isinstance(node, ast.Name):
        return {"op": "is_true", "feature": node.id}
    raise ExprError(f"{ast.unparse(node)!r} is not admitted as a condition "
                     "(use a comparison, a name, or 'and'/'or'/'not')")


def expr_condition(source: str, *, name: str, feature_types: t.Optional[t.Dict[str, str]] = None) -> "TreeConfig":
    """A boolean expression as a one-output step: a bool column `name`, true where the expression holds.

    `source` is a Python boolean expression over column names: numeric
    comparisons (`"score >= 600"`), string equality (`'crc_model == "CrC3.0"'`),
    `and`/`or`/`not`, and a bare name as a boolean column. An arithmetic side
    (`"income - expenses > 1000"`) is validated the same way as a tree's
    computed feature. A null input makes the expression false. `feature_types`
    is passed through to `TreeConfig` (e.g. `{"months": "int"}`).

    Example::

        good = expr_condition('score >= 600 and crc_model == "CrC3.0"', name="good")
        good.run(df)["good"]
    """
    from decider.steps.trees import TreeConfig  # local: TreeConfig imports this module back

    if not isinstance(source, str) or not source.strip():
        raise ExprError(f"expression is empty or not a string: {source!r}")
    try:
        body = ast.parse(source, mode="eval").body
    except SyntaxError as e:
        raise ExprError(f"{source!r} is not a valid expression: {e.msg}") from e
    condition = _condition(body)
    leaf = {"type": "leaf", "result_idx": 0}
    rule = {**condition, "then": leaf} if condition.get("type") == "composite" else {
        "type": "unary", "condition": condition, "then": leaf}
    return TreeConfig.load({
        "type": "tree", "name": name, "feature_types": feature_types or {},
        "tree": {
            "type": "flat_rule", "rule": {"rule": rule},
            "output": {"data": [{name: True}], "default": {name: False}, "dtypes": [[name, "Boolean"]]},
        },
    })
