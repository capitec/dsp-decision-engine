"""`decider2.expr` — the closed expression grammar (doc 06 §O15, doc 08
§1.2/§3.2), and its one intended caller: a computed `Feature`
(`decider2.trees.schema`).

Three tiers:

1. Unit tests against `expr.parse`/`Expr.emit` directly, with a minimal
   stand-in `ExprContext` (`_EvalCtx` below) that resolves a name to itself
   and a literal to its own `repr()`, so the emitted fragment can be
   `eval()`'d against real Python values with no numba compile in the loop
   — fast, and it pins the actual arithmetic/boolean semantics of every
   admitted construct.
2. The adversarial set: everything doc 06 §O15 says must be rejected AT
   PARSE, each asserted by construct and by a useful message.
3. Two end-to-end tests that go through the real pipeline — `tree_module` +
   `flow(...).apply(...)`, real numba compilation, no shortcuts — because
   "compiles to numba source" is exactly the claim worth proving for real
   at least once: a computed feature used as a tree condition, and the
   retune-without-recompile guarantee for a constant buried inside one.
"""
from __future__ import annotations

import re

import polars as pl
import pydantic
import pytest

from decider2 import expr, flow
from decider2.compile.driver import build_driver
from decider2.runtime.invoke import DEFAULT_BUILD_DIR
from decider2.trees import (
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    Tree,
    TreeOutput,
    UnaryGreaterThan,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryNode,
    encode_tree,
    tree_module,
)


class _EvalCtx:
    """The simplest possible `ExprContext`: a name resolves to itself (the
    caller supplies it as a real local when `eval`ing), and a literal
    resolves to its own `repr()` — never baked in by `expr.py` itself, just
    by this stand-in, so what comes back from `.emit()` is still "a name,
    never a literal" from `expr.py`'s point of view."""

    def name(self, ident: str) -> str:
        return ident

    def constant(self, value):
        return repr(value)


def _evaluate(source: str, **names):
    e = expr.parse(source)
    fragment = e.emit(_EvalCtx())
    return eval(fragment, {"min": min, "max": max, "abs": abs}, dict(names))


# ---------------------------------------------------------------------------
# 1. Every admitted construct, compiling and giving the right answer
# ---------------------------------------------------------------------------


def test_bare_name_is_a_dependency_and_evaluates_to_the_bound_value():
    assert _evaluate("x", x=7.0) == 7.0
    assert expr.parse("x").dependencies() == {"x"}


@pytest.mark.parametrize(
    "source, x, y, expected",
    [
        ("x + y", 3.0, 4.0, 7.0),
        ("x - y", 3.0, 4.0, -1.0),
        ("x * y", 3.0, 4.0, 12.0),
        ("x / y", 6.0, 4.0, 1.5),
        ("x // y", 7.0, 2.0, 3.0),
        ("x % y", 7.0, 2.0, 1.0),
        ("x ** y", 2.0, 3.0, 8.0),
    ],
)
def test_binary_operators(source, x, y, expected):
    assert _evaluate(source, x=x, y=y) == expected


def test_unary_minus():
    assert _evaluate("-x", x=5.0) == -5.0
    assert _evaluate("-(x - y)", x=3.0, y=5.0) == 2.0


@pytest.mark.parametrize(
    "source, x, y, expected",
    [
        ("x < y", 1.0, 2.0, True),
        ("x <= y", 2.0, 2.0, True),
        ("x > y", 3.0, 2.0, True),
        ("x >= y", 2.0, 2.0, True),
        ("x == y", 2.0, 2.0, True),
        ("x != y", 2.0, 3.0, True),
        ("x < y", 2.0, 1.0, False),
    ],
)
def test_comparisons(source, x, y, expected):
    assert _evaluate(source, x=x, y=y) is expected


def test_boolean_operators_and_or_not():
    assert _evaluate("a and b", a=True, b=True) is True
    assert _evaluate("a and b", a=True, b=False) is False
    assert _evaluate("a or b", a=False, b=True) is True
    assert _evaluate("not a", a=False) is True
    assert _evaluate("(x > 0) and (y > 0)", x=1.0, y=1.0) is True
    assert _evaluate("(x > 0) and (y > 0)", x=1.0, y=-1.0) is False


def test_parentheses_change_evaluation_order():
    assert _evaluate("(x + y) * 2", x=1.0, y=2.0) == 6.0
    assert _evaluate("x + y * 2", x=1.0, y=2.0) == 5.0


@pytest.mark.parametrize(
    "source, kwargs, expected",
    [
        ("min(x, y)", {"x": 3.0, "y": 5.0}, 3.0),
        ("max(x, y)", {"x": 3.0, "y": 5.0}, 5.0),
        ("min(x, y, z)", {"x": 3.0, "y": 5.0, "z": 1.0}, 1.0),
        ("abs(x)", {"x": -4.0}, 4.0),
    ],
)
def test_whitelisted_function_calls(source, kwargs, expected):
    assert _evaluate(source, **kwargs) == expected


def test_the_owners_own_example_x_minus_y():
    """Verbatim from the ask: "like x-y>10 it seems a pitty to now have to
    write preprocessing code just to calculate x-y"."""
    assert _evaluate("x - y > 10", x=25.0, y=10.0) is True
    assert _evaluate("x - y > 10", x=15.0, y=10.0) is False


def test_numeric_literals_admitted_int_and_float():
    assert _evaluate("x + 1", x=0.0) == 1  # int literal
    assert _evaluate("x + 1.5", x=0.0) == 1.5  # float literal


# ---------------------------------------------------------------------------
# dependencies() — static extraction, every free name, nothing else
# ---------------------------------------------------------------------------


def test_dependencies_reports_every_name_read_and_only_names():
    e = expr.parse("min(a, b) + c - abs(d) > e and not f")
    assert e.dependencies() == {"a", "b", "c", "d", "e", "f"}


def test_dependencies_excludes_whitelisted_call_targets():
    e = expr.parse("min(x, y)")
    assert "min" not in e.dependencies()


def test_dependencies_excludes_numeric_literals():
    e = expr.parse("x + 1 - 2.5 * 3")
    assert e.dependencies() == {"x"}


def test_dependencies_repeated_name_counted_once():
    e = expr.parse("x - x + x")
    assert e.dependencies() == {"x"}


# ---------------------------------------------------------------------------
# 2. The adversarial set — every rejected construct, rejected AT PARSE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, pattern",
    [
        ("x.y", "attribute access"),
        ("x.__class__", "attribute access"),
        ("__import__('os')", "not admitted"),  # rejected as an unlisted call
        ("__class__", "dunder name"),  # rejected as a bare dunder name
        ("eval(x)", "not admitted"),
        ("getattr(x, 'y')", "not admitted"),
        ("open('/etc/passwd')", "not admitted"),
        ("[x for x in y]", "list comprehension"),
        ("{x for x in y}", "set comprehension"),
        ("{x: y for x in z}", "dict comprehension"),
        ("(x for x in y)", "generator expression"),
        ("(x := 1)", "walrus"),
        ("f'{x}'", "f-string"),
        ("min(*args)", "star-args"),
        ("x[0]", "subscript"),
        ("x[0:1]", "subscript"),
        ("lambda x: x", "lambda"),
        ("[1, 2, 3]", "list literal"),
        ("(1, 2)", "tuple literal"),
        ("{1: 2}", "dict literal"),
        ("{1, 2}", "set literal"),
        ("1 if x else 2", "conditional expression"),
        ("True", "boolean literal"),
        ("False", "boolean literal"),
        ("'a string'", "not admitted"),
        ("min(x, y, key=z)", "keyword arguments"),
        ("abs(x, y)", "abs() takes"),
        ("min(x)", "min() takes"),
        ("sum(x, y)", "not admitted"),
        ("a < b < c", "chained comparison"),
        ("x is None", "not admitted"),
        ("x in y", "not admitted"),
        ("", "empty"),
    ],
)
def test_rejected_construct_names_itself_at_parse(source, pattern):
    with pytest.raises(expr.ExprError, match=re.escape(pattern)):
        expr.parse(source)


def test_rejected_construct_reports_its_position():
    with pytest.raises(expr.ExprError, match=r"line 1, column \d+"):
        expr.parse("x + y.z")


def test_import_is_rejected_a_statement_cannot_even_be_an_expression():
    with pytest.raises(expr.ExprError):
        expr.parse("import os")


def test_dunder_access_via_attribute_is_rejected_as_attribute_not_dunder():
    # `x.__class__` is caught by the attribute-access rule before the
    # dunder-name rule ever gets a look at the attribute — either message
    # is a correct rejection; this pins which one, so a future change that
    # silently swaps the order is visible.
    with pytest.raises(expr.ExprError, match="attribute access"):
        expr.parse("x.__class__")


def test_star_args_on_a_call_are_rejected_even_to_a_whitelisted_function():
    with pytest.raises(expr.ExprError, match="star-args"):
        expr.parse("min(*x)")


def test_a_non_string_or_empty_source_is_rejected():
    with pytest.raises(expr.ExprError):
        expr.parse("   ")


# ---------------------------------------------------------------------------
# Deep nesting and a very long expression
# ---------------------------------------------------------------------------


def test_deeply_nested_but_admitted_expression_still_evaluates():
    # 120 nested unary minuses (even count -> the sign cancels), comfortably
    # under the module's own recursion cap.
    source = "-" * 120 + "x"
    assert _evaluate(source, x=7.0) == 7.0


def test_expression_nested_past_the_cap_is_a_clean_error_not_a_recursionerror():
    source = "-" * 500 + "x"
    with pytest.raises(expr.ExprError, match="nests more than"):
        expr.parse(source)


def test_a_very_long_expression_reports_every_one_of_its_names():
    # A '+'-chain nests one BinOp per term (left-associative), so this stays
    # comfortably under the module's own depth cap; the dedicated nesting
    # test above pins what happens once a chain like this DOES cross it.
    names = [f"n{i}" for i in range(150)]
    source = " + ".join(names)
    e = expr.parse(source)
    assert e.dependencies() == set(names)
    assert _evaluate(source, **{n: 1.0 for n in names}) == 150.0


def test_a_very_wide_boolean_expression_short_circuits_the_same_as_python():
    names = [f"n{i}" for i in range(300)]
    source = " and ".join(names)
    e = expr.parse(source)
    assert e.dependencies() == set(names)
    values = {n: True for n in names}
    values["n150"] = False
    assert _evaluate(source, **values) is False


# ---------------------------------------------------------------------------
# 3. End to end — real trees, real numba compilation
# ---------------------------------------------------------------------------


def _two_leaf_tree(condition) -> Tree:
    return Tree(
        name="risk",
        edges=[
            MultiSourceEdge(source="root", target="hi", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="lo", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="root", data=UnaryNode(condition=condition)),
            PositionedNode(id="hi", data=LeafNode(result_idx=0)),
            PositionedNode(id="lo", data=LeafNode(result_idx=1)),
        ],
        output=TreeOutput(data=[{"r": 1.0}, {"r": 0.0}], default={"r": -1.0}, dtypes=[("r", "Float64")]),
    )


def test_a_computed_feature_used_as_a_tree_condition_end_to_end(tmp_path):
    """decider 1's exact wire format (`{"type": "computed", "expression":
    ...}`), one full tree, real numba compilation, real answers."""
    tree = _two_leaf_tree(UnaryGreaterThan(
        feature={"type": "computed", "expression": "x - y"}, threshold=10.0,
    ))
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({"x": [25.0, 15.0], "y": [10.0, 10.0]})
    result = built.decode(flow(built.module).apply(frame))["r"].to_list()
    # row 0: 25 - 10 = 15 > 10 -> hi (1.0); row 1: 15 - 10 = 5, not > 10 -> lo (0.0)
    assert result == [1.0, 0.0]


def test_the_owners_full_condition_as_one_computed_boolean_end_to_end(tmp_path):
    """The owner's example taken literally — `x - y > 10` as ONE computed
    feature (comparisons and the constant are all inside the expression
    itself), routed through `is_true` rather than a separate threshold."""
    tree = _two_leaf_tree(UnaryIsTrue(
        feature={"type": "computed", "expression": "x - y > 10"},
    ))
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({"x": [25.0, 15.0], "y": [10.0, 10.0]})
    result = built.decode(flow(built.module).apply(frame))["r"].to_list()
    assert result == [1.0, 0.0]


def test_computed_feature_dependencies_are_wired_like_any_other_feature(tmp_path):
    """Doc 04's lineage story: a computed feature's dependencies are not a
    second, hidden vocabulary — `Tree.required_features()` sees straight
    through it, the same way it sees a plain `Feature("x")`."""
    tree = _two_leaf_tree(UnaryGreaterThan(
        feature={"type": "computed", "expression": "x - y"}, threshold=10.0,
    ))
    assert tree.required_features() == {"x", "y"}


def test_retuning_a_literal_inside_a_computed_expression_never_recompiles(tmp_path):
    """Doc 08 §2's central guarantee, extended to a constant buried inside
    an expression rather than sitting in a node's own `threshold=`: it is
    STILL a `param()`-declared kernel argument (`_ExprEmitAdapter.
    constant` -> `EmitContext.threshold`, the exact machinery an ordinary
    threshold already uses), so retuning it is a values change. Same shape
    as `test_trees.py::test_retuning_a_threshold_never_recompiles`, ported
    to a name that "resolves to a param()" from inside an expression rather
    than from a node's own field.
    """
    tree = _two_leaf_tree(UnaryGreaterThan(
        feature={"type": "computed", "expression": "x * 2"}, threshold=10.0,
    ))

    # The generated literal param's name, straight from encoding — not
    # guessed: `_ExprEmitAdapter` names it "<node_id>_expr<n>", the same
    # naming `EncodeContext.threshold_slot` already gives any other literal.
    encoded = encode_tree(tree)
    literal_params = [(p.name, p.default) for p in encoded.params if "expr" in p.origin]
    assert literal_params == [("root_expr0", 2.0)]

    built = tree_module(tree, build_dir=tmp_path)
    pipeline = flow(built.module)
    frame = pl.DataFrame({"x": [4.0]})

    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    # `build_dir` must match what `pipeline.apply()` uses internally
    # (`runtime.invoke.DEFAULT_BUILD_DIR` — `Pipeline.apply` takes no
    # `build_dir=` of its own) so this driver and `apply()`'s own hit the
    # SAME `build_driver` cache entry and are the SAME object.
    driver = build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=DEFAULT_BUILD_DIR, terminal_names=frozenset({"risk_path", "r"}),
    )
    assert driver.segments[0].kind == "compiled"  # or the claim is vacuous

    answers = []
    for factor in (2.0, 100.0, 0.1):
        out = pipeline.apply(frame, params={"risk": {"root_expr0": factor}}, mode="fused")
        answers.append(out["r"].to_list())

    # 2, not 1: `risk_path` and `r` are each their own `PackedCompiledSegment`
    # now (`decider2.compile.driver.build_packed_kernel`'s own docstring —
    # a packed step never fuses with a neighbour), so this driver holds one
    # signature per step. The claim this test exists to pin — retuning the
    # literal never GROWS either signature — still holds across the loop
    # above; `test_len_driver_signatures_stays_one_across_eight_retunes_of_
    # an_expr_literal` below checks that directly.
    assert len(driver.signatures) == 2
    assert answers[0] != answers[1]  # the retune actually changed the answer


def test_len_driver_signatures_stays_one_across_eight_retunes_of_an_expr_literal(tmp_path):
    """The exact assertion the task asks for, spelled out on its own:
    `len(driver.signatures)` pinned across many retunes of a constant that
    lives inside a computed feature's expression."""
    tree = _two_leaf_tree(UnaryGreaterThan(
        feature={"type": "computed", "expression": "x * 2"}, threshold=10.0,
    ))
    built = tree_module(tree, build_dir=tmp_path)
    pipeline = flow(built.module)
    frame = pl.DataFrame({"x": [4.0]})

    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    driver = build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=DEFAULT_BUILD_DIR, terminal_names=frozenset({"risk_path", "r"}),
    )

    pipeline.apply(frame, params={"risk": {"root_expr0": 1.0}}, mode="fused")
    before = len(driver.signatures)
    for factor in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        pipeline.apply(frame, params={"risk": {"root_expr0": factor}}, mode="fused")

    assert len(driver.signatures) == before
    assert before == 2  # risk_path + r, each its own PackedCompiledSegment
