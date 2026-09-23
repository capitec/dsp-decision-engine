import re

import pytest

from decider.steps import expr


def _evaluate(source: str, **names):
    return expr.parse(source).evaluate(names)


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


def test_and_or_give_a_bool_not_an_operand_so_they_match_a_kernel():
    assert _evaluate("x and y", x=2.0, y=3.0) is True
    assert _evaluate("x or y", x=0.0, y=3.0) is True


def test_parentheses_change_evaluation_order():
    assert _evaluate("(x + y) * 2", x=1.0, y=2.0) == 6.0
    assert _evaluate("x + y * 2", x=1.0, y=2.0) == 5.0


@pytest.mark.parametrize(
    "source, names, expected",
    [
        ("min(x, y)", {"x": 3.0, "y": 5.0}, 3.0),
        ("max(x, y)", {"x": 3.0, "y": 5.0}, 5.0),
        ("min(x, y, z)", {"x": 3.0, "y": 5.0, "z": 1.0}, 1.0),
        ("abs(x)", {"x": -4.0}, 4.0),
    ],
)
def test_whitelisted_function_calls(source, names, expected):
    assert _evaluate(source, **names) == expected


def test_x_minus_y_greater_than_ten():
    assert _evaluate("x - y > 10", x=25.0, y=10.0) is True
    assert _evaluate("x - y > 10", x=15.0, y=10.0) is False


def test_numeric_literals_admitted_int_and_float():
    assert _evaluate("x + 1", x=0.0) == 1
    assert _evaluate("x + 1.5", x=0.0) == 1.5


def test_dependencies_reports_every_name_read_and_only_names():
    assert expr.parse("min(a, b) + c - abs(d) > e and not f").dependencies() == {"a", "b", "c", "d", "e", "f"}


def test_dependencies_excludes_whitelisted_call_targets():
    assert "min" not in expr.parse("min(x, y)").dependencies()


def test_dependencies_excludes_numeric_literals():
    assert expr.parse("x + 1 - 2.5 * 3").dependencies() == {"x"}


def test_dependencies_repeated_name_counted_once():
    assert expr.parse("x - x + x").dependencies() == {"x"}


@pytest.mark.parametrize(
    "source, pattern",
    [
        ("x.y", "attribute access"),
        ("x.__class__", "attribute access"),
        ("__import__('os')", "not admitted"),
        ("__class__", "dunder name"),
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


def test_import_is_rejected_because_a_statement_is_not_an_expression():
    with pytest.raises(expr.ExprError):
        expr.parse("import os")


def test_star_args_on_a_call_are_rejected_even_to_a_whitelisted_function():
    with pytest.raises(expr.ExprError, match="star-args"):
        expr.parse("min(*x)")


def test_a_blank_source_is_rejected():
    with pytest.raises(expr.ExprError):
        expr.parse("   ")


def test_an_expression_error_is_a_value_error():
    assert issubclass(expr.ExprError, ValueError)


def test_deeply_nested_but_admitted_expression_still_evaluates():
    assert _evaluate("-" * 120 + "x", x=7.0) == 7.0


def test_expression_nested_past_the_cap_is_a_clean_error_not_a_recursion_error():
    with pytest.raises(expr.ExprError, match="nests more than"):
        expr.parse("-" * 500 + "x")


def test_a_very_long_expression_reports_every_one_of_its_names():
    names = [f"n{i}" for i in range(150)]
    source = " + ".join(names)
    assert expr.parse(source).dependencies() == set(names)
    assert _evaluate(source, **{n: 1.0 for n in names}) == 150.0


def test_a_very_wide_boolean_expression_evaluates_like_python():
    names = [f"n{i}" for i in range(300)]
    source = " and ".join(names)
    assert expr.parse(source).dependencies() == set(names)
    values = {n: True for n in names}
    values["n150"] = False
    assert _evaluate(source, **values) is False
