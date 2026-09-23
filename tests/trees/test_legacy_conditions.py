"""Rule conditions give decider_old's answers: unary, cases, composite, special values, prioritized rules."""
import polars as pl
import pytest

from decider.exceptions import MissingInputError
from decider.steps.trees import (
    CasesBranch,
    CasesIsInRule,
    CasesRangesRule,
    CasesStringMatchRule,
    CompositeCondition,
    CompositeRule,
    FlatRuleDocument,
    IsInCondition,
    LeafRule,
    LogicOp,
    ParameterInfo,
    PrioritizedFlatRuleDocument,
    RangeCondition,
    RangeEndLogic,
    RuleMeta,
    RuleRoot,
    StringMatchCondition,
    TreeConfig,
    TreeOutput,
    UnaryBetween,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryIsFalse,
    UnaryIsIn,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryLessThanEqual,
    UnaryNotEqual,
    UnaryRule,
    UnaryStringMatch,
)


def _out(*labels, default="default"):
    return TreeOutput(data=[{"r": lbl} for lbl in labels], default={"r": default}, dtypes=[("r", "String")])


def _leaf(i):
    return LeafRule(result_idx=i)


def _if(cond, then=0, other=-1):
    return UnaryRule(condition=cond, then=_leaf(then), otherwise=_leaf(other))


def _results(run, rule, out, df, params=None, **doc):
    tree = TreeConfig(name="t", tree=FlatRuleDocument(rule=RuleRoot(rule=rule), output=out, **doc))
    return run(tree, df, params)["r"].to_list()


@pytest.mark.parametrize("cond, labels, expected", [
    (UnaryLessThan(feature="x", threshold=10.0), ("low", "high"), ["low", "high", "high"]),
    (UnaryLessThanEqual(feature="x", threshold=10.0), ("low", "high"), ["low", "low", "high"]),
    (UnaryEqual(feature="x", threshold=10.0), ("yes", "no"), ["no", "yes", "no"]),
    (UnaryGreaterThan(feature="x", threshold=10.0), ("high", "low"), ["low", "low", "high"]),
    (UnaryGreaterThanEqual(feature="x", threshold=10.0), ("high", "low"), ["low", "high", "high"]),
    (UnaryNotEqual(feature="x", threshold=10.0), ("ne", "eq"), ["ne", "eq", "ne"]),
])
def test_numeric_comparisons_route_boundary_values(run, cond, labels, expected):
    assert _results(run, _if(cond, 0, 1), _out(*labels), pl.DataFrame({"x": [5.0, 10.0, 15.0]})) == expected


@pytest.mark.parametrize("bounds, expected", [
    ({"min": 10.0, "max": 20.0}, ["no", "yes", "no", "no"]),
    ({"min": 20.0}, ["no", "no", "yes", "yes"]),
    ({"max": 10.0}, ["yes", "no", "no", "no"]),
])
def test_between_is_closed_at_both_ends_and_takes_one_bound(run, bounds, expected):
    df = pl.DataFrame({"x": [5.0, 15.0, 25.0, 35.0]})
    assert _results(run, _if(UnaryBetween(feature="x", **bounds)), _out("yes", default="no"), df) == expected


def test_is_true_and_is_false_route_booleans(run):
    df = pl.DataFrame({"flag": [True, False, True]})
    out = _out("T", "F", default="?")
    assert _results(run, _if(UnaryIsTrue(feature="flag"), 0, 1), out, df) == ["T", "F", "T"]
    assert _results(run, _if(UnaryIsFalse(feature="flag"), 1, 0), out, df) == ["T", "F", "T"]


@pytest.mark.parametrize("match_type, pattern, hits", [
    ("exact", "world", [False, True, False, False]),
    ("contains", "ello", [True, False, True, False]),
    ("starts_with", "hello", [True, False, True, False]),
    ("ends_with", "orld", [True, True, False, False]),
    ("regex", "^h.*d$", [True, False, False, False]),
])
def test_every_string_match_type(run, match_type, pattern, hits):
    df = pl.DataFrame({"s": ["hello world", "world", "hello", "goodbye"]})
    rule = _if(UnaryStringMatch(feature="s", patterns=[pattern], match_type=match_type))
    assert _results(run, rule, _out("match", default="no"), df) == ["match" if h else "no" for h in hits]


def test_string_match_case_insensitive_and_trimmed(run):
    df = pl.DataFrame({"s": ["  HELLO  ", "hello", "WORLD"]})
    out = _out("match", default="no")
    folded = _if(UnaryStringMatch(feature="s", patterns=["hello"], case_sensitive=False))
    trimmed = _if(UnaryStringMatch(feature="s", patterns=["HELLO"], trim_whitespace=True))
    assert _results(run, folded, out, df) == ["no", "match", "no"]
    assert _results(run, trimmed, out, df) == ["match", "no", "no"]


def test_a_null_string_never_matches(run):
    df = pl.DataFrame({"s": ["a", None]})
    assert _results(run, _if(UnaryStringMatch(feature="s", patterns=["a"])), _out("hit", default="no"), df) == [
        "hit", "no"]


def _bands(end_logic):
    return CasesRangesRule(
        feature="score",
        conditions=[CasesBranch(when=RangeCondition(max=30.0), then=0),
                    CasesBranch(when=RangeCondition(min=30.0, max=70.0), then=1),
                    CasesBranch(when=RangeCondition(min=70.0), then=2)],
        otherwise=3, branches=[_leaf(0), _leaf(1), _leaf(2), _leaf(-1)], end_logic=end_logic, strict=False,
    )


@pytest.mark.parametrize("end_logic, at30, at70", [
    (RangeEndLogic.lower_inclusive, "mid", "high"),
    (RangeEndLogic.upper_inclusive, "low", "mid"),
])
def test_range_cases_close_the_end_their_end_logic_names(run, end_logic, at30, at70):
    df = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]})
    got = _results(run, _bands(end_logic), _out("low", "mid", "high", default="none"), df)
    assert (got[1], got[3]) == (at30, at70)


def test_isin_cases_route_codes(run):
    rule = CasesIsInRule(
        feature="code",
        conditions=[CasesBranch(when=IsInCondition(values=[1, 2]), then=0),
                    CasesBranch(when=IsInCondition(values=[3, 4]), then=1)],
        otherwise=2, branches=[_leaf(0), _leaf(1), _leaf(-1)],
    )
    assert _results(run, rule, _out("grp1", "grp2", default="other"), pl.DataFrame({"code": [1, 3, 5]})) == [
        "grp1", "grp2", "other"]


def test_string_match_cases_route_prefixes(run):
    rule = CasesStringMatchRule(
        feature="s", match_type="starts_with",
        conditions=[CasesBranch(when=StringMatchCondition(patterns=["a"]), then=0),
                    CasesBranch(when=StringMatchCondition(patterns=["b", "c"]), then=1)],
        otherwise=2, branches=[_leaf(0), _leaf(1), _leaf(-1)],
    )
    df = pl.DataFrame({"s": ["apple", "banana", "cherry", "durian"]})
    assert _results(run, rule, _out("A", "B", default="other"), df) == ["A", "B", "B", "other"]


@pytest.mark.parametrize("op, conditions, expected", [
    (LogicOp.AND, [UnaryGreaterThan(feature="x", threshold=5.0), UnaryLessThan(feature="x", threshold=10.0)],
     ["no", "yes", "no"]),
    (LogicOp.OR, [UnaryLessThan(feature="x", threshold=5.0), UnaryGreaterThan(feature="x", threshold=10.0)],
     ["yes", "no", "yes"]),
    (LogicOp.NOT, [UnaryGreaterThan(feature="x", threshold=5.0)], ["yes", "no", "no"]),
])
def test_and_or_not_composites(run, op, conditions, expected):
    rule = CompositeRule(op=op, conditions=conditions, then=_leaf(0), otherwise=_leaf(-1))
    assert _results(run, rule, _out("yes", default="no"), pl.DataFrame({"x": [3.0, 7.0, 12.0]})) == expected


def test_a_composite_nests_inside_a_composite(run):
    inner = CompositeCondition(op=LogicOp.AND, conditions=[UnaryGreaterThan(feature="x", threshold=0.0),
                                                           UnaryLessThan(feature="x", threshold=10.0)])
    rule = CompositeRule(op=LogicOp.OR, conditions=[inner, UnaryGreaterThan(feature="x", threshold=20.0)],
                         then=_leaf(0), otherwise=_leaf(-1))
    df = pl.DataFrame({"x": [-5.0, 5.0, 15.0, 25.0]})
    assert _results(run, rule, _out("yes", default="no"), df) == ["no", "yes", "no", "yes"]


def test_the_default_row_answers_when_nothing_matches(run):
    rule = _if(UnaryLessThan(feature="x", threshold=0.0))
    assert _results(run, rule, _out("match", default="FALLBACK"), pl.DataFrame({"x": [100.0, 200.0]})) == [
        "FALLBACK", "FALLBACK"]


def test_result_idx_selects_its_output_row(run):
    rule = CasesIsInRule(
        feature="x",
        conditions=[CasesBranch(when=IsInCondition(values=[v]), then=i) for i, v in enumerate([1.0, 2.0, 3.0])],
        otherwise=3, branches=[_leaf(0), _leaf(1), _leaf(2), _leaf(-1)],
    )
    out = _out("row0", "row1", "row2", default="none")
    assert _results(run, rule, out, pl.DataFrame({"x": [1.0, 2.0, 3.0]})) == ["row0", "row1", "row2"]


def test_infinities_and_nan_fail_a_finite_range(run):
    df = pl.DataFrame({"v": [1.0, float("inf"), float("-inf"), float("nan")]})
    rule = _if(UnaryBetween(feature="v", min=0.0, max=100.0))
    assert _results(run, rule, _out("match", default="no"), df) == ["match", "no", "no", "no"]


def test_a_null_number_is_a_missing_input_under_strict_null_handling():
    tree = TreeConfig(name="t", null_handling="error", tree=FlatRuleDocument(
        rule=RuleRoot(rule=_if(UnaryBetween(feature="v", min=0.0, max=100.0))), output=_out("match", default="no")))
    with pytest.raises(MissingInputError):
        tree.run(pl.DataFrame({"v": pl.Series([1.0, None], dtype=pl.Float64)}))


def _prioritized(rules, out, **doc):
    roots = [RuleRoot(meta=RuleMeta(name=name), rule=rule) for name, rule in rules]
    return TreeConfig(name="t", tree=PrioritizedFlatRuleDocument(rules=roots, output=out, **doc))


def test_the_first_matching_rule_wins(run):
    tree = _prioritized([("under50", _if(UnaryLessThan(feature="score", threshold=50.0), 0)),
                         ("under100", _if(UnaryLessThan(feature="score", threshold=100.0), 1))],
                        _out("under50", "under100", default="over100"))
    assert run(tree, pl.DataFrame({"score": [10.0, 70.0, 150.0]}))["r"].to_list() == [
        "under50", "under100", "over100"]


def test_prioritized_rules_fall_back_to_the_default(run):
    tree = _prioritized([("low", _if(UnaryLessThan(feature="score", threshold=100.0)))], _out("low", default="high"))
    assert run(tree, pl.DataFrame({"score": [200.0, 300.0]}))["r"].to_list() == ["high", "high"]


def test_unary_isin_routes_listed_values(run):
    rule = _if(UnaryIsIn(feature="code", values=[1, 3, 5]))
    assert _results(run, rule, _out("allowed", default="denied"), pl.DataFrame({"code": [1, 2, 3, 4, 5]})) == [
        "allowed", "denied", "allowed", "denied", "allowed"]


def test_range_case_bounds_come_from_params(run):
    rule = CasesRangesRule(
        feature="score",
        conditions=[CasesBranch(when=RangeCondition(max={"key": "lo"}), then=0),
                    CasesBranch(when=RangeCondition(min={"key": "lo"}, max={"key": "hi"}), then=1)],
        otherwise=2, branches=[_leaf(0), _leaf(1), _leaf(-1)], strict=False,
    )
    parameters = {"lo": ParameterInfo(type="Float64", default_value=20.0),
                  "hi": ParameterInfo(type="Float64", default_value=60.0)}
    df = pl.DataFrame({"score": [10.0, 40.0, 80.0]})
    assert _results(run, rule, _out("low", "mid", default="high"), df, parameters=parameters) == [
        "low", "mid", "high"]
    assert _results(run, rule, _out("low", "mid", default="high"), df, {"t": {"hi": 90.0}},
                    parameters=parameters) == ["low", "mid", "mid"]


def test_a_string_case_pattern_comes_from_a_param(run):
    rule = CasesStringMatchRule(
        feature="status", conditions=[CasesBranch(when=StringMatchCondition(patterns=[{"key": "vip_tier"}]), then=0)],
        otherwise=1, branches=[_leaf(0), _leaf(-1)],
    )
    parameters = {"vip_tier": ParameterInfo(type="String", default_value="gold")}
    df = pl.DataFrame({"status": ["gold", "silver", "bronze"]})
    out = _out("vip", default="standard")
    assert _results(run, rule, out, df, parameters=parameters) == ["vip", "standard", "standard"]
    assert _results(run, rule, out, df, {"t": {"vip_tier": "silver"}}, parameters=parameters) == [
        "standard", "vip", "standard"]


def test_prioritized_rules_read_and_retune_params(run):
    tree = _prioritized(
        [("premium", _if(UnaryGreaterThanEqual(feature="score", threshold={"key": "premium_thresh"}), 0)),
         ("basic", _if(UnaryGreaterThanEqual(feature="score", threshold={"key": "basic_thresh"}), 1))],
        _out("premium", "basic", default="rejected"),
        parameters={"premium_thresh": ParameterInfo(type="Float64", default_value=80.0),
                    "basic_thresh": ParameterInfo(type="Float64", default_value=50.0)},
    )
    df = pl.DataFrame({"score": [90.0, 60.0, 30.0]})
    assert run(tree, df)["r"].to_list() == ["premium", "basic", "rejected"]
    retuned = run(tree, df, {"t": {"premium_thresh": 55.0, "basic_thresh": 50.0}})
    assert retuned["r"].to_list() == ["premium", "premium", "rejected"]
