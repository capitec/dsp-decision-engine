import pydantic
import pytest

from decider.steps.trees import (
    CasesIsIn,
    CasesIsInRule,
    CasesRanges,
    CasesRangesRule,
    CasesStringMatch,
    CasesStringMatchRule,
    CompositeCondition,
    CompositeNode,
    FlatRuleDocument,
    IsInCondition,
    LeafNode,
    LogicOp,
    RangeCondition,
    StringMatchCondition,
    UnaryBetween,
    UnaryGreaterThan,
    UnaryIsIn,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
    V3TreeDocument,
)
from decider.steps.values import ParamRef


def test_a_computed_feature_reports_the_columns_it_reads():
    cond = UnaryLessThan(feature={"type": "computed", "expression": "income - expenses"}, threshold=1.0)
    assert cond.feature.required_features() == {"income", "expenses"}


def test_computed_feature_two_column_expression_parses():
    cond = UnaryLessThan(feature={"type": "computed", "expression": "amount * quantity"}, threshold=100.0)
    assert cond.required_features() == {"amount", "quantity"}


def test_computed_feature_p_dot_attribute_syntax_is_refused():
    with pytest.raises(pydantic.ValidationError, match="attribute access"):
        UnaryLessThan(feature={"type": "computed", "expression": "amount + p.bonus"}, threshold=100.0)


def test_computed_feature_dependencies_are_wired_like_any_other_feature():
    doc = V3TreeDocument(
        nodes=[
            {"id": "root", "data": UnaryNode(condition=UnaryGreaterThan(
                feature={"type": "computed", "expression": "x - y"}, threshold=10.0))},
            {"id": "hi", "data": LeafNode(result_idx=0)},
        ],
        edges=[{"source": "root", "target": "hi", "data": {"sourceIndex": [0]}}],
    )
    assert doc.to_tree().required_features() == {"x", "y"}


def test_a_computed_feature_keeps_its_wire_format():
    cond = UnaryGreaterThan(feature={"type": "computed", "expression": "x - y"}, threshold=10.0)
    assert cond.model_dump()["feature"] == {"type": "computed", "expression": "x - y"}
    assert str(cond.feature) == "x - y"


def test_strict_range_validation_requires_contiguous_bands():
    with pytest.raises(ValueError, match="not continuous"):
        CasesRanges(feature="x", conditions=[RangeCondition(max=10.0), RangeCondition(min=20.0)], strict=True)


def test_strict_range_validation_requires_sorted_bands():
    with pytest.raises(ValueError, match="sorted order"):
        CasesRanges(feature="x", conditions=[RangeCondition(min=20.0, max=30.0), RangeCondition(min=10.0, max=20.0)])


def test_strict_max_only_bands_must_be_sorted():
    with pytest.raises(ValueError, match="sorted order"):
        CasesRanges(feature="x", conditions=[RangeCondition(max=30.0), RangeCondition(max=10.0)])


def test_non_strict_ranges_skip_order_checks():
    CasesRanges(feature="x", conditions=[RangeCondition(max=10.0), RangeCondition(min=20.0)], strict=False)


def test_strict_range_validation_skips_param_bounds():
    CasesRanges(
        feature="x",
        conditions=[RangeCondition(max={"param": "lo"}), RangeCondition(min={"param": "lo"}, max=5.0),
                    RangeCondition(min=10.0)],
    )


def test_a_range_band_needs_a_bound():
    with pytest.raises(ValueError, match="min or max"):
        RangeCondition()


def test_between_needs_a_bound():
    with pytest.raises(ValueError, match="min or max"):
        UnaryBetween(feature="x")


def test_isin_needs_a_value():
    with pytest.raises(ValueError, match="at least 1 item"):
        UnaryIsIn(feature="x", values=[])


def test_string_match_needs_a_pattern():
    with pytest.raises(ValueError, match="at least 1 item"):
        UnaryStringMatch(feature="s", patterns=[])
    with pytest.raises(ValueError, match="at least 1 item"):
        StringMatchCondition(patterns=[])


def test_composite_with_no_conditions_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="at least one condition"):
        CompositeNode(op=LogicOp.AND, conditions=[])
    with pytest.raises(pydantic.ValidationError, match="at least one condition"):
        CompositeCondition(op="or", conditions=[])


def test_not_takes_exactly_one_condition():
    two = [UnaryGreaterThan(feature="x", threshold=1.0), UnaryLessThan(feature="x", threshold=5.0)]
    with pytest.raises(pydantic.ValidationError, match="exactly 1"):
        CompositeNode(op="not", conditions=two)


def test_nested_composite_conditions_parse_from_json():
    node = CompositeNode.model_validate({
        "op": "or",
        "conditions": [
            {"type": "composite", "op": "and", "conditions": [
                {"op": ">", "feature": "x", "threshold": 0.0},
                {"op": "<", "feature": "x", "threshold": 10.0},
            ]},
            {"op": ">", "feature": "x", "threshold": 20.0},
        ],
    })
    assert isinstance(node.conditions[0], CompositeCondition)
    assert isinstance(node.conditions[1], UnaryGreaterThan)


def test_an_unknown_operator_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        UnaryNode.model_validate({"condition": {"op": "is_null", "feature": "x"}})


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (0.7, 0.7),
        ({"param": "hi_thresh", "default": 0.7}, ParamRef(param="hi_thresh", default=0.7)),
        ({"param": "base_rate", "shared": True}, ParamRef(param="base_rate", shared=True)),
        ({"key": "floor"}, ParamRef(param="floor")),
    ],
)
def test_a_threshold_is_a_literal_or_a_param_ref(threshold, expected):
    assert UnaryLessThan(feature="x", threshold=threshold).threshold == expected


def test_an_int_threshold_stays_an_int_so_large_ids_compare_exactly():
    big = 2**53 + 1
    threshold = UnaryLessThan(feature="x", threshold=big).threshold
    assert threshold == big and isinstance(threshold, int)


def test_a_threshold_is_not_an_arbitrary_string():
    with pytest.raises(pydantic.ValidationError):
        UnaryLessThan(feature="x", threshold="thirty")


def test_bounds_values_and_patterns_accept_param_refs():
    assert UnaryBetween(feature="x", min={"key": "lo"}, max=5).min == ParamRef(param="lo")
    assert UnaryIsIn(feature="x", values={"key": "allowed"}).values == ParamRef(param="allowed")
    assert UnaryIsIn(feature="x", values=[1, {"param": "p"}]).values == [1, ParamRef(param="p")]
    assert UnaryStringMatch(feature="s", patterns=["a", {"key": "tier"}]).patterns == ["a", ParamRef(param="tier")]


def _flat_tree(rule):
    output = {"data": [{"r": "a"}], "default": {"r": "d"}, "dtypes": [("r", "String")]}
    return FlatRuleDocument(rule={"rule": rule}, output=output).to_tree()


def test_cases_ranges_required_params_include_param_bounds():
    rule = CasesRangesRule(
        feature="x",
        conditions=[{"when": RangeCondition(min={"key": "lo"}, max={"key": "hi"}), "then": 0}],
        otherwise=1,
        branches=[LeafNode(result_idx=0), LeafNode(result_idx=-1)],
        strict=False,
    )
    assert {"lo", "hi"} <= _flat_tree(rule).required_params()
    node = CasesRanges(feature="x", conditions=[RangeCondition(min={"key": "lo"}, max={"key": "hi"})], strict=False)
    assert {"lo", "hi"} <= node.required_params()


def test_cases_string_match_required_params_include_param_patterns():
    rule = CasesStringMatchRule(
        feature="s",
        conditions=[{"when": StringMatchCondition(patterns=[{"key": "pat"}]), "then": 0}],
        otherwise=1,
        branches=[LeafNode(result_idx=0), LeafNode(result_idx=-1)],
    )
    assert "pat" in _flat_tree(rule).required_params()
    node = CasesStringMatch(feature="s", conditions=[StringMatchCondition(patterns=[{"key": "pat"}])])
    assert "pat" in node.required_params()


def test_cases_isin_required_params_include_a_param_value_set():
    rule = CasesIsInRule(
        feature="x",
        conditions=[{"when": IsInCondition(values={"key": "allowed"}), "then": 0}],
        otherwise=1,
        branches=[LeafNode(result_idx=0), LeafNode(result_idx=-1)],
    )
    assert "allowed" in _flat_tree(rule).required_params()
    assert "allowed" in CasesIsIn(feature="x", conditions=[IsInCondition(values={"key": "allowed"})]).required_params()


def test_cases_with_no_conditions_are_accepted():
    for node in (
        CasesRanges(feature="x", conditions=[], strict=False),
        CasesStringMatch(feature="s", conditions=[]),
        CasesIsIn(feature="x", conditions=[]),
    ):
        assert node.required_features() == {node.feature.root}
