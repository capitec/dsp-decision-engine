"""
Integration tests for rule conditions — unary, cases, composite, special values.

Each test builds a small FlatRuleModule, executes it over a DataFrame, and
checks the output values. No shape/dtype assertions — just "does it give the
right answer?"
"""

import math
import polars as pl
import pytest
from decider.modules.rules import (
    FlatRuleModule,
    PrioritizedFlatRuleModule,
    CasesRanges,
    CasesIsIn,
    CasesStringMatch,
    CasesBranch,
    CasesRule,
    CompositeRule,
    UnaryRule,
    LeafRule,
    RuleRoot,
    RuleMeta,
    TreeOutput,
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
    CompositeCondition,
    TLogicOp,
    RangeEndLogic,
    UnaryLessThan,
    UnaryLessThanEqual,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryNotEqual,
    UnaryBetween,
    UnaryIsIn,
    UnaryStringMatch,
    UnaryIsNull,
    UnaryIsNotNull,
    UnaryIsTrue,
    UnaryIsFalse,
    InputRef,
    ParameterInfo,
)
from decider.serializable.schema import PrimitiveSchema


# ---------------------------------------------------------------------------
# Helpers (path tracking — gap 3/4/5 additions use these)
# ---------------------------------------------------------------------------

def _run_with_path(rule, output: TreeOutput, df: pl.DataFrame) -> pl.DataFrame:
    """Run a rule with path tracking; returns DataFrame with columns 'r' and 'Path'."""
    from decider.modules.rules.flat_rules.nodes import BuilderConfig
    from decider.modules.rules.flat_rules.impl import execute_rule_root

    def path_output_fn(inputs, branch_stack, config, result_idx):
        result_value = config.default_expr if result_idx == -1 else config.output_literals[result_idx]
        path_parts = []
        for item in branch_stack:
            idx = item.index if item.index is not None else _branch_count(item.rule)
            feature = _rule_feature(item.rule)
            path_parts.append(f"{feature},{idx}")
        path_str = "|".join(path_parts) if path_parts else "root"
        return pl.struct(result_value.alias("r"), pl.lit(path_str).alias("Path"))

    config = BuilderConfig(
        build_result_function=path_output_fn,
        output_literals=output.output_literals,
        default_literal=output.default_literal,
    )
    from decider.modules.rules.flat_rules.nodes import RuleRoot, RuleMeta
    expr = execute_rule_root(RuleRoot(meta=RuleMeta(), rule=rule), config)
    return df.select(expr.struct.unnest())


def _branch_count(rule) -> int:
    from decider.modules.rules.flat_rules.nodes import (
        CasesRanges, CasesStringMatch, CasesIsIn, UnaryRule, CompositeRule
    )
    if isinstance(rule, (CasesRanges, CasesStringMatch, CasesIsIn)):
        return len(rule.root.conditions) if hasattr(rule, "root") else len(rule.conditions)
    if isinstance(rule, (UnaryRule, CompositeRule)):
        return 1
    return 0


def _rule_feature(rule) -> str:
    from decider.modules.rules.flat_rules.nodes import (
        CasesRanges, CasesStringMatch, CasesIsIn, UnaryRule, CompositeRule
    )
    if isinstance(rule, (CasesRanges, CasesStringMatch, CasesIsIn)):
        inner = rule.root if hasattr(rule, "root") else rule
        return str(inner.feature)
    if isinstance(rule, UnaryRule):
        return str(rule.condition.feature)
    if isinstance(rule, CompositeRule):
        return rule.id or "composite"
    return "leaf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output(*labels: str, default: str = "default") -> TreeOutput:
    return TreeOutput(
        data=[{"r": lbl} for lbl in labels],
        default={"r": default},
        dtypes=[("r", "String")],
        type_defs={},
    )


def _run(rule, output: TreeOutput, df: pl.DataFrame) -> list:
    m = FlatRuleModule(output=output, rule=RuleRoot(meta=RuleMeta(), rule=rule))
    return m({"input": df.lazy()})["r"].to_list()


# ---------------------------------------------------------------------------
# Unary numeric operators
# ---------------------------------------------------------------------------

def test_all_numeric_comparison_operators():
    """All six comparison operators route correctly for boundary and non-boundary values."""
    df = pl.DataFrame({"x": [5.0, 10.0, 15.0]})
    # x < 10  → "low", else "high"
    assert _run(UnaryRule(condition=UnaryLessThan(feature="x", threshold=10.0),    then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1)), _output("low", "high"), df) == ["low", "high", "high"]
    assert _run(UnaryRule(condition=UnaryLessThanEqual(feature="x", threshold=10.0), then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1)), _output("low", "high"), df) == ["low", "low", "high"]
    assert _run(UnaryRule(condition=UnaryEqual(feature="x", threshold=10.0),          then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1)), _output("yes", "no"),   df) == ["no", "yes", "no"]
    assert _run(UnaryRule(condition=UnaryGreaterThan(feature="x", threshold=10.0),    then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1)), _output("high", "low"), df) == ["low", "low", "high"]
    assert _run(UnaryRule(condition=UnaryGreaterThanEqual(feature="x", threshold=10.0), then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1)), _output("high", "low"), df) == ["low", "high", "high"]
    assert _run(UnaryRule(condition=UnaryNotEqual(feature="x", threshold=10.0),        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1)), _output("ne", "eq"),    df) == ["ne", "eq", "ne"]


def test_between_variants():
    """Between with min-only, max-only, and both bounds."""
    df = pl.DataFrame({"x": [5.0, 15.0, 25.0, 35.0]})
    out = _output("yes", default="no")

    # both bounds: (10, 20]  — upper_inclusive end_logic
    rule_both = UnaryRule(condition=UnaryBetween(feature="x", min=10.0, max=20.0), then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1))
    assert _run(rule_both, out, df) == ["no", "yes", "no", "no"]

    # min only: >= 20
    rule_min = UnaryRule(condition=UnaryBetween(feature="x", min=20.0), then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1))
    assert _run(rule_min, out, df) == ["no", "no", "yes", "yes"]

    # max only: <= 10
    rule_max = UnaryRule(condition=UnaryBetween(feature="x", max=10.0), then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1))
    assert _run(rule_max, out, df) == ["yes", "no", "no", "no"]


def test_null_and_boolean_operators():
    """is_null / is_not_null / is_true / is_false."""
    # null checks
    df_null = pl.DataFrame({"x": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    out = _output("null", "not_null", default="?")
    rule_null = UnaryRule(condition=UnaryIsNull(feature="x"),    then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1))
    assert _run(rule_null, out, df_null) == ["not_null", "null", "not_null"]
    rule_notnull = UnaryRule(condition=UnaryIsNotNull(feature="x"), then=LeafRule(result_idx=1), otherwise=LeafRule(result_idx=0))
    assert _run(rule_notnull, out, df_null) == ["not_null", "null", "not_null"]

    # boolean checks
    df_bool = pl.DataFrame({"flag": [True, False, True]})
    out2 = _output("T", "F", default="?")
    rule_true  = UnaryRule(condition=UnaryIsTrue(feature="flag"),  then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1))
    rule_false = UnaryRule(condition=UnaryIsFalse(feature="flag"), then=LeafRule(result_idx=1), otherwise=LeafRule(result_idx=0))
    assert _run(rule_true,  out2, df_bool) == ["T", "F", "T"]
    assert _run(rule_false, out2, df_bool) == ["T", "F", "T"]


# ---------------------------------------------------------------------------
# String conditions
# ---------------------------------------------------------------------------

def test_string_match_types():
    """exact, contains, starts_with, ends_with, regex all route correctly."""
    df = pl.DataFrame({"s": ["hello world", "world", "hello", "goodbye"]})
    out = _output("match", default="no")

    cases = [
        ("exact",       "world",    [False, True, False, False]),
        ("contains",    "ello",     [True, False, True, False]),
        ("starts_with", "hello",    [True, False, True, False]),
        ("ends_with",   "orld",     [True, True, False, False]),
        ("regex",       "^h.*d$",   [True, False, False, False]),
    ]
    for match_type, pattern, expected_match in cases:
        rule = UnaryRule(
            condition=UnaryStringMatch(feature="s", patterns=[pattern], match_type=match_type),
            then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
        )
        result = _run(rule, out, df)
        expected = ["match" if m else "no" for m in expected_match]
        assert result == expected, f"match_type={match_type} pattern={pattern!r}: {result} != {expected}"


def test_string_match_case_insensitive_and_trim():
    """Case-insensitive and whitespace-trimmed matching."""
    df = pl.DataFrame({"s": ["  HELLO  ", "hello", "WORLD"]})
    out = _output("match", default="no")

    rule_ci = UnaryRule(
        condition=UnaryStringMatch(feature="s", patterns=["hello"], match_type="exact", case_sensitive=False),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(rule_ci, out, df) == ["no", "match", "no"]

    rule_trim = UnaryRule(
        condition=UnaryStringMatch(feature="s", patterns=["HELLO"], match_type="exact", case_sensitive=True, trim_whitespace=True),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(rule_trim, out, df) == ["match", "no", "no"]


# ---------------------------------------------------------------------------
# Cases (multi-way branching)
# ---------------------------------------------------------------------------

def test_cases_ranges_lower_and_upper_inclusive():
    """Range cases with lower_inclusive and upper_inclusive end logic."""
    df = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]})
    out = _output("low", "mid", "high", default="none")

    def _cases(end_logic):
        return CasesRule(root=CasesRanges(
            feature="score",
            conditions=[
                CasesBranch(when=RangeCondition(max=30.0), then=0),
                CasesBranch(when=RangeCondition(min=30.0, max=70.0), then=1),
                CasesBranch(when=RangeCondition(min=70.0), then=2),
            ],
            otherwise=3,
            branches=[LeafRule(result_idx=0), LeafRule(result_idx=1), LeafRule(result_idx=2), LeafRule(result_idx=-1)],
            end_logic=end_logic,
            strict=False,
        ))

    # lower_inclusive: [min, max) — 30 lands in mid
    r_li = _run(_cases(RangeEndLogic.lower_inclusive), out, df)
    assert r_li[1] == "mid"   # score=30 → mid (inclusive lower)
    assert r_li[3] == "high"  # score=70 → high (inclusive lower)

    # upper_inclusive: (min, max] — 30 lands in low
    r_ui = _run(_cases(RangeEndLogic.upper_inclusive), out, df)
    assert r_ui[1] == "low"   # score=30 → low (inclusive upper)
    assert r_ui[3] == "mid"   # score=70 → mid (inclusive upper)


def test_cases_isin():
    """IsIn cases route numeric categoricals to the right branch."""
    out = _output("grp1", "grp2", default="other")

    rule = CasesRule(root=CasesIsIn(
        feature="code",
        conditions=[
            CasesBranch(when=IsInCondition(values=[1, 2]), then=0),
            CasesBranch(when=IsInCondition(values=[3, 4]), then=1),
        ],
        otherwise=2,
        branches=[LeafRule(result_idx=0), LeafRule(result_idx=1), LeafRule(result_idx=-1)],
    ))
    assert _run(rule, out, pl.DataFrame({"code": [1, 3, 5]})) == ["grp1", "grp2", "other"]


def test_cases_string_match():
    """String match cases route patterns to correct branches."""
    df = pl.DataFrame({"s": ["apple", "banana", "cherry", "durian"]})
    out = _output("A", "B", default="other")

    rule = CasesRule(root=CasesStringMatch(
        feature="s",
        match_type="starts_with",
        conditions=[
            CasesBranch(when=StringMatchCondition(patterns=["a"]), then=0),
            CasesBranch(when=StringMatchCondition(patterns=["b", "c"]), then=1),
        ],
        otherwise=2,
        branches=[LeafRule(result_idx=0), LeafRule(result_idx=1), LeafRule(result_idx=-1)],
    ))
    assert _run(rule, out, df) == ["A", "B", "B", "other"]


# ---------------------------------------------------------------------------
# Composite conditions
# ---------------------------------------------------------------------------

def test_composite_and_or_not():
    """AND, OR, NOT composites all produce correct results."""
    df = pl.DataFrame({"x": [3.0, 7.0, 12.0]})
    out = _output("yes", default="no")

    # AND: 5 < x < 10  → only 7 matches
    and_rule = CompositeRule(
        op=TLogicOp.AND,
        conditions=[UnaryGreaterThan(feature="x", threshold=5.0), UnaryLessThan(feature="x", threshold=10.0)],
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(and_rule, out, df) == ["no", "yes", "no"]

    # OR: x < 5 OR x > 10  → 3 and 12 match
    or_rule = CompositeRule(
        op=TLogicOp.OR,
        conditions=[UnaryLessThan(feature="x", threshold=5.0), UnaryGreaterThan(feature="x", threshold=10.0)],
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(or_rule, out, df) == ["yes", "no", "yes"]

    # NOT: NOT(x > 5)  → only 3 matches
    not_rule = CompositeRule(
        op=TLogicOp.NOT,
        conditions=[UnaryGreaterThan(feature="x", threshold=5.0)],
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(not_rule, out, df) == ["yes", "no", "no"]


def test_nested_composite():
    """(x > 0 AND x < 10) OR x > 20 using nested CompositeCondition inside CompositeRule."""
    df = pl.DataFrame({"x": [-5.0, 5.0, 15.0, 25.0]})
    out = _output("yes", default="no")

    inner = CompositeCondition(
        op=TLogicOp.AND,
        conditions=[UnaryGreaterThan(feature="x", threshold=0.0), UnaryLessThan(feature="x", threshold=10.0)],
    )
    outer = CompositeRule(
        op=TLogicOp.OR,
        conditions=[inner, UnaryGreaterThan(feature="x", threshold=20.0)],
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(outer, out, df) == ["no", "yes", "no", "yes"]


# ---------------------------------------------------------------------------
# Default / leaf behaviour
# ---------------------------------------------------------------------------

def test_default_returned_when_no_match():
    """result_idx=-1 always returns the default, not an output row."""
    df = pl.DataFrame({"x": [100.0, 200.0]})
    rule = UnaryRule(
        condition=UnaryLessThan(feature="x", threshold=0.0),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    out = _output("match", default="FALLBACK")
    assert _run(rule, out, df) == ["FALLBACK", "FALLBACK"]


def test_correct_output_row_indexed():
    """result_idx correctly selects a row from the output table."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    rule = CasesRule(root=CasesIsIn(
        feature="x",
        conditions=[
            CasesBranch(when=IsInCondition(values=[1.0]), then=0),
            CasesBranch(when=IsInCondition(values=[2.0]), then=1),
            CasesBranch(when=IsInCondition(values=[3.0]), then=2),
        ],
        otherwise=3,
        branches=[LeafRule(result_idx=0), LeafRule(result_idx=1), LeafRule(result_idx=2), LeafRule(result_idx=-1)],
    ))
    out = _output("row0", "row1", "row2", default="none")
    assert _run(rule, out, df) == ["row0", "row1", "row2"]


# ---------------------------------------------------------------------------
# Special numeric values
# ---------------------------------------------------------------------------

def test_special_numeric_values_through_range_rules():
    """inf, -inf, nan, None all handled correctly through numeric range conditions."""
    df = pl.DataFrame({"v": pl.Series([1.0, float("inf"), float("-inf"), float("nan"), None], dtype=pl.Float64)})

    # Only finite positives match [0, 100)
    rule = UnaryRule(
        condition=UnaryBetween(feature="v", min=0.0, max=100.0),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    out = _output("match", default="no")
    results = _run(rule, out, df)
    assert results[0] == "match"        # 1.0 in [0, 100)
    assert results[1] == "no"           # inf not in range
    assert results[2] == "no"           # -inf not in range
    # nan and None do not match numeric comparisons
    assert results[3] == "no"
    assert results[4] == "no"


# ---------------------------------------------------------------------------
# Prioritized flat rules
# ---------------------------------------------------------------------------

def test_prioritized_first_match_wins():
    """First matching rule wins; subsequent rules are skipped even if they would also match.

    Rules (in priority order):
      r1: score < 50  → "under50"
      r2: score < 100 → "under100"

    score=10  → r1 matches first → "under50"  (r2 would also match but is skipped)
    score=70  → r1 doesn't match, r2 matches  → "under100"
    score=150 → neither matches               → "over100"
    """
    r1 = RuleRoot(meta=RuleMeta(name="under50"), rule=UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=50.0),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    ))
    r2 = RuleRoot(meta=RuleMeta(name="under100"), rule=UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=100.0),
        then=LeafRule(result_idx=1), otherwise=LeafRule(result_idx=-1),
    ))
    m = PrioritizedFlatRuleModule(
        output=_output("under50", "under100", default="over100"),
        rules=[r1, r2],
    )
    df = pl.DataFrame({"score": [10.0, 70.0, 150.0]})
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["under50", "under100", "over100"]


def test_prioritized_falls_back_to_default():
    """When no rule matches any row, the default output is returned for all."""
    df = pl.DataFrame({"score": [200.0, 300.0]})
    r1 = RuleRoot(meta=RuleMeta(name="low"), rule=UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=100.0),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    ))
    m = PrioritizedFlatRuleModule(
        output=_output("low", default="high"),
        rules=[r1],
    )
    assert m({"input": df.lazy()})["r"].to_list() == ["high", "high"]


# ---------------------------------------------------------------------------
# Gap 3: UnaryIsIn operator
# ---------------------------------------------------------------------------

def test_unary_is_in():
    """UnaryIsIn routes rows whose value is in the list to then, others to otherwise."""
    df = pl.DataFrame({"code": [1, 2, 3, 4, 5]})
    out = _output("allowed", default="denied")
    rule = UnaryRule(
        condition=UnaryIsIn(feature="code", values=[1, 3, 5]),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    assert _run(rule, out, df) == ["allowed", "denied", "allowed", "denied", "allowed"]


# ---------------------------------------------------------------------------
# Gap 4: InputRef bounds inside Cases nodes
# ---------------------------------------------------------------------------

def test_cases_ranges_inputref_bounds():
    """CasesRanges where min/max come from InputRef parameters, not static values."""
    df = pl.DataFrame({
        "score": [10.0, 40.0, 80.0],
        "parameters": [{"lo": 20.0, "hi": 60.0}] * 3,
    })
    out = _output("low", "mid", default="high")

    rule = CasesRule(root=CasesRanges(
        feature="score",
        conditions=[
            CasesBranch(when=RangeCondition(max=InputRef(key="lo")), then=0),
            CasesBranch(when=RangeCondition(min=InputRef(key="lo"), max=InputRef(key="hi")), then=1),
        ],
        otherwise=2,
        branches=[LeafRule(result_idx=0), LeafRule(result_idx=1), LeafRule(result_idx=-1)],
        end_logic=RangeEndLogic.lower_inclusive,
        strict=False,
    ))
    m = FlatRuleModule(
        output=out,
        rule=RuleRoot(meta=RuleMeta(), rule=rule),
        parameters={
            "lo": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=20.0),
            "hi": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=60.0),
        },
    )
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["low", "mid", "high"]


def test_cases_string_match_inputref_pattern():
    """CasesStringMatch where a pattern comes from an InputRef parameter."""
    df = pl.DataFrame({
        "status": ["gold", "silver", "bronze"],
        "parameters": [{"vip_tier": "gold"}] * 3,
    })
    out = _output("vip", default="standard")

    rule = CasesRule(root=CasesStringMatch(
        feature="status",
        match_type="exact",
        conditions=[
            CasesBranch(when=StringMatchCondition(patterns=[InputRef(key="vip_tier")]), then=0),
        ],
        otherwise=1,
        branches=[LeafRule(result_idx=0), LeafRule(result_idx=-1)],
    ))
    m = FlatRuleModule(
        output=out,
        rule=RuleRoot(meta=RuleMeta(), rule=rule),
        parameters={
            "vip_tier": ParameterInfo(type=PrimitiveSchema(type="String"), default_value="gold"),
        },
    )
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["vip", "standard", "standard"]


# ---------------------------------------------------------------------------
# Gap 5: PrioritizedFlatRuleModule + parameters
# ---------------------------------------------------------------------------

def test_prioritized_module_with_parameters():
    """Parameters flow correctly through a PrioritizedFlatRuleModule."""
    r1 = RuleRoot(meta=RuleMeta(name="premium"), rule=UnaryRule(
        condition=UnaryGreaterThanEqual(feature="score", threshold=InputRef(key="premium_thresh")),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    ))
    r2 = RuleRoot(meta=RuleMeta(name="basic"), rule=UnaryRule(
        condition=UnaryGreaterThanEqual(feature="score", threshold=InputRef(key="basic_thresh")),
        then=LeafRule(result_idx=1), otherwise=LeafRule(result_idx=-1),
    ))
    m = PrioritizedFlatRuleModule(
        output=_output("premium", "basic", default="rejected"),
        rules=[r1, r2],
        parameters={
            "premium_thresh": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=80.0),
            "basic_thresh": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=50.0),
        },
    )
    df = pl.DataFrame({"score": [90.0, 60.0, 30.0]})
    result = m({"input": df.lazy()})["r"].to_list()
    # score=90 → premium (>=80), score=60 → basic (>=50 but not >=80), score=30 → rejected
    assert result == ["premium", "basic", "rejected"]

    # Override premium_thresh at runtime — score=60 now qualifies for premium
    df_rt = pl.DataFrame({
        "score": [90.0, 60.0, 30.0],
        "parameters": [{"premium_thresh": 55.0, "basic_thresh": 50.0}] * 3,
    })
    result_rt = m({"input": df_rt.lazy()})["r"].to_list()
    assert result_rt == ["premium", "premium", "rejected"]
