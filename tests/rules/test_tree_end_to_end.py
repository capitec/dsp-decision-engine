"""
End-to-end tests for larger rule flows.

Covers:
- Multi-level nested rule trees (UnaryRule → CasesRanges → LeafRule)
- PrioritizedFlatRuleModule in `all` mode
- Path tracking via a custom output_fn that records the branch_stack
"""

import polars as pl
import pytest
from decider.modules.rules import (
    FlatRuleModule,
    PrioritizedFlatRuleModule,
    PrioritizationMode,
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
    TLogicOp,
    RangeEndLogic,
    UnaryLessThan,
    UnaryLessThanEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryBetween,
    UnaryIsIn,
    UnaryIsNull,
    UnaryStringMatch,
)
from decider.modules.rules.flat_rules.nodes import (
    BuilderConfig,
    TBranchStack,
    IndexedBranch,
)
from decider.modules.rules.flat_rules.impl import execute_rule_root


# ---------------------------------------------------------------------------
# Path tracking helpers
# ---------------------------------------------------------------------------

def _get_feature(rule) -> str:
    if isinstance(rule, (CasesRanges, CasesStringMatch, CasesIsIn)):
        inner = rule.root if hasattr(rule, "root") else rule
        return str(inner.feature)
    if isinstance(rule, UnaryRule):
        return str(rule.condition.feature)
    if isinstance(rule, CompositeRule):
        return rule.id or "composite"
    return "leaf"


def _branch_count(rule) -> int:
    if isinstance(rule, (CasesRanges, CasesStringMatch, CasesIsIn)):
        inner = rule.root if hasattr(rule, "root") else rule
        return len(inner.conditions)
    if isinstance(rule, (UnaryRule, CompositeRule)):
        return 1
    return 0


def _path_output_fn(inputs, branch_stack: TBranchStack, config: BuilderConfig, result_idx: int) -> pl.Expr:
    """Custom output function that returns {r, path} struct.

    output_literals are already struct exprs (e.g. {r: "low"}), so extract
    the inner "r" field before building the flat {r, path} struct.
    """
    literal_struct = config.default_expr if result_idx == -1 else config.output_literals[result_idx]
    r_expr = literal_struct.struct.field("r")
    parts = []
    for item in branch_stack:
        idx = item.index if item.index is not None else _branch_count(item.rule)
        parts.append(f"{_get_feature(item.rule)},{idx}")
    path_str = "|".join(parts) if parts else "root"
    return pl.struct(r_expr.alias("r"), pl.lit(path_str).alias("path"))


def _run_with_path(rule, output: TreeOutput, df: pl.DataFrame) -> pl.DataFrame:
    config = BuilderConfig(
        build_result_function=_path_output_fn,
        output_literals=output.output_literals,
        default_literal=output.default_literal,
    )
    features = {col: pl.col(col) for col in rule.get_required_features()}
    expr = execute_rule_root(RuleRoot(meta=RuleMeta(), rule=rule), config, **features)
    return df.select(expr.struct.unnest())


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
# Nested tree flows
# ---------------------------------------------------------------------------

def test_nested_unary_then_cases_ranges():
    """
    Two-level tree:
      age < 30  → CasesRanges on score  (low / mid / high)
      age >= 30 → LeafRule("adult")

    Verifies that the then-branch of a UnaryRule can itself be a CasesRule.
    """
    score_buckets = CasesRule(root=CasesRanges(
        feature="score",
        conditions=[
            CasesBranch(when=RangeCondition(max=40.0), then=0),
            CasesBranch(when=RangeCondition(min=40.0, max=70.0), then=1),
            CasesBranch(when=RangeCondition(min=70.0), then=2),
        ],
        otherwise=3,
        branches=[
            LeafRule(result_idx=0),
            LeafRule(result_idx=1),
            LeafRule(result_idx=2),
            LeafRule(result_idx=-1),
        ],
        end_logic=RangeEndLogic.lower_inclusive,
        strict=False,
    ))

    root = UnaryRule(
        condition=UnaryLessThan(feature="age", threshold=30.0),
        then=score_buckets,
        otherwise=LeafRule(result_idx=3),
    )

    out = _output("young_low", "young_mid", "young_high", "adult", default="unknown")
    df = pl.DataFrame({
        "age":   [20.0, 25.0, 25.0, 35.0],
        "score": [20.0, 55.0, 80.0, 99.0],
    })
    result = _run(root, out, df)
    assert result == ["young_low", "young_mid", "young_high", "adult"]


def test_nested_cases_ranges_then_unary():
    """
    Two-level tree:
      CasesRanges on income  → low bucket → UnaryStringMatch on region → "rural" / "urban"
                              → high bucket → LeafRule("wealthy")

    Verifies Cases branches can themselves be UnaryRule nodes.
    """
    region_check = UnaryRule(
        condition=UnaryStringMatch(feature="region", patterns=["rural", "farm"], match_type="exact"),
        then=LeafRule(result_idx=0),
        otherwise=LeafRule(result_idx=1),
    )

    root = CasesRule(root=CasesRanges(
        feature="income",
        conditions=[
            CasesBranch(when=RangeCondition(max=50_000.0), then=0),
            CasesBranch(when=RangeCondition(min=50_000.0), then=1),
        ],
        otherwise=2,
        branches=[region_check, LeafRule(result_idx=2), LeafRule(result_idx=-1)],
        end_logic=RangeEndLogic.lower_inclusive,
        strict=False,
    ))

    out = _output("rural_low", "urban_low", "wealthy", default="unknown")
    df = pl.DataFrame({
        "income": [30_000.0, 30_000.0, 80_000.0],
        "region": ["rural",  "city",    "city"],
    })
    result = _run(root, out, df)
    assert result == ["rural_low", "urban_low", "wealthy"]


def test_three_level_nested_tree():
    """
    Three-level tree:
      age < 18  → LeafRule("minor")
      age >= 18 → score < 50  → status in ["active"] → LeafRule("active_low")
                               → status not in list  → LeafRule("inactive_low")
               → score >= 50  → LeafRule("high_score")
    """
    status_check = UnaryRule(
        condition=UnaryStringMatch(feature="status", patterns=["active"], match_type="exact"),
        then=LeafRule(result_idx=2),
        otherwise=LeafRule(result_idx=3),
    )
    score_check = UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=50.0),
        then=status_check,
        otherwise=LeafRule(result_idx=4),
    )
    root = UnaryRule(
        condition=UnaryLessThan(feature="age", threshold=18.0),
        then=LeafRule(result_idx=0),
        otherwise=score_check,
    )

    out = _output("minor", "unused", "active_low", "inactive_low", "high_score", default="unknown")
    df = pl.DataFrame({
        "age":    [15.0,  25.0,     25.0,       30.0],
        "score":  [99.0,  30.0,     30.0,       75.0],
        "status": ["x",   "active", "inactive", "active"],
    })
    result = _run(root, out, df)
    assert result == ["minor", "active_low", "inactive_low", "high_score"]


def test_composite_inside_nested_tree():
    """
    UnaryRule (age gate) → CompositeRule (AND: score > 60 AND status == "vip") → leaf
    Verifies CompositeRule works as an embedded branch, not just a root rule.
    """
    vip_check = CompositeRule(
        op=TLogicOp.AND,
        conditions=[
            UnaryGreaterThan(feature="score", threshold=60.0),
            UnaryStringMatch(feature="status", patterns=["vip"], match_type="exact"),
        ],
        then=LeafRule(result_idx=1),
        otherwise=LeafRule(result_idx=2),
    )
    root = UnaryRule(
        condition=UnaryGreaterThanEqual(feature="age", threshold=18.0),
        then=vip_check,
        otherwise=LeafRule(result_idx=0),
    )

    out = _output("minor", "vip_adult", "regular_adult", default="unknown")
    df = pl.DataFrame({
        "age":    [15.0,  25.0,  25.0,   30.0],
        "score":  [90.0,  80.0,  80.0,   40.0],
        "status": ["vip", "vip", "basic", "vip"],
    })
    result = _run(root, out, df)
    assert result == ["minor", "vip_adult", "regular_adult", "regular_adult"]


# ---------------------------------------------------------------------------
# PrioritizedFlatRuleModule — `all` mode
# ---------------------------------------------------------------------------

def test_prioritized_all_mode_returns_each_rule_independently():
    """
    `all` mode evaluates every rule independently and returns them as named struct fields.
    A row that matches rule_a but not rule_b gets rule_a's result and rule_b's default.
    """
    r_fraud = RuleRoot(meta=RuleMeta(name="fraud_flag"), rule=UnaryRule(
        condition=UnaryGreaterThan(feature="amount", threshold=10_000.0),
        then=LeafRule(result_idx=0),
        otherwise=LeafRule(result_idx=-1),
    ))
    r_vip = RuleRoot(meta=RuleMeta(name="vip_flag"), rule=UnaryRule(
        condition=UnaryStringMatch(feature="tier", patterns=["gold", "platinum"], match_type="exact"),
        then=LeafRule(result_idx=1),
        otherwise=LeafRule(result_idx=-1),
    ))

    out = TreeOutput(
        data=[{"r": "high_amount"}, {"r": "vip"}],
        default={"r": "none"},
        dtypes=[("r", "String")],
        type_defs={},
    )
    m = PrioritizedFlatRuleModule(
        output=out,
        rules=[r_fraud, r_vip],
        mode=PrioritizationMode.all,
    )
    df = pl.DataFrame({
        "amount": [500.0,   15_000.0, 15_000.0],
        "tier":   ["gold",  "silver", "gold"],
    })
    result = m({"input": df.lazy()})

    # Each rule column is a struct — unnest to access the "r" field
    fraud = result.select(pl.col("fraud_flag").struct.field("r"))["r"].to_list()
    vip   = result.select(pl.col("vip_flag").struct.field("r"))["r"].to_list()

    # Row 0: amount=500 (no fraud), tier=gold (vip)
    assert fraud[0] == "none"
    assert vip[0] == "vip"

    # Row 1: amount=15000 (fraud), tier=silver (no vip)
    assert fraud[1] == "high_amount"
    assert vip[1] == "none"

    # Row 2: both match
    assert fraud[2] == "high_amount"
    assert vip[2] == "vip"


def test_prioritized_all_mode_all_rules_default():
    """In `all` mode, when no rule matches any row, every field returns the default."""
    r1 = RuleRoot(meta=RuleMeta(name="a"), rule=UnaryRule(
        condition=UnaryLessThan(feature="x", threshold=0.0),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    ))
    r2 = RuleRoot(meta=RuleMeta(name="b"), rule=UnaryRule(
        condition=UnaryGreaterThan(feature="x", threshold=100.0),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    ))
    out = _output("match", default="none")
    m = PrioritizedFlatRuleModule(output=out, rules=[r1, r2], mode=PrioritizationMode.all)
    df = pl.DataFrame({"x": [50.0, 50.0]})
    result = m({"input": df.lazy()})
    a = result.select(pl.col("a").struct.field("r"))["r"].to_list()
    b = result.select(pl.col("b").struct.field("r"))["r"].to_list()
    assert a == ["none", "none"]
    assert b == ["none", "none"]


# ---------------------------------------------------------------------------
# Path tracking
# ---------------------------------------------------------------------------

def test_path_unary_then_branch():
    """then-branch path records feature name with index 0."""
    rule = UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=50.0),
        then=LeafRule(result_idx=0),
        otherwise=LeafRule(result_idx=-1),
    )
    out = _output("low", default="high")
    df = pl.DataFrame({"score": [30.0, 70.0]})
    result = _run_with_path(rule, out, df)

    assert result["r"].to_list() == ["low", "high"]
    # then-branch: index 0; otherwise: index 1 (branch_count for UnaryRule)
    assert result["path"][0] == "score,0"
    assert result["path"][1] == "score,1"


def test_path_otherwise_branch_index_is_branch_count():
    """otherwise path index equals the number of named branches (convention for 'no match')."""
    rule = CasesRule(root=CasesIsIn(
        feature="code",
        conditions=[
            CasesBranch(when=IsInCondition(values=[1]), then=0),
            CasesBranch(when=IsInCondition(values=[2]), then=1),
        ],
        otherwise=2,
        branches=[LeafRule(result_idx=0), LeafRule(result_idx=1), LeafRule(result_idx=-1)],
    ))
    out = _output("one", "two", default="other")
    df = pl.DataFrame({"code": [1, 2, 9]})
    result = _run_with_path(rule, out, df)

    assert result["r"].to_list() == ["one", "two", "other"]
    assert result["path"][0] == "code,0"   # first condition matched
    assert result["path"][1] == "code,1"   # second condition matched
    assert result["path"][2] == "code,2"   # otherwise = branch_count (2 conditions)


def test_path_nested_tree_depth_reflects_decisions():
    """
    Deeper paths produce longer path strings; each level appends a segment.
    Tree: age < 30 → score < 50 → "young_low" else "young_high"
          age >= 30 → "adult"
    """
    inner = UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=50.0),
        then=LeafRule(result_idx=0),
        otherwise=LeafRule(result_idx=1),
    )
    root = UnaryRule(
        condition=UnaryLessThan(feature="age", threshold=30.0),
        then=inner,
        otherwise=LeafRule(result_idx=2),
    )
    out = _output("young_low", "young_high", "adult", default="unknown")
    df = pl.DataFrame({
        "age":   [20.0, 20.0, 40.0],
        "score": [30.0, 70.0, 99.0],
    })
    result = _run_with_path(root, out, df)

    assert result["r"].to_list() == ["young_low", "young_high", "adult"]
    # Rows 0 and 1 took the age then-branch → two path segments
    assert result["path"][0].count("|") == 1
    assert result["path"][1].count("|") == 1
    # Row 2 took the age otherwise-branch → one path segment, no nesting
    assert result["path"][2].count("|") == 0

    # Exact paths
    assert result["path"][0] == "age,0|score,0"    # age then → score then
    assert result["path"][1] == "age,0|score,1"    # age then → score otherwise
    assert result["path"][2] == "age,1"             # age otherwise


def test_path_cases_ranges_records_correct_bucket_index():
    """CasesRanges path index matches the 0-based position of the matched condition."""
    rule = CasesRule(root=CasesRanges(
        feature="score",
        conditions=[
            CasesBranch(when=RangeCondition(max=30.0), then=0),
            CasesBranch(when=RangeCondition(min=30.0, max=70.0), then=1),
            CasesBranch(when=RangeCondition(min=70.0), then=2),
        ],
        otherwise=3,
        branches=[
            LeafRule(result_idx=0),
            LeafRule(result_idx=1),
            LeafRule(result_idx=2),
            LeafRule(result_idx=-1),
        ],
        end_logic=RangeEndLogic.lower_inclusive,
        strict=False,
    ))
    out = _output("low", "mid", "high", default="none")
    df = pl.DataFrame({"score": [10.0, 50.0, 80.0]})
    result = _run_with_path(rule, out, df)

    assert result["r"].to_list() == ["low", "mid", "high"]
    assert result["path"][0] == "score,0"
    assert result["path"][1] == "score,1"
    assert result["path"][2] == "score,2"


def test_path_identical_inputs_always_produce_identical_paths():
    """Repeated identical rows always yield the same result and the same path."""
    rule = UnaryRule(
        condition=UnaryBetween(feature="v", min=10.0, max=20.0),
        then=LeafRule(result_idx=0),
        otherwise=LeafRule(result_idx=-1),
    )
    out = _output("match", default="no")
    df = pl.DataFrame({"v": [15.0, 15.0, 15.0]})
    result = _run_with_path(rule, out, df)

    paths = result["path"].to_list()
    assert len(set(paths)) == 1, f"Expected all paths identical, got {paths}"
    assert result["r"].to_list() == ["match", "match", "match"]


def test_path_null_input_takes_otherwise_path():
    """A null input value does not match numeric conditions and takes the otherwise path."""
    rule = UnaryRule(
        condition=UnaryLessThan(feature="v", threshold=50.0),
        then=LeafRule(result_idx=0),
        otherwise=LeafRule(result_idx=-1),
    )
    out = _output("low", default="other")
    df = pl.DataFrame({"v": pl.Series([30.0, None], dtype=pl.Float64)})
    result = _run_with_path(rule, out, df)

    assert result["r"].to_list() == ["low", "other"]
    assert result["path"][0] == "v,0"   # matched → then
    assert result["path"][1] == "v,1"   # null → otherwise
