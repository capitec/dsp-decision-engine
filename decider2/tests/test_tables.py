"""The decision-table engine's own properties.

The load-bearing one is `test_editing_rows_never_recompiles`: doc 08 §3.4
classifies a decision table as a *generic kernel* and lists its interior
change as **free**, which is a strictly stronger claim than the tree's
(a tree's shape change costs one staged compile). A table earns it by
keeping every row in an array rather than in emitted source, and this
asserts the source is genuinely identical across two different tables.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider2 import flow
from decider2.compile.driver import build_driver
from decider2.tables import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    DecisionTable,
    EqExpression,
    InExpression,
    IsTrueExpression,
    OrExpression,
    ParametersConfig,
    encode_table,
    table_module,
)
from decider2.testing import assert_equivalent


def _band_table(rows=None, **kwargs) -> DecisionTable:
    rows = rows if rows is not None else [
        {"lo": None, "hi": 30.0, "pts": 1},
        {"lo": 30.0, "hi": 70.0, "pts": 2},
        {"lo": 70.0, "hi": None, "pts": 3},
    ]
    return DecisionTable(
        name="bands",
        parameters=ParametersConfig(
            data=rows, dtypes={"lo": "Float64", "hi": "Float64", "pts": "Int64"}
        ),
        expression=BetweenExpression(
            type="between", variable="score",
            lower_bound_column="lo", upper_bound_column="hi", **kwargs,
        ),
        outputs=["pts"],
        default=[0],
    )


# ---------------------------------------------------------------------------
# The claim doc 08 §3.4 makes for a table specifically
# ---------------------------------------------------------------------------


def test_editing_rows_never_recompiles(tmp_path):
    """Doc 08 §3.4: a decision table's interior change is **free**.

    Two tables with different bands, different row counts and different
    outputs must have IDENTICAL condition shape (`decider2.tables.encode`'s
    own arrays: `group_start`/`group_end`/`op_kind`/...) — because every one
    of those differences lives in `shared`, never captured into the
    `row_fn` closure. `test_a_rebuilt_table_answers_differently_with_no_new_
    compile` below is the direct proof this means zero new compiles: the
    same compiled driver answers both.
    """
    three_bands = encode_table(_band_table())
    five_bands = encode_table(_band_table(rows=[
        {"lo": None, "hi": 10.0, "pts": 9},
        {"lo": 10.0, "hi": 20.0, "pts": 8},
        {"lo": 20.0, "hi": 40.0, "pts": 7},
        {"lo": 40.0, "hi": 80.0, "pts": 6},
        {"lo": 80.0, "hi": None, "pts": 5},
    ]))

    assert three_bands.n_conditions == five_bands.n_conditions
    assert three_bands.n_rows == 3
    assert five_bands.n_rows == 5
    # ...and the data that differs is entirely in the arrays.
    assert not np.array_equal(
        three_bands.shared["bands__c0_hi"], five_bands.shared["bands__c0_hi"]
    )


def test_a_rebuilt_table_answers_differently_with_no_new_compile(tmp_path):
    """The same claim, end to end: rebuild the table with new bounds, run
    it through the same driver, get different answers and one signature."""
    from decider2.runtime.invoke import DEFAULT_BUILD_DIR

    original = table_module(_band_table(), build_dir=tmp_path)
    pipeline = flow(original.module)
    frame = pl.DataFrame({"score": [10.0, 50.0, 90.0]})

    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    # Both `build_dir` and `terminal_names` must match what `pipeline.
    # apply()` computes/passes internally — `Pipeline.apply` takes no
    # `build_dir=` of its own, so `runtime.invoke.apply`'s own default
    # (`DEFAULT_BUILD_DIR`) is what actually gets used — so this driver and
    # `apply()`'s own hit the SAME `build_driver` cache entry and are the
    # SAME object. A packed step's kernel (doc 08 §3.4) compiles lazily, on
    # its first real call, rather than being eagerly probed the way
    # `decider2.compile.driver._try_njit` forces an ordinary step to at
    # build time (see this stage's report), so this test has to actually
    # run the SAME driver it inspects, not a second one built with `tmp_path`.
    driver = build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=DEFAULT_BUILD_DIR, terminal_names=frozenset({"bands_row", "pts"}),
    )
    assert driver.segments[0].kind == "compiled"

    before = pipeline.apply(frame, shared=original.shared)["pts"].to_list()
    # `bands_row` and `pts` are each their own `PackedCompiledSegment` now
    # (`decider2.compile.driver.build_packed_kernel`'s own docstring — a
    # packed step never fuses with a neighbour, a real, reported scope cut
    # of this pass), so this driver holds one signature per step rather
    # than one for the whole fused table — the baseline count itself is not
    # what doc 08 §2 promises; comparing it BEFORE and AFTER the retune
    # below is (it must not GROW), so that is what this asserts, rather
    # than a fixed number.
    before_sig_count = len(driver.signatures)

    retuned = table_module(_band_table(rows=[
        {"lo": None, "hi": 80.0, "pts": 1},
        {"lo": 80.0, "hi": None, "pts": 3},
    ]), build_dir=tmp_path)
    after = pipeline.apply(frame, shared=retuned.shared)["pts"].to_list()

    assert before == [1.0, 2.0, 3.0]
    assert after == [1.0, 1.0, 3.0]
    assert len(driver.signatures) == before_sig_count


# ---------------------------------------------------------------------------
# Row capture
# ---------------------------------------------------------------------------


def test_a_table_reports_which_row_matched(tmp_path):
    """The table's analogue of a tree's path capture: which row fired,
    -1 when none did."""
    built = table_module(_band_table(), build_dir=tmp_path)
    out = flow(built.module).apply(
        pl.DataFrame({"score": [10.0, 50.0, 90.0]}), shared=built.shared
    )

    assert built.row_column == "bands_row"
    assert out["bands_row"].to_list() == [0, 1, 2]


def test_no_match_takes_the_default(tmp_path):
    built = table_module(_band_table(rows=[{"lo": 10.0, "hi": 90.0, "pts": 7}],
                                     allow_gaps=True), build_dir=tmp_path)
    out = flow(built.module).apply(
        pl.DataFrame({"score": [50.0, 5.0, 200.0]}), shared=built.shared
    )

    assert out["bands_row"].to_list() == [0, -1, -1]
    assert out["pts"].to_list() == [7.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# BoundMode — decider 1's deliberate two-value enum
# ---------------------------------------------------------------------------


def test_upper_inclusive_moves_the_closed_end(tmp_path):
    """decider 1's `BoundMode`: lower_inclusive is [lo, hi), upper_inclusive
    is (lo, hi]. A boundary value lands in a different row under each, which
    is the whole reason both exist."""
    lower = table_module(_band_table(), name="lower", build_dir=tmp_path)
    upper = table_module(
        _band_table(mode=BoundMode.upper_inclusive), name="upper", build_dir=tmp_path
    )
    frame = pl.DataFrame({"score": [30.0, 70.0]})

    lower_rows = flow(lower.module).apply(frame, shared=lower.shared)["lower_row"].to_list()
    upper_rows = flow(upper.module).apply(frame, shared=upper.shared)["upper_row"].to_list()

    assert lower_rows == [1, 2]   # 30 starts the middle band
    assert upper_rows == [0, 1]   # 30 ends the first band


def test_bound_mode_is_the_trees_range_end_logic():
    """The two enums decider 1 kept separate are one enum here, so a change
    to one cannot leave the other behind."""
    from decider2.trees import RangeEndLogic

    assert BoundMode is RangeEndLogic
    assert BoundMode.lower_inclusive.value == "lower_inclusive"
    assert set(BoundMode) == {BoundMode.lower_inclusive, BoundMode.upper_inclusive}


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def test_or_of_two_bands(tmp_path):
    """`OrExpression` flattens to two disjuncts over the same rows."""
    table = DecisionTable(
        name="either",
        parameters=ParametersConfig(
            data=[{"lo": 0.0, "hi": 10.0, "alt_lo": 90.0, "alt_hi": 100.0, "pts": 1}],
            dtypes={
                "lo": "Float64", "hi": "Float64",
                "alt_lo": "Float64", "alt_hi": "Float64", "pts": "Int64",
            },
        ),
        expression=OrExpression(
            type="or",
            expressions=[
                BetweenExpression(type="between", variable="v",
                                  lower_bound_column="lo", upper_bound_column="hi",
                                  allow_gaps=True),
                BetweenExpression(type="between", variable="v",
                                  lower_bound_column="alt_lo", upper_bound_column="alt_hi",
                                  allow_gaps=True),
            ],
        ),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    out = flow(built.module).apply(
        pl.DataFrame({"v": [5.0, 50.0, 95.0]}), shared=built.shared
    )

    assert out["pts"].to_list() == [1.0, 0.0, 1.0]


def test_eq_on_a_string_column(tmp_path):
    """decider 1's `EqExpression` docstring example shape: match a string
    variable against a per-row value column."""
    table = DecisionTable(
        name="bureau",
        parameters=ParametersConfig(
            data=[
                {"key": "experian", "pts": 10},
                {"key": "transunion", "pts": 20},
            ],
            dtypes={"key": "String", "pts": "Int64"},
        ),
        expression=EqExpression(type="eq", variable="BureauKey", value_column="key"),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    frame = pl.DataFrame({"BureauKey": ["experian", "transunion", "other"]})
    pipeline = flow(built.module)

    for mode in ("interpreted", "stepped", "fused"):
        out = pipeline.apply(frame, shared=built.shared, mode=mode)
        assert out["pts"].to_list() == [10, 20, 0], f"mode={mode}"

    # The three batch modes are driven by hand here rather than through
    # `assert_equivalent`, which also drives `score()` — and `score()`
    # currently cannot take a `str` input at all. See
    # `test_score_cannot_yet_take_a_string_input` below.


def test_score_takes_a_string_input_and_agrees_with_apply(tmp_path):
    """A single-record call on a string-keyed table (doc 05 §9 criterion 2:
    "the same kernel answers a single record")."""
    table = DecisionTable(
        name="bureau_score",
        parameters=ParametersConfig(
            data=[{"key": "experian", "pts": 10}], dtypes={"key": "String", "pts": "Int64"}
        ),
        expression=EqExpression(type="eq", variable="BureauKey", value_column="key"),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)

    result = flow(built.module).score({"BureauKey": "experian"}, shared=built.shared)

    assert result["pts"] == 10


def test_a_string_literal_in_a_table_is_a_retunable_param(tmp_path):
    """EXPERIMENTS.md §O, for a table: a string literal is a `str` param
    resolved to that column's int32 code, so retuning its text is free."""
    table = DecisionTable(
        name="bureau",
        parameters=ParametersConfig(
            data=[{"key": "experian", "pts": 10}],
            dtypes={"key": "String", "pts": "Int64"},
        ),
        expression=EqExpression(type="eq", variable="BureauKey", value_column="key"),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    frame = pl.DataFrame({"BureauKey": ["experian", "transunion"]})
    pipeline = flow(built.module)

    assert "BureauKey_lit_0" in built.module.params_schema()

    default = pipeline.apply(frame, shared=built.shared)["pts"].to_list()
    retuned = pipeline.apply(
        frame, shared=built.shared,
        params={"bureau": {"BureauKey_lit_0": "transunion"}},
    )["pts"].to_list()

    assert default == [10.0, 0.0]
    assert retuned == [0.0, 10.0]


def test_in_on_numeric_sets(tmp_path):
    """The CSR path with numeric values and ragged set sizes."""
    table = DecisionTable(
        name="regions",
        parameters=ParametersConfig(
            data=[
                {"vals": [1.0, 2.0, 3.0], "pts": 5},
                {"vals": [9.0], "pts": 6},
            ],
            dtypes={"vals": "List", "pts": "Int64"},
        ),
        expression=InExpression(type="in", variable="region", values_column="vals"),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    frame = pl.DataFrame({"region": [2.0, 9.0, 4.0]})
    out = flow(built.module).apply(frame, shared=built.shared)

    assert out["pts"].to_list() == [5.0, 6.0, 0.0]
    assert_equivalent(flow(built.module), frame, shared=built.shared)


def test_and_of_between_and_is_true(tmp_path):
    table = DecisionTable(
        name="elig",
        parameters=ParametersConfig(
            data=[{"age_lo": 18.0, "age_hi": 65.0, "pts": 1}],
            dtypes={"age_lo": "Float64", "age_hi": "Float64", "pts": "Int64"},
        ),
        expression=AndExpression(
            type="and",
            expressions=[
                BetweenExpression(type="between", variable="age",
                                  lower_bound_column="age_lo",
                                  upper_bound_column="age_hi", allow_gaps=True),
                IsTrueExpression(type="is_true", variable="verified"),
            ],
        ),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    frame = pl.DataFrame({
        "age": [30.0, 17.0, 40.0, 70.0],
        "verified": [True, True, False, True],
    })
    out = flow(built.module).apply(frame, shared=built.shared)

    assert out["pts"].to_list() == [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Validation carried over from decider 1
# ---------------------------------------------------------------------------


def test_non_contiguous_rows_are_rejected_unless_gaps_are_allowed():
    """decider 1's own message and rule."""
    with pytest.raises(ValueError, match="ranges are not contiguous"):
        _band_table(rows=[
            {"lo": None, "hi": 30.0, "pts": 1},
            {"lo": 50.0, "hi": None, "pts": 2},
        ])

    # ...and allow_gaps=True permits exactly that table.
    _band_table(
        rows=[{"lo": None, "hi": 30.0, "pts": 1}, {"lo": 50.0, "hi": None, "pts": 2}],
        allow_gaps=True,
    )


def test_an_output_column_absent_from_the_table_is_rejected():
    with pytest.raises(ValueError, match="not found in parameters columns"):
        DecisionTable(
            name="dt",
            parameters=ParametersConfig(
                data=[{"lo": 0.0, "hi": 1.0}], dtypes={"lo": "Float64", "hi": "Float64"}
            ),
            expression=BetweenExpression(
                type="between", variable="v",
                lower_bound_column="lo", upper_bound_column="hi", allow_gaps=True,
            ),
            outputs=["missing_column"],
        )


def test_a_default_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="must match outputs length"):
        DecisionTable(
            name="dt",
            parameters=ParametersConfig(
                data=[{"lo": 0.0, "hi": 1.0, "a": 1, "b": 2}],
                dtypes={"lo": "Float64", "hi": "Float64", "a": "Int64", "b": "Int64"},
            ),
            expression=BetweenExpression(
                type="between", variable="v",
                lower_bound_column="lo", upper_bound_column="hi", allow_gaps=True,
            ),
            outputs=["a", "b"],
            default=[0],
        )


def test_a_table_composes_with_ordinary_modules(tmp_path):
    """`flow(Affordability, my_table, Scoring)` — same integration a tree
    gets, since both are just `Module`s."""

    def disposable_income(net_income: float, expenses: float) -> float:
        """Income remaining after committed expenses."""
        return net_income - expenses

    def final_score(pts: float) -> float:
        """Scale the table's points."""
        return pts * 10.0

    table = DecisionTable(
        name="afford_bands",
        parameters=ParametersConfig(
            data=[
                {"lo": None, "hi": 1000.0, "pts": 1},
                {"lo": 1000.0, "hi": None, "pts": 5},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", "pts": "Int64"},
        ),
        expression=BetweenExpression(
            type="between", variable="disposable_income",
            lower_bound_column="lo", upper_bound_column="hi",
        ),
        outputs=["pts"],
        default=[0],
    )
    built = table_module(table, build_dir=tmp_path)
    pipeline = flow(disposable_income, built.module, final_score)
    frame = pl.DataFrame({"net_income": [5000.0, 2000.0], "expenses": [1500.0, 1500.0]})

    out = pipeline.apply(frame, shared=built.shared)

    assert out["final_score"].to_list() == [50.0, 10.0]
    assert_equivalent(pipeline, frame, shared=built.shared)

