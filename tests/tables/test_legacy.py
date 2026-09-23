"""decider_old's five decision-table tests, same configs, same answers, in every mode."""
import polars as pl

from decider.steps.tables import DecisionTableConfig


def _table(parameters: dict, expression: dict, outputs: list, default: list) -> DecisionTableConfig:
    # decider_old's `DecisionTableModule` document, loaded as it was written.
    return DecisionTableConfig.load({"type": "decision_table", "name": "dt", "parameters": parameters,
                                     "expression": expression, "outputs": outputs, "default": default})


def _between(rows, dtypes, variable, lo, hi, outputs, default, **kwargs):
    return _table({"data": rows, "dtypes": dtypes},
                  {"type": "between", "variable": variable, "lower_bound_column": lo, "upper_bound_column": hi,
                   **kwargs}, outputs, default)


def test_between_maps_ranges_to_output_labels(run):
    m = _between(
        rows=[{"lo": None, "hi": 30.0, "band": "low"}, {"lo": 30.0, "hi": 70.0, "band": "mid"},
              {"lo": 70.0, "hi": None, "band": "high"}],
        dtypes={"lo": "Float64", "hi": "Float64", "band": "String"},
        variable="score", lo="lo", hi="hi", outputs=["band"], default=["other"],
    )
    # lower_inclusive by default: [lo, hi)
    assert run(m, pl.DataFrame({"score": [10.0, 30.0, 70.0, 90.0]}))["band"].to_list() == ["low", "mid", "high", "high"]


def test_between_default_when_outside_all_ranges(run):
    m = _between(
        rows=[{"lo": 10.0, "hi": 90.0, "label": "in_range"}],
        dtypes={"lo": "Float64", "hi": "Float64", "label": "String"},
        variable="v", lo="lo", hi="hi", allow_gaps=True, outputs=["label"], default=["out_of_range"],
    )
    out = run(m, pl.DataFrame({"v": [50.0, 5.0, 200.0]}))
    assert out["label"].to_list() == ["in_range", "out_of_range", "out_of_range"]


def test_in_expression_categorical_lookup(run):
    m = _table(
        {"data": [{"vals": ["A", "B"], "tier": "premium"}, {"vals": ["C", "D"], "tier": "standard"}],
         "dtypes": [("vals", {"type": "List", "inner": "String"}), ("tier", "String")]},
        {"type": "in", "variable": "code", "values_column": "vals"}, ["tier"], ["unknown"],
    )
    assert run(m, pl.DataFrame({"code": ["A", "C", "X"]}))["tier"].to_list() == ["premium", "standard", "unknown"]


def test_and_expression_requires_all_conditions(run):
    m = _table(
        {"data": [{"age_lo": 18.0, "age_hi": 65.0, "flag": True, "outcome": "eligible"}],
         "dtypes": {"age_lo": "Float64", "age_hi": "Float64", "flag": "Boolean", "outcome": "String"}},
        {"type": "and", "expressions": [
            {"type": "between", "variable": "age", "lower_bound_column": "age_lo", "upper_bound_column": "age_hi",
             "allow_gaps": True},
            {"type": "is_true", "variable": "verified"},
        ]},
        ["outcome"], ["ineligible"],
    )
    df = pl.DataFrame({"age": [30.0, 17.0, 40.0, 70.0], "verified": [True, True, False, True]})
    assert run(m, df)["outcome"].to_list() == ["eligible", "ineligible", "ineligible", "ineligible"]


def test_multiple_output_columns_all_populated(run):
    m = _between(
        rows=[{"lo": None, "hi": 50.0, "label": "low", "pts": 10},
              {"lo": 50.0, "hi": None, "label": "high", "pts": 20}],
        dtypes={"lo": "Float64", "hi": "Float64", "label": "String", "pts": "Int64"},
        variable="v", lo="lo", hi="hi", outputs=["label", "pts"], default=["other", 0],
    )
    out = run(m, pl.DataFrame({"v": [20.0, 80.0, 200.0]}))
    assert out["label"].to_list() == ["low", "high", "high"]
    assert out["pts"].to_list() == [10, 20, 20]


def test_unnest_output_is_accepted_and_outputs_are_always_flat_columns(run):
    m = DecisionTableConfig.load({
        "type": "decision_table", "name": "dt", "unnest_output": True,
        "parameters": {"data": [{"lo": 0.0, "hi": 1.0, "src": "a"}]},
        "expression": {"type": "between", "variable": "v", "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["src"],
    })
    assert run(m, pl.DataFrame({"v": [0.5, 3.0]}))["src"].to_list() == ["a", None]
