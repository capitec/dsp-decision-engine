"""The acceptance test for the vertical slice.

This is doc 03 §1.1's flagship example, verbatim. If it passes in all three
modes and they agree exactly, the core design claim holds: one rule costs one
artefact, and the three-mode ladder is real.
"""
import polars as pl
import pytest

from decider2 import flow, module, param


# --- doc 03 §1.1, verbatim -------------------------------------------------

def disposable_income(net_income: float, expenses: float) -> float:
    """Income remaining after committed expenses."""
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    """Affordability ratio."""
    return disposable_income / instalment


def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60),
    income_threshold: float = param(5000.0, ge=0),
) -> float:
    """Cap term at 48 months below the income floor.

    Implements: Credit Policy §7.4.2
    """
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


# `|` needs a Module on the left (a plain function has no __or__), so a
# pipeline of only bare functions is built with flow(). Doc 03 §5.3.
pipeline = flow(disposable_income, affordability_ratio, cap_by_income_band)


FRAME = pl.DataFrame({
    "net_income":     [9200.0, 4100.0, 15000.0, 4999.0],
    "expenses":       [3100.0, 1500.0,  6000.0, 2000.0],
    "instalment":     [1200.0,  800.0,  2500.0,  700.0],
    "term_cap":       [  60.0,   60.0,    60.0,   60.0],
    "min_net_salary": [9200.0, 4100.0, 15000.0, 4999.0],
})


# --- 1. a plain function is still a plain function -------------------------

def test_step_is_directly_callable_with_its_declared_default():
    """param() returns a real value: no decorator, no pipeline, no import order."""
    assert cap_by_income_band(term_cap=60.0, min_net_salary=4000.0) == 48.0


def test_step_default_is_overridable_in_a_direct_call():
    assert cap_by_income_band(term_cap=60.0, min_net_salary=4000.0, cap=36.0) == 36.0


def test_step_above_the_floor_is_untouched():
    assert cap_by_income_band(term_cap=60.0, min_net_salary=9200.0) == 60.0


# --- 2. one rule costs one artefact ---------------------------------------

def test_bare_function_is_a_pipeline_element():
    assert "cap_by_income_band" in pipeline.interface.outputs


def test_params_are_namespaced_by_module_instance():
    schema = pipeline.params_schema()
    assert schema["cap_by_income_band"]["cap"] == 48.0
    assert schema["cap_by_income_band"]["income_threshold"] == 5000.0


def test_the_docstring_is_the_description():
    step = pipeline.step("cap_by_income_band")
    assert step.doc.startswith("Cap term at 48 months")
    assert step.implements == "Credit Policy §7.4.2"


# --- 3. the equivalence ladder --------------------------------------------

@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_every_mode_produces_the_same_answer(mode):
    out = pipeline.apply(FRAME, mode=mode)
    assert out["cap_by_income_band"].to_list() == [60.0, 48.0, 60.0, 48.0]


def test_the_three_modes_agree_exactly():
    frames = [pipeline.apply(FRAME, mode=m)
              for m in ("interpreted", "stepped", "fused")]
    assert frames[0].equals(frames[1])
    assert frames[1].equals(frames[2])


# --- 4. the output frame is additive --------------------------------------

def test_output_is_additive():
    out = pipeline.apply(FRAME)
    for col in FRAME.columns:
        assert col in out.columns
    assert "cap_by_income_band" in out.columns


def test_untapped_intermediates_are_not_materialised():
    """disposable_income feeds affordability_ratio and nothing else asks for it."""
    out = pipeline.apply(FRAME)
    assert "disposable_income" not in out.columns


def test_emit_materialises_an_intermediate():
    out = pipeline.emit("disposable_income").apply(FRAME)
    assert out["disposable_income"].to_list() == [6100.0, 2600.0, 9000.0, 2999.0]


def test_drop_removes_a_declared_column():
    out = pipeline.drop("min_net_salary").apply(FRAME)
    assert "min_net_salary" not in out.columns


# --- 5. retuning is a value change ----------------------------------------

def test_retuning_changes_the_answer_without_editing_code():
    out = pipeline.apply(FRAME, params={"cap_by_income_band": {"cap": 36.0}})
    assert out["cap_by_income_band"].to_list() == [60.0, 36.0, 60.0, 36.0]


def test_a_param_outside_its_bounds_is_rejected_by_pydantic():
    with pytest.raises(Exception):
        pipeline.apply(FRAME, params={"cap_by_income_band": {"cap": 999.0}})


# --- 6. the single-record path --------------------------------------------

def test_score_takes_a_dict_and_returns_a_dict():
    out = pipeline.score({
        "net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0,
        "term_cap": 60.0, "min_net_salary": 4100.0,
    })
    assert out["cap_by_income_band"] == 48.0


def test_score_agrees_with_apply_row_for_row():
    batch = pipeline.apply(FRAME)["cap_by_income_band"].to_list()
    single = [
        pipeline.score(dict(zip(FRAME.columns, row)))["cap_by_income_band"]
        for row in FRAME.iter_rows()
    ]
    assert batch == single
