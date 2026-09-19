"""Scratch tests for decider2/params.py — the parameter declaration surface.

Not the acceptance test (tests/test_flagship.py is), but exercises the module
directly: doc 03 §4.4 (param()), §1 (missing_as/not_applicable_as, the four
null-policy tiers), §1.3 (docstring parsing), §2 (params/shared reserved
names).
"""
import pytest
from pydantic import BaseModel

from decider2.params import (
    MissingAs,
    NotApplicableAs,
    ParamSpec,
    build_params_model,
    harvest_signature,
    harvest_step,
    missing_as,
    not_applicable_as,
    param,
    parse_docstring,
)
from decider2.types import NullPolicy


# --- param() is a thin adapter over Field, and IS its default -------------


def test_param_returns_the_default_value():
    p = param(48.0, ge=6, le=60)
    assert p == 48.0
    assert isinstance(p, float)


def test_param_is_directly_usable_where_a_float_is_expected():
    def f(x: float = param(48.0, ge=6, le=60)) -> float:
        return x * 2

    assert f() == 96.0
    assert f(10.0) == 20.0


def test_param_carries_field_info_with_forwarded_kwargs():
    p = param(48.0, ge=6, le=60, description="Term cap in months")
    assert p.field_info.default == 48.0
    assert p.field_info.description == "Term cap in months"


def test_param_is_an_instance_of_paramspec():
    p = param(48.0, ge=6, le=60)
    assert isinstance(p, ParamSpec)
    assert not isinstance(p, MissingAs)
    assert not isinstance(p, NotApplicableAs)


@pytest.mark.parametrize("value", [1, "s", [1, 2], {"a": 1}, (1, 2)])
def test_param_supports_every_documented_carrier_type(value):
    p = param(value)
    assert p == value
    assert isinstance(p, type(value))
    assert isinstance(p, ParamSpec)


def test_param_rejects_bool_naming_the_alternative():
    with pytest.raises(TypeError, match="enable mask"):
        param(True)


def test_param_rejects_none():
    with pytest.raises(TypeError):
        param(None)


# --- missing_as() / not_applicable_as(): tiers 2 and 4 ---------------------


def test_missing_as_returns_the_fill_value_and_is_directly_callable():
    def f(bureau_score: float = missing_as(0.0)) -> float:
        return bureau_score * 0.01

    assert f() == 0.0
    assert f(500.0) == 5.0


def test_not_applicable_as_returns_the_fill_value():
    v = not_applicable_as(0.0)
    assert v == 0.0


def test_missing_as_and_not_applicable_as_are_distinguishable():
    m = missing_as(0.0)
    n = not_applicable_as(0.0)
    assert isinstance(m, MissingAs)
    assert not isinstance(m, NotApplicableAs)
    assert isinstance(n, NotApplicableAs)
    assert not isinstance(n, MissingAs)
    assert not isinstance(m, ParamSpec)
    assert not isinstance(n, ParamSpec)


def test_missing_as_rejects_bool_and_none():
    with pytest.raises(TypeError, match="bool \\| None"):
        missing_as(True)
    with pytest.raises(TypeError):
        missing_as(None)


def test_not_applicable_as_rejects_bool_and_none():
    with pytest.raises(TypeError, match="bool \\| None"):
        not_applicable_as(True)
    with pytest.raises(TypeError):
        not_applicable_as(None)


# --- signature harvesting: the four null-policy tiers ----------------------


def test_harvest_required_tier():
    def f(net_income: float, expenses: float) -> float:
        return net_income - expenses

    inputs, params, reads_params, reads_shared = harvest_signature(f)
    assert [i.name for i in inputs] == ["net_income", "expenses"]
    assert all(i.null_policy is NullPolicy.REQUIRED for i in inputs)
    assert params == ()
    assert reads_params is False
    assert reads_shared is False


def test_harvest_missing_as_tier():
    def f(bureau_score: float = missing_as(0.0)) -> float:
        return bureau_score

    inputs, params, _, _ = harvest_signature(f)
    assert len(inputs) == 1
    assert inputs[0].name == "bureau_score"
    assert inputs[0].null_policy is NullPolicy.MISSING_AS
    assert inputs[0].fill == 0.0
    assert inputs[0].annotation is float


def test_harvest_optional_tier_from_union_annotation():
    def f(bureau_score: float | None) -> float:
        return bureau_score or 0.0

    inputs, params, _, _ = harvest_signature(f)
    assert len(inputs) == 1
    assert inputs[0].null_policy is NullPolicy.OPTIONAL


def test_harvest_not_applicable_as_tier():
    def f(spouse_income: float = not_applicable_as(0.0)) -> float:
        return spouse_income

    inputs, params, _, _ = harvest_signature(f)
    assert inputs[0].null_policy is NullPolicy.NOT_APPLICABLE_AS
    assert inputs[0].fill == 0.0


def test_harvest_separates_params_from_inputs():
    def cap_by_income_band(
        term_cap: float,
        min_net_salary: float,
        cap: float = param(48.0, ge=6, le=60),
        income_threshold: float = param(5000.0, ge=0),
    ) -> float:
        return min(term_cap, cap) if min_net_salary < income_threshold else term_cap

    inputs, params, _, _ = harvest_signature(cap_by_income_band)
    assert [i.name for i in inputs] == ["term_cap", "min_net_salary"]
    assert {p.name for p in params} == {"cap", "income_threshold"}
    cap_decl = next(p for p in params if p.name == "cap")
    assert cap_decl.default == 48.0
    assert cap_decl.annotation is float
    assert cap_decl.field_info.default == 48.0


def test_harvest_bare_params_and_shared_are_flagged_not_inputs():
    def f(term_cap: float, params, shared) -> float:
        return term_cap

    inputs, params_decls, reads_params, reads_shared = harvest_signature(f)
    assert [i.name for i in inputs] == ["term_cap"]
    assert reads_params is True
    assert reads_shared is True


def test_harvest_rejects_var_args():
    def f(*args) -> float:
        return 0.0

    with pytest.raises(TypeError):
        harvest_signature(f)


# --- pydantic model generation ----------------------------------------------


def test_build_params_model_is_namespaced_and_matches_hand_written():
    def cap_by_income_band(
        term_cap: float,
        min_net_salary: float,
        cap: float = param(48.0, ge=6, le=60),
        income_threshold: float = param(5000.0, ge=0),
    ) -> float:
        return term_cap

    _, params, _, _ = harvest_signature(cap_by_income_band)
    Model = build_params_model("cap_by_income_band", params)

    assert Model.__name__ == "CapByIncomeBandParams"
    assert issubclass(Model, BaseModel)
    instance = Model()
    assert instance.cap == 48.0
    assert instance.income_threshold == 5000.0

    with pytest.raises(Exception):
        Model(cap=999.0)


def test_build_params_model_returns_none_when_no_params():
    def f(x: float) -> float:
        return x

    _, params, _, _ = harvest_signature(f)
    assert build_params_model("f", params) is None


def test_a_hand_written_model_and_a_harvested_one_agree_on_values():
    """Doc 03 §4.4: harvested and hand-written are "the same object ... the
    same validators". The comparator carries `extra="forbid"` because doc 03
    §10 requires it of a params model either way — "a misspelled param is a
    hard error, not silence" — so a hand-written model without it is not the
    thing §4.4 is comparing against.
    """
    from pydantic import ConfigDict, Field, create_model

    def cap_by_income_band(cap: float = param(48.0, ge=6, le=60)) -> float:
        return cap

    _, params, _, _ = harvest_signature(cap_by_income_band)
    Harvested = build_params_model("cap_by_income_band", params)
    HandWritten = create_model(
        "CapByIncomeBandParams",
        __config__=ConfigDict(extra="forbid"),
        cap=(float, Field(48.0, ge=6, le=60)),
    )

    assert Harvested().model_dump() == HandWritten().model_dump()
    assert Harvested.model_json_schema() == HandWritten.model_json_schema()


def test_a_misspelled_param_is_a_hard_error():
    """Doc 03 §10, verbatim: "A misspelled param is a **hard error**
    (`extra="forbid"`), not silence." """
    import pydantic

    def cap_by_income_band(cap: float = param(48.0, ge=6, le=60)) -> float:
        return cap

    _, params, _, _ = harvest_signature(cap_by_income_band)
    model = build_params_model("cap_by_income_band", params)
    with pytest.raises(pydantic.ValidationError):
        model(capp=36.0)


# --- docstring parsing: doc 03 §1.3 -----------------------------------------


def test_parse_docstring_with_implements_line():
    doc = """Cap term at 48 months below the income floor.

    Implements: Credit Policy §7.4.2
    """
    description, implements = parse_docstring(doc)
    assert description == "Cap term at 48 months below the income floor."
    assert implements == "Credit Policy §7.4.2"


def test_parse_docstring_without_implements_line():
    description, implements = parse_docstring("Affordability ratio.")
    assert description == "Affordability ratio."
    assert implements is None


def test_parse_docstring_none():
    assert parse_docstring(None) == (None, None)


# --- harvest_step ties it all together --------------------------------------


def test_harvest_step_end_to_end():
    def cap_by_income_band(
        term_cap: float,
        min_net_salary: float,
        cap: float = param(48.0, ge=6, le=60),
    ) -> float:
        """Cap term at 48 months below the income floor.

        Implements: Credit Policy §7.4.2
        """
        return min(term_cap, cap) if min_net_salary < 5000 else term_cap

    step = harvest_step(cap_by_income_band)
    assert step.name == "cap_by_income_band"
    assert step.fn is cap_by_income_band
    assert [i.name for i in step.inputs] == ["term_cap", "min_net_salary"]
    assert [p.name for p in step.params] == ["cap"]
    assert step.doc == "Cap term at 48 months below the income floor."
    assert step.implements == "Credit Policy §7.4.2"
    assert step.reads_params is False
    assert step.reads_shared is False


def test_harvest_step_name_override():
    def fn(x: float) -> float:
        return x

    step = harvest_step(fn, name="term_0042_cap_by_income_band")
    assert step.name == "term_0042_cap_by_income_band"
