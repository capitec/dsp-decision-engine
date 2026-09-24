import pydantic
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

from decider import missing_as, param
from decider.engine.ir.decls import FeatureKind, NullPolicy, Output, feature_kind
from decider.engine.params import MissingAs, NodeParams, ParamSpec, call_with_defaults, harvest


# --- param(): the default itself, plus a declaration -------------------------


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


def test_param_is_an_instance_of_paramspec_not_missing_as():
    p = param(48.0)
    assert isinstance(p, ParamSpec)
    assert not isinstance(p, MissingAs)


@pytest.mark.parametrize("value", [1, "s", [1, 2], {"a": 1}, (1, 2)])
def test_param_carries_every_subclassable_default_type(value):
    p = param(value)
    assert p == value
    assert isinstance(p, type(value))
    assert isinstance(p, ParamSpec)


@pytest.mark.parametrize("value", [True, False, None])
def test_bool_and_none_defaults_are_accepted_as_plain_markers(value):
    p = param(value)
    assert type(p) is ParamSpec
    assert p.default is value


def test_param_needs_a_default_or_required_but_not_both():
    with pytest.raises(TypeError):
        param()
    with pytest.raises(TypeError):
        param(1.0, required=True)


def test_param_rejects_an_unknown_on_invalid_policy():
    with pytest.raises(ValueError, match="on_invalid"):
        param(1.0, on_invalid="ignore")


# --- missing_as() -------------------------------------------------------------


def test_missing_as_returns_the_fill_value_and_is_directly_callable():
    def f(bureau_score: float = missing_as(0.0)) -> float:
        return bureau_score * 0.01

    assert f() == 0.0
    assert f(500.0) == 5.0


def test_missing_as_and_param_are_distinguishable():
    m = missing_as(0.0)
    assert isinstance(m, MissingAs)
    assert not isinstance(m, ParamSpec)


def test_missing_as_rejects_none_naming_the_optional_spelling():
    with pytest.raises(TypeError, match="T \\| None"):
        missing_as(None)


def test_missing_as_accepts_bool_as_a_plain_marker():
    m = missing_as(False)
    assert type(m) is MissingAs
    assert m.fill is False


# --- harvest(): inputs, params and outputs from the signature ----------------


def test_harvest_required_inputs():
    def f(net_income: float, expenses: float) -> float:
        return net_income - expenses

    inputs, params, outputs = harvest(f)
    assert [i.name for i in inputs] == ["net_income", "expenses"]
    assert all(i.null_policy is NullPolicy.REQUIRED for i in inputs)
    assert params == ()
    assert outputs == (Output("f", float),)


def test_harvest_missing_as_input():
    def f(bureau_score: float = missing_as(0.0)) -> float:
        return bureau_score

    (inp,), _, _ = harvest(f)
    assert inp.name == "bureau_score"
    assert inp.null_policy is NullPolicy.MISSING_AS
    assert inp.fill == 0.0
    assert inp.annotation is float


def test_harvest_optional_input_from_union_annotation():
    def f(bureau_score: float | None) -> float:
        return bureau_score or 0.0

    (inp,), _, _ = harvest(f)
    assert inp.null_policy is NullPolicy.OPTIONAL


def test_harvest_separates_params_from_inputs():
    def cap_by_income_band(
        term_cap: float,
        min_net_salary: float,
        cap: float = param(48.0, ge=6, le=60),
        base_rate: float = param(5.0, shared_key="base_rate", on_invalid="warn"),
        floor: float = param(required=True),
    ) -> float:
        return min(term_cap, cap) if min_net_salary < 5000 else term_cap

    inputs, params, _ = harvest(cap_by_income_band)
    assert [i.name for i in inputs] == ["term_cap", "min_net_salary"]
    cap, base_rate, floor = params
    assert (cap.name, cap.annotation, cap.default, cap.shared_key) == ("cap", float, 48.0, None)
    assert cap.field_info.default == 48.0
    assert type(cap.default) is float
    assert (base_rate.shared_key, base_rate.on_invalid) == ("base_rate", "warn")
    assert floor.required


def test_harvest_takes_an_unannotated_params_type_from_its_default():
    def f(x: float, k=param(3)) -> float:
        return x * k

    _, (k,), _ = harvest(f)
    assert k.annotation is int


def test_params_and_shared_are_ordinary_input_names():
    def f(params: float, shared: float) -> float:
        return params + shared

    inputs, _, _ = harvest(f)
    assert [i.name for i in inputs] == ["params", "shared"]


def test_harvest_rejects_var_args():
    def f(*args) -> float:
        return 0.0

    with pytest.raises(TypeError, match="args"):
        harvest(f)


def test_harvest_rejects_a_bare_default_naming_both_spellings():
    def f(x: float = 1.0) -> float:
        return x

    with pytest.raises(TypeError, match=r"param\(1.0\).*missing_as\(1.0\)"):
        harvest(f)


def test_harvest_multiple_outputs_from_a_tuple_annotation():
    def banding(ratio: float) -> tuple[int, float]:
        return (1, 10.0) if ratio > 2 else (0, 0.0)

    _, _, outputs = harvest(banding, outputs=("band", "band_score"))
    assert outputs == (Output("band", int), Output("band_score", float))


def test_harvest_multiple_outputs_with_a_mismatched_annotation_raises():
    def banding(ratio: float) -> tuple[int]:
        return (1,)

    with pytest.raises(TypeError, match="tuple"):
        harvest(banding, outputs=("band", "band_score"))


# --- a bare function stays callable with its defaults -------------------------


def test_call_with_defaults_replaces_bool_and_none_markers():
    def gate(x: float, on: bool = param(True), floor: float | None = param(None)) -> float:
        return (x if on else 0.0) if floor is None else floor

    assert call_with_defaults(gate, 2.0) == 2.0
    assert call_with_defaults(gate, 2.0, on=False) == 0.0
    assert call_with_defaults(gate, 2.0, False, 7.0) == 7.0


def test_call_with_defaults_replaces_a_bool_missing_as_marker():
    def f(flag: bool = missing_as(False)) -> bool:
        return flag

    assert call_with_defaults(f) is False


def test_call_with_defaults_names_a_missing_required_param():
    def f(x: float, k: float = param(required=True)) -> float:
        return x * k

    assert call_with_defaults(f, 2.0, k=3.0) == 6.0
    with pytest.raises(TypeError, match="required param 'k'"):
        call_with_defaults(f, 2.0)


# --- the generated per-node model ----------------------------------------------


def _cap_params():
    def cap_by_income_band(
        term_cap: float,
        cap: float = param(48.0, ge=6, le=60),
        income_threshold: float = param(5000.0, ge=0),
    ) -> float:
        return term_cap

    return harvest(cap_by_income_band)[1]


def test_node_model_is_named_by_path_and_enforces_bounds():
    model = NodeParams("term/cap_by_income_band", _cap_params()).model
    assert model.__name__ == "term/cap_by_income_band"
    assert issubclass(model, BaseModel)
    instance = model()
    assert (instance.cap, instance.income_threshold) == (48.0, 5000.0)
    with pytest.raises(pydantic.ValidationError):
        model(cap=999.0)


def test_a_node_with_no_params_gets_an_empty_bundle():
    node = NodeParams("f", ())
    assert node.defaults == ()


def test_a_hand_written_model_and_a_harvested_one_agree():
    def cap_by_income_band(cap: float = param(48.0, ge=6, le=60)) -> float:
        return cap

    harvested = NodeParams("cap_by_income_band", harvest(cap_by_income_band)[1]).model
    hand_written = create_model(
        "cap_by_income_band",
        __config__=ConfigDict(extra="forbid"),
        cap=(float, Field(48.0, ge=6, le=60)),
    )
    assert harvested().model_dump() == hand_written().model_dump()
    assert harvested.model_json_schema() == hand_written.model_json_schema()


def test_a_misspelled_param_is_a_hard_error_in_the_model():
    model = NodeParams("cap", _cap_params()).model
    with pytest.raises(pydantic.ValidationError):
        model(capp=36.0)


# --- feature kinds keep their fixed values ---------------------------------------


def test_feature_kind_values_are_fixed():
    assert [int(k) for k in FeatureKind] == [0, 1, 2, 3, 4]
    assert feature_kind(bytes) is FeatureKind.STR
    assert feature_kind(bool) is FeatureKind.BOOL
    assert feature_kind(object) is FeatureKind.F64


def test_bool_and_none_markers_test_like_their_value_on_a_direct_call():
    def excluded(deceased: bool = missing_as(False), review: bool = param(False),
                 on: bool = param(True), note: str | None = param(None)) -> tuple:
        return bool(deceased), bool(review), bool(on), bool(note)

    assert excluded() == (False, False, True, False)
    assert missing_as(True) and not param(0.0)
