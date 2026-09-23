import pickle
import sys

import pytest

from decider import param
from decider.engine.ir.decls import ParamDecl
from decider.engine.params import (
    NodeParams, ParamsCache, ParamsError, Status, bundle_class, document_key, harvest,
    record_shared_type, validate_node,
)


def node_for(path, fn):
    return NodeParams(path, harvest(fn)[1])


def cap_by_income(term_cap: float, cap: float = param(48.0, ge=6, le=60),
                  base_rate: float = param(5.0, shared_key="base_rate", ge=0)) -> float:
    return min(term_cap, cap)


def net_of_floor(term_cap: float, floor: float = param(10.0)) -> float:
    return term_cap - floor


def plus_headroom(net_of_floor: float, headroom: float = param(3.0)) -> float:
    return net_of_floor + headroom


# --- nested documents: each node reads its own entry --------------------------


def test_a_node_reads_its_params_from_its_nested_path():
    node = node_for("term/cap_by_income", cap_by_income)
    doc = {"shared": {"base_rate": 7.0}, "term": {"cap_by_income": {"cap": 36.0}}}
    result = validate_node(node, doc)
    assert result.status is Status.OK
    assert result.bundle == (36.0, 7.0)
    assert result.bundle._fields == ("cap", "base_rate")


def test_missing_keys_fall_back_to_the_nodes_own_defaults():
    result = validate_node(node_for("term/cap_by_income", cap_by_income), {})
    assert result.status is Status.OK
    assert result.bundle == (48.0, 5.0)


def test_each_node_gets_only_its_own_params_not_its_siblings():
    doc = {"band": {"net_of_floor": {"floor": 20.0}, "plus_headroom": {"headroom": 5.0}}}
    floor = validate_node(node_for("band/net_of_floor", net_of_floor), doc).bundle
    headroom = validate_node(node_for("band/plus_headroom", plus_headroom), doc).bundle
    assert floor._asdict() == {"floor": 20.0}
    assert headroom._asdict() == {"headroom": 5.0}


def test_a_misspelled_param_is_invalid_with_a_suggestion():
    result = validate_node(node_for("cap", cap_by_income), {"cap": {"capp": 12.0}})
    assert result.status is Status.INVALID
    assert result.errors == ("cap: unknown param 'capp'; did you mean 'cap'?",)


def test_a_shared_param_set_under_the_nodes_own_entry_is_invalid():
    result = validate_node(node_for("cap", cap_by_income), {"cap": {"base_rate": 1.0}})
    assert result.status is Status.INVALID


def test_a_non_mapping_node_entry_is_invalid():
    result = validate_node(node_for("term/cap", cap_by_income), {"term": 3})
    assert result.status is Status.INVALID


# --- shared params --------------------------------------------------------------


def test_two_nodes_may_put_different_defaults_and_bounds_on_one_shared_key():
    def fee(amount: float, base_rate: float = param(9.0, shared_key="base_rate", le=8)) -> float:
        return amount * base_rate

    cap, fee_node = node_for("term/cap", cap_by_income), node_for("pricing/fee", fee)
    assert validate_node(cap, {}).bundle.base_rate == 5.0
    assert validate_node(fee_node, {"pricing": {}, "shared": {"base_rate": 7.5}}).bundle.base_rate == 7.5
    too_high = {"shared": {"base_rate": 8.5}}
    assert validate_node(cap, too_high).status is Status.OK
    assert validate_node(fee_node, too_high).status is Status.INVALID


def test_a_shared_type_conflict_names_both_nodes():
    types = {}
    record_shared_type(types, ParamDecl("base_rate", float, 5.0, shared_key="base_rate"), "term/cap_by_income")
    record_shared_type(types, ParamDecl("rate", float, 1.0, shared_key="base_rate"), "other/node")
    with pytest.raises(TypeError) as exc:
        record_shared_type(types, ParamDecl("base_rate", int, 5, shared_key="base_rate"), "pricing/fee")
    assert str(exc.value) == (
        "shared param 'base_rate' is declared float by term/cap_by_income and int by pricing/fee"
    )


def test_an_invalid_shared_value_names_the_shared_key():
    result = validate_node(node_for("term/cap", cap_by_income), {"shared": {"base_rate": -1.0}})
    assert result.errors[0].startswith("term/cap: shared param 'base_rate':")


# --- on_invalid -----------------------------------------------------------------


def _policy_node(policy):
    def f(x: float, cap: float = param(48.0, le=60, on_invalid=policy)) -> float:
        return x

    return node_for("term/f", f)


BAD = {"term": {"f": {"cap": 99.0}}}


def test_on_invalid_is_recorded_on_the_param_decl():
    assert [d.on_invalid for d in _policy_node("warn").decls] == ["warn"]


def test_on_invalid_error_raises_naming_the_node_path_and_param():
    result = validate_node(_policy_node("error"), BAD)
    assert result.status is Status.INVALID and result.bundle is None
    with pytest.raises(ParamsError, match=r"\(affects 10 rows\).*\nterm/f: param 'cap'"):
        result.check(rows=10)


def test_on_invalid_warn_records_a_warning_and_uses_the_default():
    result = validate_node(_policy_node("warn"), BAD)
    assert result.status is Status.OK
    assert result.bundle.cap == 48.0
    assert result.warnings == (
        "term/f: param 'cap': Input should be less than or equal to 60 (got 99.0); using the default",
    )


def test_on_invalid_default_uses_the_default_silently():
    result = validate_node(_policy_node("default"), BAD)
    assert (result.status, result.bundle.cap, result.warnings) == (Status.OK, 48.0, ())


def test_a_valid_value_is_kept_whatever_the_policy():
    assert validate_node(_policy_node("default"), {"term": {"f": {"cap": 50.0}}}).bundle.cap == 50.0


# --- required params and None/bool defaults ------------------------------------


def test_a_missing_required_param_is_invalid():
    def f(x: float, floor: float = param(required=True, on_invalid="default")) -> float:
        return x

    node = node_for("f", f)
    missing = validate_node(node, {})
    assert missing.status is Status.INVALID
    assert missing.errors == ("f: param 'floor' is required but missing",)
    assert validate_node(node, {"f": {"floor": 3.0}}).bundle.floor == 3.0


def test_none_and_bool_defaults_validate_and_can_be_overridden():
    def f(x: float, on: bool = param(True), floor: float | None = param(None)) -> float:
        return x

    node = node_for("f", f)
    assert validate_node(node, {}).bundle == (True, None)
    assert validate_node(node, {"f": {"on": False, "floor": 2.0}}).bundle == (False, 2.0)


# --- bundles --------------------------------------------------------------------


def test_the_defaults_bundle_has_the_same_type_as_a_validated_bundle():
    def f(x: float, cap: float = param(48, ge=6), n: int = param(3),
          k: float = param(required=True, ge=1)) -> float:
        return x

    node = node_for("f", f)
    validated = validate_node(node, {"f": {"cap": 50, "k": 2}}).bundle
    assert type(node.defaults) is type(validated)
    assert [type(v) for v in node.defaults] == [type(v) for v in validated] == [float, int, float]


def test_one_field_set_is_one_bundle_class_picklable_by_reference():
    cls = bundle_class(("cap", "base_rate"))
    assert bundle_class(("cap", "base_rate")) is cls
    assert pickle.loads(pickle.dumps(cls)) is cls
    assert pickle.loads(pickle.dumps(cls(1.0, 2.0))) == (1.0, 2.0)


def test_a_bundle_class_named_by_another_process_is_rebuilt_on_demand():
    import decider.engine.params.bundles as bundles

    cls = getattr(bundles, "Bundle_7_neverco_3_abc")
    assert cls._fields == ("neverco", "abc")
    with pytest.raises(AttributeError):
        getattr(bundles, "not_a_bundle")


# --- the validation cache -------------------------------------------------------


def test_an_unseen_document_is_unknown_then_cached():
    cache, node = ParamsCache(), node_for("term/cap", cap_by_income)
    doc = {"term": {"cap": {"cap": 36.0}}}
    key = document_key(doc)
    assert cache.status(key, "term/cap") is Status.UNKNOWN
    first = cache.validate(key, doc, node)
    assert cache.status(key, "term/cap") is Status.OK
    assert cache.validate(key, doc, node) is first


def test_an_equal_document_is_never_revalidated(monkeypatch):
    import decider.engine.params.validate as v

    cache, node = ParamsCache(), node_for("term/cap", cap_by_income)
    doc = {"term": {"cap": {"cap": 36.0}}}
    cache.validate(document_key(doc), doc, node)
    monkeypatch.setattr(v, "validate_node", lambda *a: pytest.fail("revalidated"))
    copy = {"term": {"cap": {"cap": 36.0}}}
    assert cache.validate(document_key(copy), copy, node).bundle.cap == 36.0


def test_an_invalid_result_is_cached_as_invalid():
    cache, node = ParamsCache(), _policy_node("error")
    key = document_key(BAD)
    cache.validate(key, BAD, node)
    assert cache.status(key, "term/f") is Status.INVALID


def test_an_unknown_param_with_no_close_match_lists_the_nodes_params():
    result = validate_node(node_for("cap", cap_by_income), {"cap": {"ceiling": 12.0}})
    assert result.errors == ("cap: unknown param 'ceiling'; its params are ['cap']",)


def test_a_param_given_by_the_argument_it_feeds_points_at_its_document_key():
    node = NodeParams("rule", (ParamDecl("min_ratio", float, 2.0, arg="threshold"),))
    assert validate_node(node, {"rule": {"threshold": 10.0}}).errors == (
        "rule: unknown param 'threshold'; did you mean 'min_ratio' (it feeds argument 'threshold')?",
    )


def test_the_params_error_header_says_how_many_rows_it_affects():
    invalid = validate_node(node_for("cap", cap_by_income), {"cap": {"cap": 100.0}})
    with pytest.raises(ParamsError, match=r"^invalid params \(affects 1 row\):"):
        invalid.check(rows=1)
    with pytest.raises(ParamsError, match=r"^invalid params \(affects 3 rows\):"):
        invalid.check(rows=3)
