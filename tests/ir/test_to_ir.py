from __future__ import annotations

import gc

import pytest

from decider import branch, dag, engine, flow, param, step
from decider.engine.ir.context import _BUILT, IRContext
from decider.engine.ir.decls import Output, ParamDecl
from decider.engine.ir.nodes import CallNode, SequenceNode, iter_nodes
from decider.serializable.dataframe import DataFrame
from decider.steps import ConfigurableStep, ParamRef, Step, TableRef, TableValue


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float, min_ratio: float = param(0.3, shared_key="min_ratio", on_invalid="warn")) -> bool:
    return ratio >= min_ratio


@step(output="term_cap")
def cap_by_income(term_cap: float, base_rate: float = param(5.0, shared_key="base_rate")) -> float:
    return term_cap


@step(output="term_cap")
def cap_again(term_cap: float, base_rate: float = param(6.0, shared_key="base_rate")) -> float:
    return term_cap


def fee(amount: float, base_rate: int = param(5, shared_key="base_rate")) -> float:
    return amount


class PriceTable(ConfigurableStep):
    rows: TableValue

    def to_ir(self, ctx):
        rows = ctx.table(self.rows, {"product": "str", "rate": "float"})
        params = (rows,) if isinstance(rows, ParamDecl) else ()
        return CallNode(ctx.origin(self), "row", lambda row, params: (0.0,), (), (Output("rate", float),), params)


# --- origins and paths ---


def test_every_node_has_a_unique_path_and_an_import_path_source():
    p = flow(dag(ratio, affordable, name="afford"), flow(cap_by_income, name="term"))
    nodes = list(iter_nodes(engine.to_ir(p)))
    paths = [n.origin.path for n in nodes]
    assert len(paths) == len(set(paths))
    assert all(":" in n.origin.source for n in nodes)
    assert nodes[2].origin.source == f"{__name__}:ratio"


def test_the_same_step_placed_twice_gets_two_paths():
    s = step(ratio)
    node = engine.to_ir(flow(flow(s, name="a"), flow(s, name="b")))
    a, b = (n.children()[0] for n in node.children())
    assert (a.origin.path, b.origin.path) == ("a/ratio", "b/ratio")
    assert a is not b


def test_a_path_clash_raises_suggesting_a_name():
    with pytest.raises(ValueError, match=r"two nodes have path 'ratio'.*\.named\('ratio_2'\)"):
        engine.to_ir(flow(ratio, ratio))
    engine.to_ir(flow(ratio, step(ratio).named("ratio_2")))


@pytest.mark.parametrize("where", ["top", "nested"])
def test_a_step_named_shared_is_rejected(where):
    s = step(ratio, name="shared")
    with pytest.raises(ValueError, match="'shared'"):
        engine.to_ir(s if where == "top" else flow(s, name="term"))


def test_a_node_without_an_origin_is_rejected():
    class Bad(Step):
        name = "bad"

        def to_ir(self, ctx):
            return SequenceNode(None, ())

    with pytest.raises(TypeError, match="origin"):
        engine.to_ir(Bad())


def test_to_ir_accepts_a_plain_function():
    node = engine.to_ir(ratio)
    assert isinstance(node, CallNode) and node.origin.path == "ratio"


# --- shared params ---


def test_a_shared_param_type_conflict_names_both_nodes():
    p = flow(flow(cap_by_income, name="term"), flow(fee, name="pricing"))
    with pytest.raises(TypeError) as err:
        engine.to_ir(p)
    assert str(err.value) == "shared param 'base_rate' is declared float by term/cap_by_income and int by pricing/fee"


def test_shared_params_may_differ_in_default():
    schema = flow(cap_by_income, cap_again).parameters()
    assert schema["shared"]["base_rate"]["used_by"] == ["cap_by_income", "cap_again"]
    assert schema["shared"]["base_rate"]["default"] == 5.0


def test_on_invalid_is_recorded_on_the_param_decl():
    assert engine.to_ir(affordable).params[0].on_invalid == "warn"


# --- config values ---


def test_value_turns_a_literal_or_ref_into_a_literal_or_decl():
    ctx = IRContext()
    assert ctx.value(0.7) == 0.7
    assert ctx.value(ParamRef(param="hi", default=0.7)) == ParamDecl("hi", float, 0.7)
    shared = ctx.value(ParamRef(param="base_rate", shared=True), float)
    assert shared == ParamDecl("base_rate", float, None, required=True, shared_key="base_rate")


def test_table_turns_inline_rows_or_a_ref_into_rows_or_decl():
    ctx = IRContext()
    rows = DataFrame(data=[{"product": "a", "rate": 1.0}])
    assert ctx.table(rows, {"product": "str"}) is rows
    decl = ctx.table(TableRef(table="prices"), {"product": "str", "rate": "float"})
    assert (decl.name, decl.required, decl.schema) == ("prices", True, {"product": "str", "rate": "float"})


def test_a_table_param_reports_its_schema():
    cfg = PriceTable.model_validate({"name": "price_by_product", "rows": {"table": "prices"}})
    schema = cfg.parameters()
    assert schema == {"price_by_product": {"prices": {"type": "table", "schema": {"product": "str", "rate": "float"}}}}
    assert schema.defaults() == {}
    node_schema = schema.json_schema()["properties"]["price_by_product"]
    assert node_schema["required"] == ["prices"] and node_schema["properties"]["prices"]["type"] == "array"


def test_inline_table_rows_are_not_a_param():
    cfg = PriceTable.model_validate({"name": "p", "rows": [{"product": "a", "rate": 1.0}]})
    assert cfg.parameters() == {}


def test_required_params_are_left_out_of_defaults():
    @step
    def scaled(x: float, k: float = param(required=True), j: float = param(1.0)) -> float:
        return x * k * j

    schema = scaled.parameters()
    assert schema["scaled"]["k"] == {"type": "float", "required": True}
    assert schema.defaults() == {"scaled": {"j": 1.0}}


# --- caching ---


def test_to_ir_is_cached_for_an_unchanged_step():
    p = flow(ratio, affordable, name="p")
    assert engine.to_ir(p) is engine.to_ir(p)


def test_rebuilding_a_pipeline_reuses_unchanged_members():
    a = step(ratio)
    first = engine.to_ir(flow(a, affordable))
    second = engine.to_ir(flow(a, affordable))
    assert first is not second
    assert first.children()[0] is second.children()[0]


def test_the_cache_forgets_dead_steps():
    s = step(ratio)
    engine.to_ir(s)
    key = id(s)
    assert key in _BUILT
    del s
    gc.collect()
    assert key not in _BUILT


def test_a_branch_arm_placed_in_two_branches_gets_two_nodes():
    def is_private(sector_code: int) -> bool:
        return sector_code == 1

    arm = cap_by_income
    one = branch(is_private, arm, cap_again, modifies=["term_cap"], name="one")
    two = branch(is_private, arm, cap_again, modifies=["term_cap"], name="two")
    node = engine.to_ir(flow(one, two))
    assert [n.origin.path for n in iter_nodes(node) if n.origin.path.endswith("cap_by_income")] == [
        "one/cap_by_income", "two/cap_by_income"
    ]
