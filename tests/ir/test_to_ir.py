from __future__ import annotations

import gc

import pytest

from decider import branch, dag, engine, flow, loop, param, step
from decider.engine.ir.context import _BUILT, IRContext
from decider.engine.ir.decls import Input, Output, ParamDecl
from decider.engine.ir.nodes import CallNode, SequenceNode, iter_nodes
from decider.serializable.dataframe import DataFrame
from decider.steps import ConfigurableStep, ParamRef, Step, TableRef, TableValue, Value


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
        return CallNode(ctx.origin(self), "row", lambda row, params, consts: (0.0,), (), (Output("rate", float),), params)


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
    assert (decl.name, decl.required, decl.schema) == ("prices", True, (("product", "str"), ("rate", "float")))


def test_a_table_param_reports_its_schema():
    cfg = PriceTable.model_validate({"name": "price_by_product", "rows": {"table": "prices"}})
    schema = cfg.parameters()
    assert schema == {"price_by_product": {"prices": {"type": "table", "schema": {"product": "str", "rate": "float"},
                                                      "required": True}}}
    assert schema.defaults() == {"price_by_product": {"prices": []}}
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


# --- literal config values ---


class Cut(ConfigurableStep):
    cut: Value[float] = 0.5

    def to_ir(self, ctx):
        cut = ctx.value(self.cut, float)
        params, consts = ((cut,), ()) if isinstance(cut, ParamDecl) else ((), (("cut", cut),))
        return CallNode(ctx.origin(self), "row", _above, (Input("x", float),), (Output(self.name, bool),), params,
                        consts=consts)


def _above(row, params, consts):
    return (row[0] > (params or consts)[0],)


def test_a_literal_value_is_a_const_not_a_param():
    node = engine.to_ir(Cut(name="c", cut=0.9))
    assert node.consts == (("cut", 0.9),) and node.params == ()
    assert Cut(name="c", cut=0.9).parameters() == {}
    assert node.fn((1.0,), (), tuple(v for _, v in node.consts)) == (True,)


def test_a_literal_in_params_is_rejected():
    class Bad(ConfigurableStep):
        def to_ir(self, ctx):
            return CallNode(ctx.origin(self), "row", _above, (), (Output(self.name, bool),), (0.5,))

    with pytest.raises(TypeError, match="not a ParamDecl.*consts"):
        engine.to_ir(Bad(name="bad"))


def test_param_decls_hash_even_with_unhashable_defaults():
    decl = ParamDecl("tiers", list, [1, 2], schema=(("product", "str"),))
    assert hash(decl) == hash(ParamDecl("tiers", list, [3], schema=(("product", "str"),)))
    assert {IRContext().table(TableRef(table="prices"), {"product": "str"})}


# --- step names ---


@pytest.mark.parametrize("bad", ["", "a/b", "a#b"])
def test_a_name_that_breaks_paths_is_rejected(bad):
    with pytest.raises(ValueError, match="without '/' or '#'"):
        step(ratio, name=bad)
    with pytest.raises(ValueError, match="without '/' or '#'"):
        step(ratio).named(bad)
    with pytest.raises(ValueError, match="without '/' or '#'"):
        flow(ratio, name=bad)
    with pytest.raises(ValueError, match="without '/' or '#'"):
        engine.to_ir(Cut(name=bad))


# --- configs built from other steps ---


class Afford(ConfigurableStep):
    def to_ir(self, ctx):
        return ctx.expand(self, flow(ratio, affordable))


def test_a_config_expanding_helper_steps_owns_their_path():
    node = engine.to_ir(flow(Afford(name="cfg"), name="outer"))
    cfg = node.children()[0]
    assert (cfg.origin.path, cfg.origin.source) == ("outer/cfg", f"{__name__}:Afford")
    assert [n.origin.path for n in cfg.children()] == ["outer/cfg/ratio", "outer/cfg/affordable"]
    assert isinstance(engine.step_map(flow(Afford(name="cfg"), name="outer"))["outer/cfg"], Afford)


# --- path -> step map ---


def test_the_step_map_covers_conditions_arms_and_loop_bodies():
    def is_private(sector_code: int) -> bool:
        return sector_code == 1

    def more(term_cap: float) -> bool:
        return term_cap < 10

    by_sector = branch(is_private, cap_by_income, cap_again, modifies=["term_cap"], name="by_sector")
    grow = loop(more, cap_again, carries=["term_cap"], max_iterations=3, name="grow")
    p = flow(by_sector, grow, name="p")
    engine.to_ir(p)  # cached subtrees must still report their steps
    steps = engine.step_map(p)
    assert steps["p"] is p and steps["p/by_sector"] is by_sector and steps["p/grow"] is grow
    assert steps["p/by_sector/is_private"].fn is is_private
    assert steps["p/by_sector/cap_by_income"] is cap_by_income
    assert steps["p/grow/more"].fn is more and steps["p/grow/cap_again"] is cap_again
    assert set(steps) == {n.origin.path for n in iter_nodes(engine.to_ir(p))}
