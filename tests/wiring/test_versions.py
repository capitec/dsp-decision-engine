"""Version chains, the waterfall, input columns and outputs."""
from __future__ import annotations

from decider import dag, flow, param, step
from decider.engine.wiring import Call, resolve


def _term_cap_step(name: str, delta: float):
    def fn(term_cap: float) -> float:
        return term_cap - delta

    fn.__name__ = name
    return step(output="term_cap")(fn)


def _producers(plan, name):
    return [v.producer for v in plan.chains[name]]


def test_a_waterfall_keeps_every_write_as_a_version():
    plan = resolve(flow(_term_cap_step("seed", 0), _term_cap_step("income", 1), _term_cap_step("sector", 2)))
    assert _producers(plan, "term_cap") == ["seed", "income", "sector"]
    assert [i.name for i in plan.inputs] == ["term_cap"]


def test_each_node_reads_the_latest_version():
    plan = resolve(flow(_term_cap_step("seed", 0), _term_cap_step("income", 1), _term_cap_step("sector", 2)))
    seed, income, sector = plan.calls
    assert seed.reads[0].producer is None
    assert income.reads[0] is seed.writes[0]
    assert sector.reads[0] is income.writes[0]


def test_the_final_version_is_the_output():
    plan = resolve(flow(_term_cap_step("seed", 0), _term_cap_step("income", 1)))
    assert plan.outputs["term_cap"].producer == "income"


def test_a_consumed_intermediate_is_not_an_output_a_value_nothing_reads_is():
    def a(x: float) -> float:
        return x + 1

    def b(a: float) -> float:
        return a * 2

    def c(x: float) -> float:
        return x - 1

    plan = resolve(flow(a, b, c))
    assert set(plan.outputs) == {"x", "b", "c"}


def test_input_columns_pass_through_to_the_output():
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def affordability_ratio(disposable_income: float, instalment: float) -> float:
        return disposable_income / instalment

    def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0)) -> float:
        return min(term_cap, cap) if min_net_salary < 5000 else term_cap

    plan = resolve(flow(disposable_income, affordability_ratio, cap_by_income_band))
    assert [i.name for i in plan.inputs] == ["net_income", "expenses", "instalment", "term_cap", "min_net_salary"]
    assert set(plan.outputs) == {
        "net_income", "expenses", "instalment", "term_cap", "min_net_salary", "affordability_ratio",
        "cap_by_income_band",
    }


def test_a_step_may_narrow_a_value_it_also_reads():
    @step(output="term_cap")
    def product_ceiling(requested_term: float, ceiling: float = param(60.0)) -> float:
        return min(requested_term, ceiling)

    @step(output="term_cap")
    def cap_by_income(term_cap: float, min_net_salary: float, cap: float = param(48.0)) -> float:
        return min(term_cap, cap) if min_net_salary < 5000 else term_cap

    plan = resolve(flow(product_ceiling, cap_by_income))
    assert _producers(plan, "term_cap") == ["product_ceiling", "cap_by_income"]
    assert [i.name for i in plan.inputs] == ["requested_term", "min_net_salary"]


def test_a_single_self_read_is_seeded_from_the_input_column():
    @step(output="term_cap")
    def cap_by_income_band(term_cap: float, cap: float = param(48.0)) -> float:
        return min(term_cap, cap)

    plan = resolve(cap_by_income_band)
    assert [i.name for i in plan.inputs] == ["term_cap"]
    assert plan.outputs["term_cap"].producer == "cap_by_income_band"


def test_an_unknown_name_becomes_an_input_column_with_its_declaration():
    def f(a: float, b: int | None) -> float:
        return a

    plan = resolve(f)
    assert [(i.name, i.annotation) for i in plan.inputs] == [("a", float), ("b", int | None)]
    assert plan.inputs[1].null_policy.value == "optional"
    assert all(v.producer is None for v in plan.calls[0].reads)


def test_relabelled_reads_become_the_input_column_names():
    def f(net_income: float) -> float:
        return net_income

    plan = resolve(step(f).relabel(reads={"net_income": "monthly_net_salary"}))
    assert [i.name for i in plan.inputs] == ["monthly_net_salary"]


def test_a_dag_member_reads_its_sibling_by_dependency():
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def ratio(disposable_income: float, instalment: float) -> float:
        return disposable_income / instalment

    plan = resolve(dag(ratio, disposable_income, name="m"))
    first, second = plan.calls
    assert second.reads[0] is first.writes[0]
    assert "instalment" in [i.name for i in plan.inputs]


def test_calls_are_numbered_by_position_and_keep_their_origin():
    def a(x: float) -> float:
        return x

    def b(a: float) -> float:
        return a

    plan = resolve(flow(a, b, name="p"))
    assert [c.id for c in plan.calls] == [0, 1]
    assert [c.node.origin.path for c in plan.calls] == ["p/a", "p/b"]
    assert all(isinstance(c, Call) for c in plan.root.children)
    assert [v.id for v in plan.versions] == list(range(len(plan.versions)))


def test_ids_do_not_depend_on_names():
    def a(x: float) -> float:
        return x

    def b(a: float) -> float:
        return a

    one, two = resolve(flow(a, b, name="p")), resolve(flow(step(a).named("aa"), b, name="q"))
    assert [c.id for c in one.calls] == [c.id for c in two.calls]
    assert [[v.id for v in c.reads] for c in one.calls] == [[v.id for v in c.reads] for c in two.calls]
    assert [[v.id for v in c.writes] for c in one.calls] == [[v.id for v in c.writes] for c in two.calls]


def test_resolve_accepts_a_built_ir():
    from decider import engine

    def a(x: float) -> float:
        return x

    plan = resolve(engine.to_ir(step(a)))
    assert plan.calls[0].node.origin.path == "a"


def test_a_legitimate_name_resembling_its_own_output_is_no_typo():
    def term_cap_a(term_cap: float) -> float:
        return term_cap - 1.0

    plan = resolve(term_cap_a)
    assert [i.name for i in plan.inputs] == ["term_cap"]
