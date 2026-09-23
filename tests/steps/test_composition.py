from __future__ import annotations

import pytest

from decider import branch, dag, engine, flow, frame_step, loop, missing_as, param, step
from decider.engine.ir.nodes import BranchNode, CallNode, LoopNode, SequenceNode
from decider.steps import DagStep, FunctionStep, SequentialStep


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float) -> bool:
    return ratio >= 0.3


def _term_cap_rule(fn_name: str, delta: float) -> FunctionStep:
    def fn(term_cap: float) -> float:
        return term_cap - delta

    fn.__name__ = fn.__qualname__ = fn_name
    return step(output="term_cap")(fn)


def _paths(node) -> list[str]:
    return [c.origin.path for c in node.children()]


# --- dag ---


def test_dag_of_one_step_returns_that_step():
    s = step(ratio)
    assert dag(s) is s
    assert type(dag(ratio)) is FunctionStep and dag(ratio).fn is ratio


def test_dag_sorts_members_by_dependency():
    d = dag(affordable, ratio, disposable_income, name="afford")
    assert [s.name for s in d.steps] == ["affordable", "ratio", "disposable_income"]
    assert _paths(engine.to_ir(d)) == ["afford/disposable_income", "afford/ratio", "afford/affordable"]


def test_dag_keeps_written_order_for_independent_members():
    def a(x: float) -> float:
        return x

    def b(y: float) -> float:
        return y

    assert _paths(engine.to_ir(dag(b, a))) == ["b", "a"]


def test_dag_with_two_writers_of_one_name_raises_suggesting_flow():
    d = dag(_term_cap_rule("cap_private", 1), _term_cap_rule("cap_public", 2), name="caps")
    with pytest.raises(ValueError, match=r"cap_private and cap_public both write 'term_cap'.*flow\("):
        engine.to_ir(d)


def test_dag_cycle_raises():
    def a(b: float) -> float:
        return b

    def b(a: float) -> float:
        return a

    with pytest.raises(ValueError, match="cycle"):
        engine.to_ir(dag(a, b))


def test_dag_sorts_composite_members_by_what_they_read_and_write():
    later = flow(affordable, name="later")
    earlier = flow(disposable_income, ratio, name="earlier")
    assert _paths(engine.to_ir(dag(later, earlier))) == ["earlier", "later"]


# --- flow and | ---


def test_plain_functions_have_no_pipe():
    with pytest.raises(TypeError):
        disposable_income | ratio


def test_function_pipe_step_works():
    p = disposable_income | step(ratio)
    assert type(p) is SequentialStep
    assert _paths(engine.to_ir(p)) == ["disposable_income", "ratio"]


def test_a_pipe_chain_is_one_flow():
    p = step(disposable_income) | ratio | affordable
    assert type(p) is SequentialStep and p.name is None
    assert _paths(engine.to_ir(p)) == ["disposable_income", "ratio", "affordable"]
    assert [path for path, _ in p.walk()] == ["disposable_income", "ratio", "affordable"]


def test_a_named_flow_stays_a_unit():
    named = flow(disposable_income, ratio, name="afford")
    p = named | affordable
    assert p.steps[0] is named and len(p.steps) == 2
    assert [path for path, _ in p.walk()] == ["afford", "afford/disposable_income", "afford/ratio", "affordable"]


def test_flow_and_pipe_agree():
    a, b = step(disposable_income), step(ratio)
    assert engine.to_ir(flow(a, b)).children_ == engine.to_ir(a | b).children_


def test_flow_merges_anonymous_flows():
    inner = flow(disposable_income, ratio).emit("disposable_income")
    ir = engine.to_ir(flow(inner, affordable))
    assert _paths(ir) == ["disposable_income", "ratio", "affordable"] and ir.emits == ("disposable_income",)


def test_flow_needs_a_step():
    with pytest.raises(ValueError):
        flow()


def test_a_waterfall_is_allowed_in_a_flow():
    rules = [_term_cap_rule(n, d) for n, d in (("seed", 0), ("income", 1), ("sector", 2))]
    node = engine.to_ir(flow(*rules, name="term"))
    assert _paths(node) == ["term/seed", "term/income", "term/sector"]
    assert all(c.outputs[0].name == "term_cap" for c in node.children())


def test_an_anonymous_dag_inside_a_flow_is_transparent():
    node = engine.to_ir(flow(dag(ratio, disposable_income), affordable, name="p"))
    assert _paths(node) == ["p/disposable_income", "p/ratio", "p/affordable"]
    assert [path for path, _ in flow(dag(ratio, disposable_income), affordable, name="p").walk()] == [
        "p", "p/ratio", "p/disposable_income", "p/affordable"
    ]


def test_emit_and_drop_return_new_flows_and_reach_the_ir():
    p = flow(disposable_income, ratio)
    emitted = p.emit("disposable_income").drop("expenses")
    assert (p.emits, p.drops) == ((), ())
    node = engine.to_ir(emitted)
    assert (node.emits, node.drops) == (("disposable_income",), ("expenses",))


def test_emit_on_a_dag():
    d = dag(disposable_income, ratio, name="d").emit("disposable_income")
    assert type(d) is DagStep and engine.to_ir(d).emits == ("disposable_income",)


def test_emits_of_a_merged_flow_carry_over():
    p = (step(disposable_income) | ratio).emit("disposable_income") | affordable
    assert engine.to_ir(p).emits == ("disposable_income",)


# --- relabel ---


def test_relabel_reads_renames_an_input_without_touching_the_original():
    s = step(disposable_income)
    relabelled = s.relabel(reads={"net_income": "monthly_net_salary"})
    assert [i.name for i in engine.to_ir(relabelled).inputs] == ["monthly_net_salary", "expenses"]
    assert [i.name for i in engine.to_ir(s).inputs] == ["net_income", "expenses"]


def test_relabel_writes_renames_an_output():
    node = engine.to_ir(step(ratio).relabel(writes={"ratio": "dti"}))
    assert [o.name for o in node.outputs] == ["dti"]


def test_relabel_on_a_flow_renames_the_boundary_and_its_internal_readers():
    p = flow(disposable_income, ratio, name="afford").relabel(
        reads={"instalment": "monthly_instalment"}, writes={"disposable_income": "spare"}
    )
    first, second = engine.to_ir(p).children()
    assert [o.name for o in first.outputs] == ["spare"]
    assert [i.name for i in second.inputs] == ["spare", "monthly_instalment"]


def test_relabel_merges_with_an_earlier_relabel():
    s = step(ratio).relabel(reads={"instalment": "a"}).relabel(writes={"ratio": "b"})
    assert (dict(s.reads), dict(s.writes)) == ({"instalment": "a"}, {"ratio": "b"})


def test_relabel_of_a_waterfall_reads_outside_then_its_own_writes():
    p = flow(_term_cap_rule("a", 1), _term_cap_rule("b", 2), name="t").relabel(
        reads={"term_cap": "requested"}, writes={"term_cap": "capped"}
    )
    a, b = engine.to_ir(p).children()
    assert [i.name for i in a.inputs] == ["requested"] and [i.name for i in b.inputs] == ["capped"]
    assert a.outputs[0].name == b.outputs[0].name == "capped"


def test_relabel_can_not_rename_frame_columns():
    @frame_step(reads=["client_id"], writes=["score"])
    def join(df):
        return df

    with pytest.raises(TypeError, match="frame"):
        engine.to_ir(join.relabel(writes={"score": "bureau"}))


# --- branch and loop ---


def is_private(sector_code: int) -> bool:
    return sector_code == 1


def band_index(ratio: float) -> int:
    return int(ratio)


def test_branch_builds_a_branch_node():
    b = branch(is_private, _term_cap_rule("cap_private", 1), _term_cap_rule("cap_public", 2),
               modifies=["term_cap"], name="by_sector")
    node = engine.to_ir(b)
    assert isinstance(node, BranchNode) and node.modifies == ("term_cap",)
    assert isinstance(node.condition, CallNode) and node.condition.origin.path == "by_sector/is_private"
    assert [a.origin.path for a in node.arms] == ["by_sector/cap_private", "by_sector/cap_public"]


def test_branch_with_an_int_condition_takes_n_arms():
    arms = [_term_cap_rule(f"band_{i}", i) for i in range(3)]
    node = engine.to_ir(branch(band_index, *arms, modifies=["term_cap"], name="by_band"))
    assert len(node.arms) == 3


def test_branch_with_a_bool_condition_needs_two_arms():
    arms = [_term_cap_rule(f"band_{i}", i) for i in range(3)]
    with pytest.raises(ValueError, match="2 arms"):
        engine.to_ir(branch(is_private, *arms, modifies=["term_cap"], name="b"))


def test_branch_names_anonymous_arms_by_position():
    arm = step(disposable_income) | ratio
    node = engine.to_ir(branch(is_private, arm, ratio, modifies=["ratio"], name="b"))
    assert [a.origin.path for a in node.arms] == ["b/arm0", "b/ratio"]
    assert isinstance(node.arms[0], SequenceNode) and _paths(node.arms[0]) == ["b/arm0/disposable_income", "b/arm0/ratio"]


def test_branch_needs_arms_modifies_and_a_name():
    with pytest.raises(ValueError, match="two arms"):
        branch(is_private, ratio, modifies=["ratio"], name="b")
    with pytest.raises(ValueError, match="modifies"):
        branch(is_private, ratio, ratio, modifies=[], name="b")
    with pytest.raises(TypeError):
        branch(is_private, ratio, ratio, modifies=["ratio"])


def test_a_condition_must_be_one_function():
    with pytest.raises(TypeError, match="condition"):
        engine.to_ir(branch(flow(is_private, band_index, name="c"), ratio, affordable, modifies=["x"], name="b"))


def keep_going(offer: float) -> bool:
    return offer < 10


def improve(offer: float) -> float:
    return offer + 1


def test_loop_builds_a_loop_node():
    node = engine.to_ir(loop(keep_going, improve, carries=["improve"], max_iterations=50, name="best"))
    assert isinstance(node, LoopNode) and (node.carries, node.max_iterations) == (("improve",), 50)
    assert (node.condition.origin.path, node.body.origin.path) == ("best/keep_going", "best/improve")


def test_loop_names_an_anonymous_body():
    node = engine.to_ir(loop(keep_going, step(improve) | ratio, carries=["x"], max_iterations=3, name="l"))
    assert node.body.origin.path == "l/body"


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_loop_needs_a_positive_int_bound(bad):
    with pytest.raises(ValueError, match="max_iterations"):
        loop(keep_going, improve, carries=["x"], max_iterations=bad, name="l")


def test_loop_needs_carries_and_a_bound():
    with pytest.raises(ValueError, match="carries"):
        loop(keep_going, improve, carries=[], max_iterations=3, name="l")
    with pytest.raises(TypeError):
        loop(keep_going, improve, carries=["x"], name="l")


# --- running ---


def test_relabel_keeps_the_argument_each_input_feeds():
    def capped(a: float, cap: float = param(1.0), *, x: float = missing_as(0.0)) -> float:
        return min(a + x, cap)

    node = engine.to_ir(step(capped).relabel(reads={"a": "amount", "x": "extra"}))
    assert [(i.name, i.arg) for i in node.inputs] == [("amount", "a"), ("extra", "x")]
    row = {"amount": 0.25, "extra": 0.5}
    args = {i.arg: row[i.name] for i in node.inputs}
    assert node.fn(**args, **dict(node.consts), **{p.name: p.default for p in node.params}) == 0.75
